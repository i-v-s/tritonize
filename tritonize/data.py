from typing import List, Tuple, Optional
import ast
from copy import copy

import torch
import triton
import triton.language as tl

from .utils import ast_len, ast_sum, ast_product, ast_and


class NamedTensor:
    def __init__(self, *dimensions, need_contiguous=False, **kwargs):
        # super(NamedTensor, self).__init__(torch.zeros(1))
        self.dimensions = dimensions
        self.need_contiguous = need_contiguous


class Globals:
    def __init__(self, globs):
        self.globals = copy(globs)
        self.tl = 'tl'
        self.triton = 'triton'
        self.torch = 'torch'
        for k, v in self.globals.items():
            if v is triton:
                self.triton = k
            if v is tl:
                self.tl = k
            if v is torch:
                self.torch = k

    def ast_tl(self, attr):
        return ast.Attribute(ast.Name(self.tl, ast.Load()), attr, ast.Load())

    def ast_triton(self, attr):
        return ast.Attribute(ast.Name(self.triton, ast.Load()), attr, ast.Load())

    def cdiv(self, a: ast.expr, b: ast.expr):
        return ast.Call(self.ast_triton('cdiv'), [a, b], [])

    def tl_cdiv(self, a: ast.expr, b: ast.expr):
        return ast.Call(self.ast_tl('cdiv'), [a, b], [])


class Writer:
    def __init__(self, ns=None):
        self.ns = ns or set()
        self.body = []

    def init(self, var: str, value: ast.expr) -> ast.Name:
        if var in self.ns:
            raise ValueError(f'Attempt of variable reinitialization {var}')
        self.body.append(ast.Assign([ast.Name(var, ast.Store())], value))
        self.ns.add(var)
        return ast.Name(var, ast.Load())

    def aug_assign(self, target: str, op: ast.operator, value: ast.expr):
        assert target in self.ns
        self.body.append(ast.AugAssign(ast.Name(target, ast.Store()), op, value))

    def assert_(self, *args, msg: Optional[str] = None):
        if args:
            self.body.append(
                ast.Assert(
                    ast.BoolOp(ast.And(), list(args)) if len(args) > 1 else args[0],
                    ast.Constant(msg)))

    def assert_eq(self, arg, *args, msg: Optional[str] = None):
        self.body.append(
            ast.Assert(
                ast.Compare(arg, [ast.Eq()] * len(args), list(args)),
                ast.Constant(msg)))

    def call(self, *args, **kwargs):
        self.body.append(ast.Expr(ast.Call(*args, **kwargs)))

    def write(self, node):
        self.body.append(node)


class Axis:
    def __init__(self, dims: Tuple[str, ...], block_size: int, globs: Globals, *, one_block=False, no_mask=False):
        self.dims = dims
        self.name = '_'.join(dims)
        self.block_size = block_size
        self.g = globs
        # Flags:
        self.one_block = one_block
        self.no_mask = no_mask
        # Initialized vars:
        self.index: Optional[ast.Name] = None
        self.mask:  Optional[ast.Name] = None

    def __str__(self):
        return self.name

    @staticmethod
    def block_size_name_(dims: Tuple[str, ...]) -> str:
        return '_'.join(dims).upper() + '_BS'

    def block_size_name(self) -> str:
        return Axis.block_size_name_(self.dims)

    def size_name(self):
        return '_'.join(self.dims) + '_size'

    def ast_block_size(self) -> ast.Name:
        return ast.Name(self.block_size_name(), ast.Load())

    def ast_block_size_arg(self):
        return ast.arg(self.block_size_name(), self.g.ast_tl('constexpr'))

    def ast_size(self) -> ast.Name:
        return ast.Name(self.size_name(), ast.Load())

    def ast_meta_block_size(self):
        return ast.Subscript(ast.Name('meta', ast.Load()), ast.Constant(self.block_size_name()), ast.Load())

    def init_kernel(self, writer: Writer, pid: str):
        name = '_'.join(self.dims)
        if self.one_block:
            self.index = writer.init(
                name + '_i',
                ast.Call(self.g.ast_tl('arange'), [ast.Constant(0), self.ast_block_size()], []))
        else:
            blocks = writer.init(self.name + '_blocks', self.g.tl_cdiv(self.ast_size(), self.ast_block_size()))
            block_index = ast.BinOp(ast.Name(pid, ast.Load()), ast.Mod(), blocks)
            self.index = writer.init(
                name + '_i',
                ast.BinOp(
                    ast.BinOp(block_index, ast.Mult(), self.ast_block_size()),
                    ast.Add(),
                    ast.Call(self.g.ast_tl('arange'), [ast.Constant(0), self.ast_block_size()], [])))
            writer.aug_assign(pid, ast.FloorDiv(), blocks)
        if not self.no_mask:
            self.mask = writer.init(
                name + '_m',
                ast.Compare(self.index, [ast.Lt()], [self.ast_size()]))

    def ast_grid_dim(self):
        return self.g.cdiv(self.ast_size(), self.ast_meta_block_size())


class TensorArgument:
    def __init__(self, name, dims, axes: List[Axis], globs: Globals, need_contiguous=False):
        self.name = name
        self.dims = dims
        self.axes = self.choose_axes(axes)
        self.g = globs
        self.field_map = {
            field: (i, j)
            for i, dim in enumerate(dims)
            if isinstance(dim, list)
            for j, field in enumerate(dim)
        }
        # Flags:
        self.need_contiguous = need_contiguous
        # Initialized vars:
        self.value_p = None
        self.mask = None

    def choose_axes(self, axes: List[Axis]):
        buf = ()
        axes = {a.dims: a for a in axes}
        result = []
        for dim in self.dims:
            if isinstance(dim, list):
                assert not buf
            else:
                buf += (dim,)
                axis = axes.get(buf, None)
                if axis is not None:
                    result.append(axis)
                    buf = ()
        assert not buf
        return result

    def ast_name(self):
        return ast.Name(self.name, ast.Load())

    def ast_attr(self, attr):
        return ast.Attribute(self.ast_name(), attr, ast.Load())

    def ast_get_shape(self, i):
        return ast.Subscript(
            self.ast_attr('shape'),
            ast.Constant(i), ast.Load())

    def ast_get_stride(self, i):
        return ast.Call(
            self.ast_attr('stride'),
            [ast.Constant(i)], [])

    def prepare_args(self, writer: Writer, with_assert=True):
        if not with_assert:
            return
        dim_len = len(self.dims)
        writer.assert_eq(
            ast_len(self.ast_attr('shape')), ast.Constant(dim_len),
            msg=f'Shape of {self.name} must have len {dim_len}')
        for i, dim in enumerate(self.dims):
            if isinstance(dim, list):
                writer.assert_eq(
                    self.ast_get_shape(i), ast.Constant(len(dim)),
                    msg=f'Dim {i} of {self.name} must have size {len(dim)} for fields: {", ".join(dim)}')
        if self.need_contiguous:
            writer.assert_(
                ast.Call(
                    ast.Attribute(
                        ast.Name(self.name, ast.Load()),
                        'is_contiguous', ast.Load()),
                    [], []
                ),
                msg=f'{self.name} must be contiguous'
            )

    def axis_ranges(self):
        i = 0
        for axis in self.axes:
            j = i + len(axis.dims)
            yield axis, range(i, j)
            i = j

    def ast_kernel_args(self):
        result = [ast.Name(self.name, ast.Load())]
        if not self.need_contiguous:
            for axis, r in self.axis_ranges():
                result.append(self.ast_get_stride(r[-1]))
            for i, d in enumerate(self.dims):
                if isinstance(d, list):
                    result.append(self.ast_get_stride(i))
        return result

    def kernel_args_def(self):
        result = [ast.arg(self.name + '_ptr')]
        if not self.need_contiguous:
            for axis in self.axes:
                result.append(ast.arg(f'{self.name}_{axis.name}_stride'))
            for i, d in enumerate(self.dims):
                if isinstance(d, list):
                    result.append(ast.arg(f'{self.name}_{i}_stride'))
        return result

    def init_kernel(self, writer):
        offsets = []
        for axis in self.axes:
            offsets.append(ast_product(
                ast.Name(f'{axis}_i', ast.Load()),
                ast.Name(f'{self.name}_{axis}_stride', ast.Load())
            ))
        self.value_p = writer.init(self.name + '_p', ast_sum(ast.Name(self.name + '_ptr', ast.Load()), *offsets))
        masks = [a.mask for a in self.axes if a.mask is not None]
        if masks:
            self.mask = writer.init(f'{self.name}_m', ast_product(*masks)) if len(masks) > 1 else masks[0]

    def ast_pointer(self, fields=None):
        result = self.value_p
        for field in fields:
            if isinstance(field, ast.Slice):
                assert field.lower is None and field.upper is None, 'Slicing not implemented'
                continue
            i, j = self.field_map[field]
            if j > 0:
                result = ast_sum(result, ast.BinOp(
                    ast.Constant(j),
                    ast.Mult(),
                    ast.Name(f'{self.name}_{i}_stride', ast.Load())))
        return result

    def ast_store(self, value, fields=None, mask=None):
        kwargs = {}
        masks = [m for m in [self.mask, mask] if m is not None]
        if self.mask:
            kwargs['mask'] = ast_and(*masks)
        return ast.Call(
            self.g.ast_tl('store'),
            [self.ast_pointer(fields), value],
            [ast.keyword(k, v) for k, v in kwargs.items()]
        )

    def ast_load(self, fields=None, mask=None):
        kwargs = {}
        masks = [m for m in [self.mask, mask] if m is not None]
        if masks:
            kwargs['mask'] = ast_and(*masks)
            # kwargs.append(ast.keyword('other', ast.Name('other', ast.Load())))
        return ast.Call(
            self.g.ast_tl('load'),
            [self.ast_pointer(fields)],
            [ast.keyword(k, v) for k, v in kwargs.items()]
        )
