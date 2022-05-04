from typing import Dict, List, Tuple, Any, Iterable, Set, Optional, NamedTuple, get_type_hints
from copy import copy

import torch
import triton
import triton.language as tl

import ast
from ast import parse, get_source_segment, unparse
from inspect import getclosurevars, getsource, getfullargspec, FullArgSpec
import linecache


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


def ast_bin_op(a1, *args, op=None):
    if args:
        a2, *other = args
        return ast_bin_op(ast.BinOp(a1, op, a2), *other, op=op)
    else:
        return a1


def ast_product(*factors):
    return ast_bin_op(*factors, op=ast.Mult())


def ast_sum(*terms):
    return ast_bin_op(*terms, op=ast.Add())


def ast_len(c):
    return ast.Call(ast.Name('len', ast.Load()), [c], [])


class AxisVars(NamedTuple):
    index: ast.AST
    mask: ast.AST


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
        self.need_contiguous = need_contiguous

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
        return result

    def kernel_args_def(self):
        result = [ast.arg(self.name + '_ptr')]
        if not self.need_contiguous:
            for axis in self.axes:
                result.append(ast.arg(self.name + '_' + axis.name + '_stride'))
        return result

    def ast_calc_pointer(self):
        offsets = []
        for axis in self.axes:
            offsets.append(ast_product(
                ast.Name(f'{axis}_i', ast.Load()),
                ast.Name(f'{self.name}_{axis}_stride', ast.Load())
            ))
        return ast_sum(ast.Name(self.name + '_ptr', ast.Load()), *offsets)

    def ast_pointer(self, field=None):
        result = ast.Name(self.name + '_p', ast.Load())
        if field is not None:
            result = ast_sum(result, ast.Constant(self.dims[-1].index(field)))
        return result

    def ast_store(self, value, field=None):
        kwargs = {}

        return ast.Call(
            self.g.ast_tl('store'),
            [self.ast_pointer(), value],
            [ast.keyword(k, v) for k, v in kwargs.items()]
        )

    def ast_load(self, field=None):
        kwargs = {}
        # if self.mask is not None:
            # kwargs.append(ast.keyword('mask', ast.Name('mask', ast.Load())))
            # kwargs.append(ast.keyword('other', ast.Name('other', ast.Load())))
        return ast.Call(
            self.g.ast_tl('load'),
            [self.ast_pointer(field)],
            [ast.keyword(k, v) for k, v in kwargs.items()]
        )


def make_grid_lambda(axes: Iterable[Axis]):
    return ast.Lambda(
        ast.arguments([], [ast.arg('meta')], None, [], [], None, []),
        ast.Tuple([
            ast_product(*[
                a.ast_grid_dim()
                for a in axes
                if not a.one_block
            ])
        ], ast.Load()))


def ast_get_axes_shapes(axes: Iterable[Axis], args: Iterable[TensorArgument]):
    result = {a: [] for a in axes}
    for arg in args:
        for axis, indices in arg.axis_ranges():
            result[axis].append(ast_product(*map(arg.ast_get_shape, indices)))
    return result


def make_wrapper(parsed_func: ast.FunctionDef, axes, tensor_args: Dict[str, TensorArgument], with_assert=True):
    parsed_args = parsed_func.args
    writer = Writer()
    if with_assert:
        writer.assert_(
            *[ast.Attribute(ast.Name(a, ast.Load()), "is_cuda", ast.Load()) for a in tensor_args.keys()],
            msg='All tensors must be stored in GPU')
    for ta in tensor_args.values():
        ta.prepare_args(writer, with_assert)
    kernel_args = []
    for a in parsed_args.args:
        ta = tensor_args.get(a.arg, None)
        if ta is None:
            kernel_args.append(ast.Name(a.arg, ast.Load()))
        else:
            kernel_args.extend(ta.ast_kernel_args())
    for axis, shapes in ast_get_axes_shapes(axes, tensor_args.values()).items():
        size = writer.init(axis.size_name(), shapes[0])
        if with_assert and len(shapes) > 1:
            writer.assert_eq(size, *shapes[1:], msg=f'{size.id} not equal on tensors')
        kernel_args.append(size)
    grid = writer.init('_grid', make_grid_lambda(axes))
    writer.call(
        ast.Subscript(
            ast.Name(parsed_func.name + "_kernel", ctx=ast.Load()),
            grid,
            ast.Load()
        ),
        kernel_args,
        [ast.keyword(a.block_size_name(), ast.Constant(a.block_size))
         for a in axes]
    )
    wrapper = ast.FunctionDef(parsed_func.name,
                              ast.arguments(
                                  parsed_args.posonlyargs,
                                  [ast.arg(a.arg, ast.Attribute(ast.Name("torch", ast.Load()), 'Tensor', ast.Load()))
                                   if a.arg in tensor_args else a
                                   for a in parsed_args.args],
                                  parsed_args.vararg,
                                  parsed_args.kwonlyargs,
                                  parsed_args.kw_defaults,
                                  parsed_args.kwarg,
                                  parsed_args.defaults,
                                ),
                              writer.body,
                              [])
    return wrapper


class TensorArgumentReplacer(ast.NodeTransformer):
    def __init__(self, args: Dict[str, TensorArgument]):
        self.args = args

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = node.value
        if isinstance(value, ast.Name) and value.id in self.args:
            arg = self.args[value.id]
            if isinstance(node.ctx, ast.Load):
                return arg.ast_load(node.attr)
            else:
                raise TypeError('Unexpected context')
        else:
            return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        arg = self.args.get(node.id, None)
        if arg is None:
            return self.generic_visit(node)
        elif isinstance(node.ctx, ast.Load):
            return arg.ast_load()
        else:
            raise TypeError('Unexpected context')

    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = node.targets
        if len(targets) == 1:
            target = targets[0]
            if isinstance(target, ast.Name) and target.id in self.args:
                return ast.Expr(self.args[target.id].ast_store(self.visit(node.value)))
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                arg = self.args.get(target.value.id, None)
                if arg is not None:
                    return ast.Expr(arg.ast_store(self.visit(node.value), field=target.attr))
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        target = node.target
        if isinstance(target, ast.Name):
            arg = self.args.get(target.id, None)
            if arg is not None:
                return ast.Expr(arg.ast_store(ast.BinOp(arg.ast_load(), node.op, self.generic_visit(node.value))))
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            arg = self.args.get(target.value.id, None)
            if arg is not None:
                field = target.attr
                return ast.Expr(arg.ast_store(
                    ast.BinOp(arg.ast_load(), node.op, self.generic_visit(node.value), field=field),
                    field=field))
        return self.generic_visit(node)


class ReductionFinder(ast.NodeVisitor):
    def __init__(self, all_dims):
        self.axes = set()
        self.all_dims = all_dims

    def visit_Call(self, node: ast.Call) -> Any:
        if (
                isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'tl' and node.func.attr in {'sum', 'max', 'min'}
        ):
            axis = [kw.value for kw in node.keywords if kw.arg == 'axis']
            if axis and isinstance(axis[0], ast.Constant):
                axis = axis[0].value
                if axis not in self.all_dims:
                    raise ValueError('Unknown axis: ' + axis)
                self.axes.add((axis,))
            else:
                raise ValueError('Axis not specified')

    @staticmethod
    def find_axes(parsed: ast.FunctionDef, all_dims: Set[str]):
        rf = ReductionFinder(all_dims)
        for node in parsed.body:
            rf.visit(node)
        return rf.axes


def trace_variables(body: List[Any], args):
    local = {}
    args.append(local)
    for item in body:
        if isinstance(item, ast.If):
            ...
            # tracer = ExprTracer()
            # item.test = tracer.visit(item.test)
            # item.body = \
            trace_variables(item.body, args)
        elif isinstance(item, ast.Assign):
            tracer = ExprTracer(args)
            # item.value = \
            tracer.visit(item.value)
        elif isinstance(item, ast.AugAssign):
            tracer = ExprTracer(args)
            # item.value =\
            tracer.visit(item.value)
        else:
            raise NotImplementedError('Unknown AST type: ' + str(item))
    return args.pop()
    # return result


def make_kernel(parsed_func: ast.FunctionDef, args: FullArgSpec, globs: Globals,
                tensor_args: Dict[str, TensorArgument], axes: List[Axis]):
    args = []
    for a in parsed_func.args.args:
        ta = tensor_args.get(a.arg, None)
        if ta is None:
            args.append(a)
        else:
            args.extend(ta.kernel_args_def())
    for a in axes:
        args.append(ast.arg(a.size_name()))
    for a in axes:
        args.append(a.ast_block_size_arg())

    writer = Writer()
    pid = writer.init('pid', ast.Call(globs.ast_tl('program_id'), [], [ast.keyword('axis', ast.Constant(0))]))
    for a in axes:
        a.init_kernel(writer, pid.id)

    for n, ta in tensor_args.items():
        writer.init(n + '_p', ta.ast_calc_pointer())
    replacer = TensorArgumentReplacer(tensor_args)
    body = [replacer.visit(node) for node in parsed_func.body]
    kernel = ast.FunctionDef(parsed_func.name + '_kernel',
                             ast.arguments([], args, None, [], [], None, []),
                             writer.body + body, [globs.ast_triton('jit')])
    return kernel


def distribute_axes(annotations, reduced) -> Dict[Tuple[str, ...], Axis]:
    dims = [tp.dimensions for name, tp in annotations.items() if isinstance(tp, NamedTensor)]
    seqs = set()
    for d in dims:
        r = ()
        for i in d:
            if isinstance(i, str):
                r += i,
            elif r:
                seqs.add(r)
                r = ()
        if r:
            seqs.add(r)
    for r in reduced:
        seqs.add(r)
    seq_map = {}
    for s in seqs:
        l = len(s) - 1
        for i, w in enumerate(s):
            sm = seq_map.get(w, None)
            pre = s[i - 1] if i > 0 else None
            nex = s[i + 1] if i < l else None
            if sm is None:
                seq_map[w] = [pre, nex]
            else:
                p, n = sm
                if p != pre:
                    sm[0] = None
                if n != nex:
                    sm[1] = None
    result = []
    for d, (p, n) in seq_map.items():
        if p is None:
            item = [d]
            while n is not None:
                item.append(n)
                n = seq_map[n][1]
            result.append(tuple(item))
    return result


def patch_cache(name, code):
    getlines = linecache.getlines

    def monkey_patch(filename, module_globals=None):
        if filename == name:
            return code.splitlines(keepends=True)
        else:
            return getlines(filename, module_globals)
    linecache.getlines = monkey_patch


def tritonize(save_to: Optional[str] = None, DEFAULT_BS=None, **kwargs):
    def decorator(f):
        globs = Globals(f.__globals__)
        print('hints:', get_type_hints(f))
        args = getfullargspec(f)
        source = getsource(f)
        parsed = parse(source).body[0]
        all_dims = {
            dim
            for name, tp in args.annotations.items() if isinstance(tp, NamedTensor)
            for dim in tp.dimensions if isinstance(dim, str)
        }
        reduced_axes = ReductionFinder.find_axes(parsed, all_dims)

        axes: List[Axis] = [
            Axis(a, kwargs.get(Axis.block_size_name_(a), DEFAULT_BS), globs)
            for a in distribute_axes(args.annotations, reduced_axes)
        ]

        tensor_args = {name: TensorArgument(name, tp.dimensions, axes, globs, need_contiguous=tp.need_contiguous)
                       for name, tp in args.annotations.items()
                       if isinstance(tp, NamedTensor)}

        kernel = make_kernel(parsed, args, globs, tensor_args, axes)
        wrapper = make_wrapper(parsed, axes, tensor_args)
        module = ast.Module([kernel, wrapper], [])
        ast.fix_missing_locations(module)

        code = ast.unparse(module)
        print(code)

        #print(ast.dump(wrapper, indent=4))
        if save_to is not None:
            module_file = save_to
            with open(save_to, 'w') as file:
                imports = ast.Module([
                    ast.Import([ast.alias('torch', globs.torch)]),
                    ast.Import([ast.alias('triton', globs.triton)]),
                    ast.Import([ast.alias('triton.language', globs.tl)])
                ], [])
                file.write(ast.unparse(imports))
                file.write('\n\n\n')
                file.write(code)
                file.write('\n')
        else:
            module_file = f'<{parsed.name}_kernel>'
            patch_cache(module_file, code)
        compiled = compile(module, module_file, 'exec')
        exec(compiled, globs.globals)
        return globs.globals[parsed.name]
    return decorator
