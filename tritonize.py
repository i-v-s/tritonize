import torch
import ast
from ast import parse, get_source_segment, unparse
from typing import Dict, List, Tuple, Any, Iterable, Set, get_type_hints
from inspect import getclosurevars, getsource, getfullargspec, FullArgSpec


class NamedTensor:
    def __init__(self, *dimensions, need_contiguous=False, **kwargs):
        # super(NamedTensor, self).__init__(torch.zeros(1))
        self.dimensions = dimensions
        self.need_contiguous = need_contiguous


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


class Axis:
    def __init__(self, dims: Tuple[str, ...], reduced: bool):
        self.dims = dims
        self.reduced = reduced
        self.block_size = None

    def block_size_name(self):
        return '_'.join(self.dims).upper() + '_BS'

    def size_name(self):
        return '_'.join(self.dims) + '_size'

    def ast_block_size(self):
        return ast.Name(self.block_size_name(), ast.Load())

    def ast_block_size_arg(self):
        return ast.arg(self.block_size_name(), ast.Attribute(ast.Name("tl", ast.Load()), 'constexpr'))

    def ast_size(self):
        return ast.Name(self.size_name(), ast.Load())

    def ast_meta_block_size(self):
        return ast.Subscript(ast.Name('meta', ast.Load()), ast.Constant(self.block_size_name()), ast.Load())


class TensorArgument:
    def __init__(self, name, dims, axes, need_contiguous=False):
        self.name = name
        self.dims = dims
        self.axes = axes
        self.need_contiguous = need_contiguous

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

    def prepare_args(self, with_assert=True):
        if not with_assert:
            return []
        dim_len = len(self.dims)
        result = (
            [ast.Assert(
                ast.Compare(ast_len(self.ast_attr('shape')), [ast.Eq()], [ast.Constant(dim_len)]),
                ast.Constant(f'Shape of {self.name} must have len {dim_len}'))] +
            [ast.Assert(
                ast.Compare(self.ast_get_shape(i), [ast.Eq()], [ast.Constant(len(dim))]),
                ast.Constant(f'Dim {i} of {self.name} must have size {len(dim)} for fields: {", ".join(dim)}'))
             for i, dim in enumerate(self.dims)
             if isinstance(dim, list)])
        if self.need_contiguous:
            result.append(ast.Assert(
                ast.Call(
                    ast.Attribute(
                        ast.Name(self.name, ast.Load()),
                        'is_contiguous', ast.Load()),
                    [], []
                )
            ))
        return result

    def axis_indices(self, axis: Tuple[str, ...]):
        try:
            i = self.dims.index(axis[0])
            assert axis == self.dims[i:i + len(axis)]
            return range(i, i + len(axis))
        except ValueError:
            return None

    def ast_kernel_args(self):
        result = [ast.Name(self.name, ast.Load())]
        if not self.need_contiguous:
            for axis in self.axes:
                indices = self.axis_indices(axis)
                if indices is not None:
                    result.append(ast_product(*[self.ast_get_stride(i) for i in indices]))
        return result

    def kernel_args_def(self):
        result = [ast.arg(self.name + '_ptr')]
        if not self.need_contiguous:
            for axis in self.axes:
                indices = self.axis_indices(axis)
                if indices is not None:
                    result.append(ast.arg(self.name + '_' + '_'.join(axis) + '_stride'))
        return result

    def ast_calc_pointer(self):
        offsets = []
        for axis in self.axes:
            indices = self.axis_indices(axis)
            if indices is not None:
                offsets.append(ast_product(
                    ast.Name('_'.join(axis) + '_i', ast.Load()),
                    ast.Name(self.name + '_' + '_'.join(axis) + '_stride', ast.Load())
                ))
        return ast_sum(ast.Name(self.name + '_ptr', ast.Load()), *offsets)

    def ast_pointer(self, field=None):
        result = ast.Name(self.name + '_p', ast.Load())
        if field is not None:
            result = ast_sum(result, ast.Constant(self.dims[-1].index(field)))
        return result

    def ast_store(self, value, field=None):
        return ast.Call(
            ast.Attribute(ast.Name('tl', ast.Load()), 'store'),
            [self.ast_pointer(), value],
            [ast.keyword('mask', ast.Name('mask', ast.Load()))]
        )

    def ast_load(self, field=None):
        return ast.Call(
            ast.Attribute(ast.Name('tl', ast.Load()), 'load'),
            [self.ast_pointer(field)],
            [ast.keyword('mask', ast.Name('mask', ast.Load())),
             ast.keyword('other', ast.Name('other', ast.Load()))]
        )


def ast_grid_dim(axis: Axis):
    return ast.Call(
        ast.Attribute(ast.Name('triton', ast.Load()), 'cdiv', ast.Load()),
        [
            axis.ast_size(),
            axis.ast_meta_block_size()
        ],
        []
    )


def make_grid_lambda(axes):
    return ast.Lambda(
        ast.arguments([], [ast.arg('meta')], None, [], [], None, []),
        ast.Tuple([
            ast_product(*[
                ast_grid_dim(a)
                for a in axes.values()
                if not a.reduced
            ])
        ], ast.Load()))


def ast_get_axis_shapes(axis: Tuple[str, ...], args: Iterable[TensorArgument]):
    result = []
    for arg in args:
        indices = arg.axis_indices(axis)
        if indices is not None:
            result.append(ast_product(*[
                arg.ast_get_shape(j)
                for j in indices
            ]))
    return result


def make_wrapper(parsed_func: ast.FunctionDef, axes, tensor_args: Dict[str, TensorArgument], with_assert=True):
    parsed_args = parsed_func.args
    body = [
        ast.Assert(ast.BoolOp(ast.And(),
                              [ast.Attribute(ast.Name(a, ast.Load()), "is_cuda", ast.Load())
                               for a in tensor_args.keys()]))
    ] if with_assert else []
    for ta in tensor_args.values():
        body.extend(ta.prepare_args(with_assert))
    kernel_args = []
    for a in parsed_args.args:
        ta = tensor_args.get(a.arg, None)
        if ta is None:
            kernel_args.append(ast.Name(a.arg, ast.Load()))
        else:
            kernel_args.extend(ta.ast_kernel_args())
    for axis in axes:
        size = '_'.join(axis) + '_size'
        shapes = ast_get_axis_shapes(axis, tensor_args.values())
        body.append(ast.Assign(
            [ast.Name(size, ast.Store())],
            shapes[0]
        ))
        if with_assert and len(shapes) > 1:
            body.append(ast.Assert(ast.Compare(
                ast.Name(size, ast.Load()),
                [ast.Eq()] * (len(shapes) - 1),
                shapes[1:]
            )))
    body.extend([
        ast.Assign(
             [ast.Name('__grid__', ast.Store())],
             make_grid_lambda(axes)),
        ast.Expr(ast.Call(
            ast.Subscript(
                ast.Name(parsed_func.name + "_kernel", ctx=ast.Load()),
                ast.Name('__grid__', ast.Load()),
                ast.Load()
            ),
            kernel_args,
            [ast.keyword(a.block_size_name(), ast.Constant(a.block_size))
             for a in axes.values()]))
    ])
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
                              # [ast.Expr(ast.Call(ast.Name('print', ast.Load()), [ast.Name('bars', ast.Load())], []))],
                              body,
                              [])
    setattr(wrapper, 'lineno', parsed_func.lineno)
    ast.fix_missing_locations(wrapper)
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
                return ast.Expr(self.args[target.id].ast_store(self.generic_visit(node.value)))
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                arg = self.args.get(target.value.id, None)
                if arg is not None:
                    return ast.Expr(arg.ast_store(self.generic_visit(node.value), field=target.attr))
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


def make_kernel(parsed_func: ast.FunctionDef, args: FullArgSpec, tensor_args: Dict[str, TensorArgument], axes):
    args = []
    for a in parsed_func.args.args:
        ta = tensor_args.get(a.arg, None)
        if ta is None:
            args.append(a)
        else:
            args.extend(ta.kernel_args_def())
    for a in axes.values():
        args.append(a.ast_block_size_arg())

    replacer = TensorArgumentReplacer(tensor_args)
    init_body = (
        [
            ast.Assign(
                [ast.Name('_'.join(a) + '_i', ast.Store())],
                ast.Call(
                    ast.Attribute(ast.Name('tl', ast.Load()), 'program_id'),
                    [], [ast.keyword('axis', ast.Constant(i))]
                )
            )
            for i, a in enumerate(map(lambda i: i[0], filter(lambda i: not i[1], axes.items())))
        ] +
        [
            ast.Assign(
                [ast.Name('_'.join(a) + '_i', ast.Store())],
                ast.Call(
                    ast.Attribute(ast.Name('tl', ast.Load()), 'arange'),
                    [ast.Constant(0), ast.Name('BLOCK_SIZE_' + '_'.join(a).upper(), ast.Load())], []
                )
            )
            for a, r in axes.items()
            if r
        ] + [
            ast.Assign(
                [ast.Name(n + '_p', ast.Store())],
                ta.ast_calc_pointer()
            )
            for n, ta in tensor_args.items()
        ])
    body = [replacer.visit(node) for node in parsed_func.body]
    kernel = ast.FunctionDef(parsed_func.name + '_kernel',
                             ast.arguments([], args, [], [], [], [], []),
                             init_body + body, [])
    setattr(kernel, 'lineno', parsed_func.lineno)
    ast.fix_missing_locations(kernel)
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
    return {r: Axis(r, r in reduced) for r in result}


def tritonize(DEFAULT_BS=None, **kwargs):
    def decorator(f):
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

        axes = distribute_axes(args.annotations, reduced_axes)
        for a in axes.values():
            a.block_size = kwargs.get(a.block_size_name(), DEFAULT_BS)

        assert len([1 for r in axes.values() if not r]) <= 3
        tensor_args = {name: TensorArgument(name, tp.dimensions, axes, need_contiguous=tp.need_contiguous)
                       for name, tp in args.annotations.items()
                       if isinstance(tp, NamedTensor)}
        #block_sizes =
        wrapper = make_wrapper(parsed, axes, tensor_args)
        print('wrapper:\n', unparse(wrapper))

        kernel = make_kernel(parsed, args, tensor_args, axes)
        print('kernel:\n', unparse(kernel))

        # print(ast.dump(wrapper, indent=4))

        code = compile(ast.Module([wrapper], []), '<string>', 'exec')
        namespace = {'torch': torch}
        exec(code, namespace)
        return namespace[parsed.name]
    return decorator
