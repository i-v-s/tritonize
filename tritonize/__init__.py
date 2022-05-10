from typing import Dict, List, Tuple, Any, Iterable, Set, Optional, NamedTuple, get_type_hints
from copy import copy

import ast
from ast import parse, get_source_segment, unparse
from inspect import getclosurevars, getsource, getfullargspec, FullArgSpec
import linecache

from .utils import ast_product
from .data import NamedTensor, Axis, TensorArgument, Writer, Globals
from .tf import ReductionFinder, Inliner, replace_tensor_argument, replace_if


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


#class AttributeReplacer(ast.NodeTransformer):
#    def __init__(self, args: Dict[str, TensorArgument]):
#        self.args = args


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
        ta.init_kernel(writer)
    # replacer = TensorArgumentReplacer(tensor_args)
    body = replace_tensor_argument(parsed_func.body, tensor_args)  # [replacer.visit(node) for node in parsed_func.body]
    replace_if(writer, body)
    kernel = ast.FunctionDef(parsed_func.name + '_kernel',
                             ast.arguments([], args, None, [], [], None, []),
                             writer.body, [globs.ast_triton('jit')])
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


def tritonize(save_to: Optional[str] = None,
              anno: Optional[Dict[str, NamedTensor]] = None,
              DEFAULT_BS: Optional[int] = 128,
              **kwargs):
    anno = anno or {}

    def decorator(f):
        globs = Globals(f.__globals__)
        print('hints:', get_type_hints(f))
        args = getfullargspec(f)
        source = getsource(f)
        parsed = parse(source).body[0]
        parsed = Inliner(globs.globals).visit(parsed)
        f_anno = copy(args.annotations)
        f_anno.update(anno)
        all_dims = {
            dim
            for name, tp in f_anno.items() if isinstance(tp, NamedTensor)
            for dim in tp.dimensions if isinstance(dim, str)
        }
        reduced_axes = ReductionFinder.find_axes(parsed, all_dims)
        axes: List[Axis] = [
            Axis(a, kwargs.get(Axis.block_size_name_(a), DEFAULT_BS), globs)
            for a in distribute_axes(f_anno, reduced_axes)
        ]

        tensor_args = {name: TensorArgument(name, tp.dimensions, axes, globs, need_contiguous=tp.need_contiguous)
                       for name, tp in f_anno.items()
                       if isinstance(tp, NamedTensor)}

        kernel = make_kernel(parsed, args, globs, tensor_args, axes)
        wrapper = make_wrapper(parsed, axes, tensor_args)
        imports = [ast.Import([ast.alias(j, i)])
                   for i, j in [(globs.torch, 'torch'), (globs.triton, 'triton'), (globs.tl, 'triton.language')]
                   if i not in globs.globals]
        module = ast.Module(imports + [kernel, wrapper], [])
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
