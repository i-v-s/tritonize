from typing import Dict, List, Tuple, Any, Iterable, Set, Optional, Union
from copy import copy

import ast
from ast import parse, get_source_segment, unparse
from inspect import getclosurevars, getsource, getfullargspec, FullArgSpec
import linecache

from .utils import ast_product, build_seq, scan_vars
from .data import NamedTensor, Axis, TensorArgument, Writer, Context
from .tf import Tritonize, Inliner, ReductionFinder


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
        ta.prepare(writer, with_assert)
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
        axis.check_size(writer, size)
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


def make_kernel(parsed_func: ast.FunctionDef, globs: Context,
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
    body = Tritonize(globs, tensor_args, axes).visit(parsed_func).body
    body = writer.body + body
    kernel = ast.FunctionDef(parsed_func.name + '_kernel',
                             ast.arguments([], args, None, [], [], None, []),
                             body, [globs.ast_triton('jit')])
    return kernel


def distribute_axes(annotations, reduced) -> List[Tuple[str, ...]]:
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
    return build_seq(seqs)


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
              one_block: Optional[List[Union[str, Tuple[str, ...]]]] = None,
              no_mask: Optional[List[Union[str, Tuple[str, ...]]]] = None,
              print_inlined: bool = False, print_result: bool = False,
              **kwargs):
    anno = anno or {}
    one_block = set(d if isinstance(d, tuple) else (d,) for d in one_block or [])
    no_mask = set(d if isinstance(d, tuple) else (d,) for d in no_mask or [])

    def decorator(f):
        args = getfullargspec(f)
        source = getsource(f)
        parsed = parse(source).body[0]
        assert isinstance(parsed, ast.FunctionDef)
        ctx = Context(f.__globals__, scan_vars(parsed))
        parsed = Inliner(ctx).visit(parsed)
        if print_inlined:
            ast.fix_missing_locations(parsed)
            print('\n\n### Inlined:')
            print(ast.unparse(parsed))
        f_anno = copy(args.annotations)
        f_anno.update({k: v for k, v in anno.items() if k in args.args})
        all_dims = {
            dim
            for name, tp in f_anno.items() if isinstance(tp, NamedTensor)
            for dim in tp.dimensions if isinstance(dim, str)
        }
        reduced_axes = ReductionFinder(all_dims).find_axes(parsed, all_dims)
        axes: List[Axis] = [
            Axis(a, kwargs.get(Axis.block_size_name_(a), DEFAULT_BS), ctx,
                 one_block=a in one_block,
                 no_mask=a in no_mask)
            for a in distribute_axes(f_anno, reduced_axes)
        ]

        tensor_args = {name: TensorArgument(name, tp.dimensions, axes, ctx, need_contiguous=tp.need_contiguous)
                       for name, tp in f_anno.items()
                       if isinstance(tp, NamedTensor)}

        kernel = make_kernel(parsed, ctx, tensor_args, axes)
        wrapper = make_wrapper(parsed, axes, tensor_args)
        imports = [ast.Import([ast.alias(j, i)])
                   for i, j in [(ctx.torch, 'torch'), (ctx.triton, 'triton'), (ctx.tl, 'triton.language')]
                   if i not in ctx.globals]
        module = ast.Module(imports + [kernel, wrapper], [])
        ast.fix_missing_locations(module)
        code = ast.unparse(module)

        if print_result:
            print('\n### Result:')
            print(code)

        if save_to is not None:
            module_file = save_to
            with open(save_to, 'w') as file:
                imports = ast.Module([
                    ast.Import([ast.alias('torch', ctx.torch)]),
                    ast.Import([ast.alias('triton', ctx.triton)]),
                    ast.Import([ast.alias('triton.language', ctx.tl)])
                ], [])
                file.write(ast.unparse(imports))
                file.write('\n\n\n')
                file.write(code)
                file.write('\n')
        else:
            module_file = f'<{parsed.name}_kernel>'
            patch_cache(module_file, code)
        compiled = compile(module, module_file, 'exec')
        exec(compiled, ctx.globals)
        return ctx.globals[parsed.name]
    return decorator
