import ast
from typing import NamedTuple, Dict, List, Set, Tuple, Iterable, Union, Optional, Any
from collections import namedtuple


class TensorValue(NamedTuple):
    axes: List[str]

    def without_field(self, field: str):  # TODO: Implement
        return TensorValue(self.axes)

    def without_axis(self, axis: Union[str, int]):
        if isinstance(axis, str):
            return TensorValue([a for a in self.axes if a != axis])
        elif isinstance(axis, int):
            return TensorValue(self.axes[:axis] + self.axes[axis + 1:])
        raise TypeError(f'Wrong axis type: {axis}')

    def __str__(self):
        return '(' + ', '.join(map(str, self.axes)) + ')'


def expand(tensor: ast.expr, dims: List[bool]) -> ast.expr:
    """
    Expand tensor dimensions
    :param tensor: tensor to expand
    :param dims: list of dimensions, True for new dimension, False for existing
    :return:
    """
    assert not all(dims), 'No existing dim specified'
    if not any(dims):
        return tensor
    slices = [ast.Constant(None) if e else ast.Slice() for e in dims]
    if len(slices) == 1:
        sl = slices[0]
    else:
        sl = ast.Tuple(slices, ast.Load())
    return ast.Subscript(tensor, sl, ast.Load())


def expand_one(value: ast.expr, dim: int, total: int) -> ast.expr:
    return expand(value, [i != dim for i in range(total)])


def broadcast(*args: ast.expr, axes_map: Dict[str, int]) -> Tuple[Optional[TensorValue], ...]:
    args: List[ast.expr] = list(args)
    assert all(issubclass(a.__class__, ast.expr) for a in args)
    indices = [
        [axes_map.get(str(axis)) for axis in getattr(a, 'value_type').axes]
        for a in args if hasattr(a, 'value_type')
    ]
    if indices:
        f_i = sorted(set(sum(indices, [])))
        if len(indices) > 1:
            for i, ai in zip((i for i, a in enumerate(args) if hasattr(a, 'value_type')), indices):
                args[i] = expand(args[i], [i not in ai for i in f_i])
        inv_map = {v: k for k, v in axes_map.items()}
        tv = TensorValue(list(map(inv_map.get, f_i)))
    else:
        tv = None
    return tv, *args


def ast_bin_op(a1, *args, op=None, axes_map: Optional[Dict[str, int]] = None):
    if axes_map is not None:
        rt, a1, *args = broadcast(a1, *args, axes_map=axes_map)
    else:
        rt = None
    if args:
        a2, *other = args
        result = ast_bin_op(ast.BinOp(a1, op, a2), *other, op=op)
    else:
        result = a1
    if rt is not None:
        setattr(result, 'value_type', rt)
    return result


def ast_product(*factors, **kwargs):
    return ast_bin_op(*factors, op=ast.Mult(), **kwargs) if factors else ast.Constant(1)


def ast_sum(*terms, **kwargs):
    return ast_bin_op(*terms, op=ast.Add(), **kwargs)


def ast_and(*args, **kwargs):
    return ast_bin_op(*args, op=ast.BitAnd(), **kwargs)


def ast_len(c):
    return ast.Call(ast.Name('len', ast.Load()), [c], [])


def ast_invert(arg):
    result = ast.UnaryOp(ast.Invert(), arg)
    if t := getattr(arg, 'value_type', False):
        setattr(result, 'value_type', t)
    return result


def none(*_) -> Any:
    return None


def build_seq(seqs, fn=none):
    seq_map = {}
    for seq in seqs:
        sl = len(seq) - 1
        for i, w in enumerate(seq):
            sm = seq_map.get(w, None)
            pre = seq[i - 1] if i > 0 else None
            nex = seq[i + 1] if i < sl else None
            if sm is None:
                seq_map[w] = [pre, nex]
            else:
                p, n = sm
                if p != pre:
                    sm[0] = fn(p, pre)
                if n != nex:
                    sm[1] = fn(p, pre)
    result = []
    for d, (p, n) in seq_map.items():
        if p is None:
            item = [d]
            while n is not None:
                item.append(n)
                n = seq_map[n][1]
            result.append(tuple(item))
    return result


def ast_equals(a: Any, b: Any) -> bool:
    if not type(a) == type(b):
        return False
    if isinstance(a, ast.AST):
        fields = getattr(a, '_fields')
        assert fields == getattr(b, '_fields')
        for f in fields:
            if not ast_equals(getattr(a, f), getattr(b, f)):
                return False
    elif isinstance(a, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not ast_equals(x, y):
                return False
    else:
        return a == b
    return True


def call_args(node: ast.Call, args: Union[str, List[str]], defaults: Optional[Iterable[Any]] = None):
    return namedtuple('t', args, defaults=defaults)(*node.args, **{kw.arg: kw.value for kw in node.keywords})


class VarScan(ast.NodeVisitor):
    def __init__(self):
        self.vars = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.vars.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.vars.add(node.arg)


def scan_vars(fn: ast.FunctionDef) -> Set[str]:
    vs = VarScan()
    vs.visit(fn)
    return vs.vars
