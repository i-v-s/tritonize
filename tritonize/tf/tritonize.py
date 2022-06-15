import ast
from collections import Counter
from copy import copy
from dataclasses import dataclass
from typing import Any, NamedTuple, List, Generator, Optional, Iterable, Union, Tuple, Set, Dict

from tritonize import Context, TensorArgument
from tritonize.utils import ast_and, expand, call_args
from tritonize.data import Axis, TensorValue


class MaskedBody(NamedTuple):
    mask_id: str
    test: ast.expr
    body: List[ast.stmt]
    is_else: bool = False


@dataclass
class Node:
    target: Optional[str]
    values: Union[ast.AST, List[Tuple[Optional[ast.expr], ast.expr]]]
    next: List['Node']
    first: bool = True


def body_union(g: Context, bodies: List[List[ast.AST]], masks: List[ast.expr],
               unconditional_vars: Set[str], present_vars: Set[str]) -> List[ast.stmt]:

    # Prepare bodies, rejecting poorly initialized variables from merging
    ctrs = [
        Counter(t.id for st in b if isinstance(st, ast.Assign) for t in st.targets if isinstance(t, ast.Name))
        for b in bodies
    ]
    last_else = masks[-1].is_else
    good_vars = set(k for c in ctrs for k in c.keys() if k not in unconditional_vars)
    for var in list(good_vars):
        sparse = sum(1 for c in ctrs if var in c) < len(bodies)
        present = var in present_vars
        have_else = last_else and var in ctrs[-1]
        if not present and (sparse or (not sparse and not have_else)):
            good_vars.remove(var)

    for i, var in ((i, v) for i, c in enumerate(ctrs) for v in good_vars if c[v] > 1):
        raise NotImplementedError('Need renaming')

    # Build DAG from bodies, merging assigns with same target
    nodes = []
    var_map: Dict[str, Node] = {}
    for mask, body in zip(masks, bodies):
        last: Optional[Node] = None
        for item in body:
            assert not isinstance(item, ast.If), 'Unexpected if'
            if isinstance(item, ast.Assign):
                assert len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                target = item.targets[0].id
                if target not in good_vars:
                    node = Node(target, item.value, [])
                elif (node := var_map.get(target)) is None:
                    node = Node(target, [(mask, item.value)], [])
                    var_map[target] = node
                    nodes.append(node)
                else:
                    node.values.append((mask, item.value))
            else:
                node = Node(None, item, [])
                nodes.append(node)
            if last is not None:
                last.next.append(node)
            last = node

    # Try to extrude sequence from DAG
    sequence = []
    while nodes:
        for node in nodes:
            for nx in node.next:
                nx.first = False
        seq = [node for node in nodes if node.first]
        assert seq, "Circular initialization"
        sequence.extend(seq)
        nodes = [node for node in nodes if not node.first]
        for node in nodes:
            node.first = True

    # Try to combine multiple values with tl.where()
    result = []
    for node in sequence:
        node.next.clear()
        target = node.target
        values = node.values
        if target is None:
            assert isinstance(values, ast.stmt)
            statement = values
        else:
            if isinstance(values, list):
                if len(bodies) > len(values):
                    if target in present_vars:
                        value = g.fold_where(values, ast.Name(target, ast.Load()))
                    else:
                        raise ValueError(f'Unable to deduce value for {target}')
                else:
                    last_mask, last_value = values[-1]
                    if last_mask.is_else:
                        value = g.fold_where(values[:-1], last_value)
                    elif target in present_vars:
                        value = g.fold_where(values, ast.Name(target, ast.Load()))
                    else:
                        raise ValueError(f'Unable to deduce value for {target}')
            else:
                assert isinstance(values, ast.expr)
                value = values
            statement = ast.Assign([ast.Name(target, ast.Store())], value)
        result.append(statement)
    return result


class Tritonize(ast.NodeTransformer):
    type_attr = 'value_type'
    var_attr = 'variable'
    fields_attr = 'fields'
    glob_attr = 'glob'

    def __init__(self, g: Context, tensor_args: Dict[str, TensorArgument], axes: List[Axis]):
        super(Tritonize, self).__init__()
        self.g = g
        self.args = tensor_args
        self.local_vars: Dict[str, TensorValue] = {}
        self.unconditional_vars: Set[str] = set()
        self.mask_id: str = 'mask'
        self.mask_index: int = 0
        self.mask = None
        self.axes = axes
        self.axes_map = {str(a): i for i, a in enumerate(axes)}

    def visit_Name(self, node: ast.Name) -> Any:
        if (arg := self.args.get(node.id, None)) is not None:
            assert isinstance(node.ctx, ast.Load), 'Tensor argument reassignment not implemented'
            result = arg.ast_load()
            setattr(result, self.type_attr, TensorValue(arg.axes_order))
            setattr(result, self.var_attr, arg)
            setattr(result, self.fields_attr, [])
            return result
        if (val := self.local_vars.get(node.id, None)) is not None:
            setattr(node, self.type_attr, val)
            return node
        if node.id == self.g.tl:
            setattr(node, self.glob_attr, 'tl')
            return node
        if node.id == 'float':
            setattr(node, self.glob_attr, 'float')
            return node
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        item = self.visit(node.value)
        if hasattr(item, self.var_attr):
            fields = item.fields + [node.attr]
            arg: TensorArgument = getattr(item, self.var_attr)
            result = arg.ast_load(fields=fields)
            setattr(result, self.type_attr, getattr(item, self.type_attr).without_field(node.attr))
            setattr(result, self.var_attr, arg)
            setattr(result, self.fields_attr, fields)
            return result
        elif hasattr(item, self.type_attr):
            setattr(node, self.type_attr, getattr(item, self.type_attr).without_field(node.attr))
        elif hasattr(item, self.glob_attr):
            setattr(node, self.glob_attr, f'{getattr(item, self.glob_attr)}.{node.attr}')
        node.value = item
        return node

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        sl = self.visit(node.slice)
        if hasattr(value, self.var_attr):
            if isinstance(sl, ast.Slice) and sl.upper is sl.lower is sl.step is None:
                return value
            raise NotImplementedError()
        return ast.Subscript(value, sl, node.ctx)

    def visit_Call(self, node: ast.Call) -> Any:
        node = self.generic_visit(node)
        func = node.func
        glob = getattr(func, self.glob_attr, None)
        if isinstance(func, ast.Attribute) and glob in ['tl.max', 'tl.min', 'tl.sum']:
            arg, axis, other = call_args(node, 'input axis other', (None, None))
            if (vt := getattr(arg, self.type_attr, False)) and isinstance(axis, ast.Constant):
                axis = axis.value
                if isinstance(axis, str):
                    axes = list(map(str, vt.axes))
                    assert axis in axes, f'Unable to reduce by axis {axis} in {glob}'
                    axis = axes.index(axis)
                    if other is not None and self.mask is not None:
                        arg = self.g.ast_where(self.mask, arg, other)
                    node.args = [arg, ast.Constant(axis)]
                    node.keywords.clear()
                assert isinstance(axis, int) and 0 <= axis < len(vt.axes)
                vt.axes.pop(axis)
                setattr(node, self.type_attr, vt)
        return node

    def broadcast(self, left: ast.expr, right: ast.expr) -> Tuple[ast.expr, ast.expr, Optional[TensorValue]]:
        lt, rt = (getattr(a, self.type_attr) if hasattr(a, self.type_attr) else None for a in [left, right])
        if lt is not None and rt is not None:
            l_i, r_i = ([self.axes_map.get(str(axis)) for axis in t.axes] for t in [lt, rt])
            f_i = sorted(set(l_i + r_i))
            left = expand(left, [i not in l_i for i in f_i])
            right = expand(right, [i not in r_i for i in f_i])
            tv = TensorValue(list(map(self.axes.__getitem__, f_i)))
        else:
            tv = lt or rt
        return left, right, tv

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right, tv = self.broadcast(*map(self.visit, [node.left, node.right]))
        node.left, node.right = left, right
        if tv is not None:
            setattr(node, self.type_attr, tv)
        return node

    def visit_Compare(self, node: ast.Compare) -> Any:
        left, *other = map(self.visit, [node.left] + node.comparators)
        comps = []
        tv = None
        for o in other:
            left, o, tv = self.broadcast(left, o)
            comps.append(o)
        node.left, node.comparators = left, comps
        if tv is not None:
            setattr(node, self.type_attr, tv)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:  # TODO: Test
        values = list(map(self.visit, node.values))
        if isinstance(node.op, ast.Or) and len(values) == 2:
            l, r = values
            if tv := getattr(l, self.type_attr, False):
                if hasattr(l, self.var_attr):
                    assert isinstance(l, ast.Call) and all(kw.arg != 'other' for kw in l.keywords)
                    l.keywords.append(ast.keyword('other', r))
                    return l
        node.values = values
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            node.op = ast.Invert()
        if tv := getattr(node.operand, self.type_attr, False):
            setattr(node, self.type_attr, tv)
        return node

    def add_unconditional(self, names: Iterable[str]) -> None:
        exists = self.unconditional_vars.intersection(names)
        assert not exists, 'Try to add exists unconditional variable(s): ' + ', '.join(exists)
        self.unconditional_vars.update(names)

    def visit_Assign(self, node: ast.Assign) -> Any:
        node.value = self.visit(node.value)
        node.targets = list(map(self.visit, node.targets))
        target = node.targets[0]
        if len(node.targets) == 1 and hasattr(node.value, self.type_attr):
            if hasattr(target, self.var_attr):
                var: TensorArgument = getattr(target, self.var_attr)
                target_type, value_type = getattr(target, self.type_attr), getattr(node.value, self.type_attr)
                if target_type != value_type:
                    ta, va = (list(map(str, t.axes)) for t in [target_type, value_type])
                    if not all(a in ta for a in va):
                        v = f'{ast.unparse(node.value)} {value_type}'
                        raise AssertionError(f'Unable to assign {v} to {var.name} {target_type}')
                    dims = [va.index(a) if a in va else None for a in ta]
                    assert (d := list(filter(lambda t: isinstance(t, int), dims))) == sorted(d)
                    node.value = expand(node.value, [t is None for t in dims])
                return ast.Expr(var.ast_store(node.value, getattr(target, self.fields_attr), self.mask))
            elif hasattr(node.value, self.type_attr) and isinstance(target, ast.Name):
                setattr(node, 'new_vars', {target.id: getattr(node.value, self.type_attr)})
        else:
            assert all(hasattr(t, 'ctx') and isinstance(t.ctx, ast.Store) for t in node.targets),\
                f'Unable to assign: {ast.unparse(node)}'

        if len(node.targets) == 1 and isinstance(target, ast.Name)\
                and isinstance(node.value, ast.Name) and target.id == node.value.id:
            return None
        else:
            return node

    def parse_if(self, node: ast.If, index: int) -> Generator[MaskedBody, None, None]:
        loc_mask = self.mask
        mask_id = 'mask' if loc_mask is None else loc_mask.id
        while True:
            loc_mask_id = self.g.new_name(f'{mask_id}_{index}')
            node_test = self.visit(node.test)
            yield MaskedBody(
                loc_mask_id,
                node_test if loc_mask is None else ast_and(loc_mask, node_test),
                node.body
            )
            loc_mask = ast.Name(loc_mask_id, ast.Load())
            if node.orelse:
                e = node.orelse
                if isinstance(e, list) and len(e) == 1 and isinstance(e[0], ast.If):
                    node = e[0]
                    loc_mask = ast.UnaryOp(ast.Invert(), loc_mask)
                    index += 1
                    continue
                else:
                    yield MaskedBody(
                        f'{mask_id}_{index + 1}',
                        ast.UnaryOp(ast.Invert(), loc_mask),
                        e,
                        is_else=True
                    )
                    break
            else:
                break

    def replace_if(self, body: List[ast.stmt], mask: ast.expr = None,
                   ns: Optional[Dict[str, TensorValue]] = None):
        loc_vars = dict() if ns is None else copy(ns)
        index = 0
        result = []
        for node in body:
            self.mask = mask
            self.local_vars = loc_vars
            if isinstance(node, ast.If):
                branches: List[MaskedBody] = list(self.parse_if(node, index + 1))
                masks = []
                for b in branches:
                    assert b.mask_id not in loc_vars
                    result.append(ast.Assign([ast.Name(b.mask_id, ast.Store())], b.test))
                    m = ast.Name(b.mask_id, ast.Load(), is_else=b.is_else)
                    if hasattr(b.test, self.type_attr):
                        setattr(m, self.type_attr, getattr(b.test, self.type_attr))
                    masks.append(m)
                bodies, lvs = zip(*(
                    self.replace_if(b.body, m, loc_vars)
                    for m, b in zip(masks, branches)
                ))
                self.add_unconditional(b.mask_id for b in branches)
                result.extend(body_union(self.g, list(bodies), masks, self.unconditional_vars, set(loc_vars)))
                for lv in lvs:
                    for var, tv in lv.items():
                        if (t := loc_vars.get(var)) is not None:
                            assert t == tv, f'Attempt to initialize {var} with different dims'
                        loc_vars[var] = tv
                index += len(branches)
            else:
                node = self.visit(node)
                if hasattr(node, 'new_vars'):
                    loc_vars.update(getattr(node, 'new_vars'))
                if node is not None:
                    result.append(node)
        return result, loc_vars

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.body = self.replace_if(node.body)[0]
        return node
