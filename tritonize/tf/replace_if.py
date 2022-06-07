import ast
from collections import Counter
from copy import copy
from dataclasses import dataclass
from typing import Any, NamedTuple, List, Generator, Optional, Union, Tuple, Set, Dict

from tritonize import Globals
from tritonize.tf import TAName
from tritonize.utils import ast_and


class LoadStoreInsert(ast.NodeTransformer):
    def __init__(self, mask=None):
        self.mask = mask

    def visit_TAName(self, node: TAName):
        if isinstance(node.ctx, ast.Load):
            return node.arg.ast_load(fields=node.fields, mask=self.mask)
        else:
            return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = [self.visit(target) for target in node.targets]
        value = self.visit(node.value)
        if any(filter(lambda t: isinstance(t, TAName), targets)):
            if len(targets) > 1:
                raise NotImplementedError('Assign destruction not implemented')
            target: TAName = targets[0]
            assert target.fields, 'Tensor arguments reassign not implemented'
            return ast.Expr(target.arg.ast_store(value, fields=target.fields, mask=self.mask))
        else:
            return ast.Assign(targets, value, node.type_comment)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        target = self.visit(node.target)
        value = self.visit(node.value)
        if isinstance(target, TAName):
            arg = target.arg
            if arg is not None:
                return ast.Expr(arg.ast_store(
                    ast.BinOp(arg.ast_load(), node.op, arg.ast_load(fields=target.fields, mask=self.mask)),
                    fields=target.fields, mask=self.mask
                ))
        else:
            node.target = target
            node.value = value
            return node


class MaskedBody(NamedTuple):
    mask_id: str
    test: ast.expr
    body: List[ast.stmt]
    is_else: bool = False


def parse_if(node, loc_mask, mask_id, index) -> Generator[MaskedBody, None, None]:
    test_mask = loc_mask
    while True:
        loc_mask_id = f'{mask_id}_{index}'
        node_test = LoadStoreInsert(test_mask).visit(node.test)
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


@dataclass
class Node:
    target: Optional[str]
    values: Union[ast.AST, List[Tuple[Optional[ast.expr], ast.expr]]]
    next: List['Node']
    first: bool = True


def body_union(g: Globals, bodies: List[List[ast.AST]], masks: List[ast.expr],
               unconditional_vars: Set[str], present_vars: Set[str]) -> List[ast.AST]:
    seqs = [[item.targets[0].id for item in body if isinstance(item, ast.Assign)] for body in bodies]
    assert all(v == 1 for seq in seqs for v in Counter(seq).values())
    nodes = []
    var_map: Dict[str, Node] = {}
    for mask, body in zip(masks, bodies):
        last: Optional[Node] = None
        for item in body:
            assert not isinstance(item, ast.If), 'Unexpected if'
            if isinstance(item, ast.Assign):
                assert len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                target = item.targets[0].id
                if target in unconditional_vars:
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
            statement.lineno = 1
        result.append(statement)
    return result


def replace_if(g: Globals, body: List[ast.stmt], mask: ast.expr = None, mask_id: str = 'mask',
               ns: Optional[Set[str]] = None):
    loc_vars = set() if ns is None else copy(ns)
    index = 0
    result = []
    unconditional_vars = set()
    for node in body:
        if isinstance(node, ast.If):
            branches: List[MaskedBody] = list(parse_if(node, mask, mask_id, index + 1))
            masks = []
            for b in branches:
                assert b.mask_id not in loc_vars
                result.append(ast.Assign([ast.Name(b.mask_id, ast.Store())], b.test))
                mask = ast.Name(b.mask_id, ast.Load())
                setattr(mask, 'is_else', b.is_else)
                masks.append(mask)
            bodies, lvs, mns = zip(*(
                replace_if(g, b.body, m, b.mask_id, loc_vars)
                for m, b in zip(masks, branches)
            ))
            unconditional_vars = unconditional_vars.union(b[0] for b in branches)
            for lv, mn in zip(lvs, mns):
                unconditional_vars = unconditional_vars.union(mn)
                assert lv == lvs[0]
            result.append(body_union(g, list(bodies), masks, unconditional_vars, loc_vars))
            loc_vars = loc_vars.union(lvs[0])
            index += len(branches)
        else:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                loc_vars.add(node.targets[0].id)
            result.append(LoadStoreInsert(mask).visit(node))
    return result, loc_vars.difference(ns or ()), unconditional_vars
