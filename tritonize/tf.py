from types import FunctionType
from typing import List, Dict, Set, Any, Optional
from inspect import getsource
import ast

from .data import TensorArgument, Writer
from .utils import ast_and


class Renamer(ast.NodeTransformer):
    def __init__(self, rename_map: Dict[str, str], result: Optional[ast.Name] = None):
        self.result = result
        self.rename_map = rename_map

    def visit_Name(self, node: ast.Name) -> Any:
        value = self.rename_map.get(node.id, None)
        if value is not None:
            node.id = value
            return node
        else:
            return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        if self.result is not None:
            return ast.Assign([self.result], node.value)
        else:
            return self.generic_visit(node)


class Inliner(ast.NodeTransformer):
    def __init__(self, ns):
        self.ns = ns
        self.locals = set()
        self.generated = []
        self.count = 0

    def process_body(self, body: List[ast.stmt]):
        result = []
        for stmt in body:
            item = self.visit(stmt)
            if self.generated:
                result.extend(self.generated)
                self.generated.clear()
            if item is not None:
                result.append(item)
        return result

    def visit_If(self, node: ast.If) -> Any:
        node.test = self.visit(node.test)
        node.body = self.process_body(node.body)
        if node.orelse:
            node.orelse = self.process_body(node.orelse)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        for arg in node.args.args:
            self.locals.add(arg.arg)
        node.body = self.process_body(node.body)
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.locals.add(target.id)
        if isinstance(node.value, ast.Call) and len(node.targets) == 1:
            value = self.visit_Call(node.value, result=node.targets[0])
            if value is None:
                return None
            else:
                node.value = value
                return node
        else:
            return self.generic_visit(node)

    def visit_Call(self, node: ast.Call, result: Optional[ast.Name] = None) -> Any:
        if isinstance(node.func, ast.Name):
            func = self.ns.get(node.func.id, None)
            if isinstance(func, FunctionType):
                tree: ast.FunctionDef = ast.parse(getsource(func)).body[0]
                r_map = {}
                for fa, ca in zip(tree.args.args, node.args):
                    r_map[fa.arg] = ca.id
                ren = Renamer(r_map, result)
                self.generated.extend(ren.visit(item) for item in tree.body)
        return self.generic_visit(node)


class TAName(ast.expr):
    def __init__(self, arg: TensorArgument, *fields, ctx=None):
        super(TAName, self).__init__()
        self.arg = arg
        self.fields = list(fields)
        self.ctx = ctx

    _fields = ()


class TLLoad(ast.expr):
    """ tl.load """
    def __init__(self, arg: TensorArgument, *fields, other=None):
        super(TLLoad, self).__init__()
        self.arg = arg
        self.fields = list(fields)
        self.other = other

    _fields = ('other',)


def replace_tensor_argument(body: List[ast.stmt], args: Dict[str, TensorArgument]):
    class ArgsReplace(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> Any:
            arg = args.get(node.id, None)
            if arg is None:
                return self.generic_visit(node)
            else:
                return TAName(arg, ctx=node.ctx)

        def visit_Attribute(self, node: ast.Attribute) -> Any:
            item = self.visit(node.value)
            if isinstance(item, TAName):
                item.fields.append(node.attr)
                item.ctx = node.ctx
                return item
            else:
                node.value = item
                return node

        def visit_Subscript(self, node: ast.Subscript) -> Any:
            value = self.visit(node.value)
            sl = self.visit(node.slice)
            if isinstance(value, TAName):
                value.fields.append(sl)
                value.ctx = node.ctx
                return value
            else:
                return ast.Subscript(value, sl, node.ctx)

    ar = ArgsReplace()
    return [ar.visit(item) for item in body]


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


def replace_if(writer: Writer, body: List[ast.stmt], mask: ast.expr = None, mask_id: str = 'mask'):
    index = 0
    for node in body:
        if isinstance(node, ast.If):
            loc_mask = mask
            while True:
                index += 1
                loc_mask_id = f'{mask_id}_{index}'
                node_test = LoadStoreInsert(loc_mask).visit(node.test)
                loc_mask = writer.init(
                    loc_mask_id,
                    node_test
                    if loc_mask is None else
                    ast_and(loc_mask, node_test)
                )
                replace_if(writer, node.body, loc_mask, loc_mask_id)
                if node.orelse:
                    e = node.orelse
                    if isinstance(e, list) and len(e) == 1 and isinstance(e[0], ast.If):
                        node = e[0]
                        loc_mask = ast.UnaryOp(ast.Invert(), loc_mask)
                        continue
                    else:
                        replace_if(writer, e, ast.UnaryOp(ast.Invert(), loc_mask), f'{mask_id}_{index + 1}')
                        break
                else:
                    break
        else:
            writer.write(LoadStoreInsert(mask).visit(node))


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
