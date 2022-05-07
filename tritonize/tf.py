from types import FunctionType
from typing import List, Dict, Set, Any
from inspect import getsource
import ast

from .data import TensorArgument, Writer
from .utils import ast_and


class Inliner(ast.NodeTransformer):
    def __init__(self, ns):
        self.ns = ns

    # def visit_

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            func = self.ns.get(node.func.id, None)
            if isinstance(func, FunctionType):
                tree = ast.parse(getsource(func)).body[0]
                print()
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


# TODO: Remove
class TensorArgumentReplacer(ast.NodeTransformer):
    def __init__(self, args: Dict[str, TensorArgument]):
        self.args = args

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = node.value
        if isinstance(value, ast.Name) and value.id in self.args:
            arg = self.args[value.id]
            if isinstance(node.ctx, ast.Load):
                return TLLoad(arg, node.attr)  # arg.ast_load(node.attr)
            else:
                raise TypeError('Unexpected context')
        else:
            return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        arg = self.args.get(node.id, None)
        if arg is None:
            return self.generic_visit(node)
        elif isinstance(node.ctx, ast.Load):
            return TLLoad(arg)  # arg.ast_load()
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
                    ast.BinOp(arg.ast_load(field=field), node.op, self.generic_visit(node.value)),
                    field=field))
        return self.generic_visit(node)


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
        raise NotImplementedError('AugAssign: not implemented')


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
                        replace_if(writer, e, ast.UnaryOp(ast.Invert(), loc_mask, f'{mask_id}_{index + 1}'))
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
