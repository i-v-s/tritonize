import ast
from inspect import getsource
from types import FunctionType
from typing import Dict, Optional, Any, List

from tritonize.data import Context


class Renamer(ast.NodeTransformer):
    def __init__(self, rename_map: Dict[str, str], ctx: Context, result: Optional[ast.Name] = None, auto_add=False):
        self.ctx = ctx
        self.result = result
        self.rename_map = rename_map
        self.auto_add = auto_add

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if value := self.rename_map.get(node.id, False):
            node.id = value
        elif self.auto_add and isinstance(node.ctx, ast.Store):
            node.id = self.rename_map[node.id] = self.ctx.new_name(node.id)
        return node

    def visit_Return(self, node: ast.Return) -> Any:
        if self.result is not None:
            return ast.Assign([self.result], self.visit(node.value))
        else:
            return self.generic_visit(node)


class Inliner(ast.NodeTransformer):
    def __init__(self, ctx: Context):
        self.ctx = ctx
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
        node.body = self.process_body(node.body)
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        if isinstance(node.value, ast.Call) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
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
            func = self.ctx.globals.get(node.func.id, None)
            if isinstance(func, FunctionType):
                tree = ast.parse(getsource(func)).body[0]
                assert isinstance(tree, ast.FunctionDef)
                r_map = {}
                for fa, ca in zip(tree.args.args, node.args):
                    if isinstance(ca, ast.Name):
                        r_map[fa.arg] = ca.id
                    else:
                        nn = self.ctx.new_name(fa.arg)
                        r_map[fa.arg] = nn
                        self.generated.append(ast.Assign([ast.Name(nn, ast.Store())], ca))
                if result is None:
                    result = ast.Name(self.ctx.new_name(f'{node.func.id}_result'), ast.Store())
                else:
                    assert isinstance(result, ast.Name) and isinstance(result.ctx, ast.Store)
                ren = Renamer(r_map, self.ctx, result, auto_add=True)
                self.generated.extend(ren.visit(tree).body)
                return ast.Name(result.id, ast.Load())
        return self.generic_visit(node)
