import ast
from inspect import getsource
from types import FunctionType
from typing import Dict, Optional, Any, List


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
