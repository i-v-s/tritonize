from typing import List, Any
import ast


class ValueTracer(ast.NodeTransformer):
    def __init__(self):
        super(ValueTracer, self).__init__()
        self.args = set()

    def process_body(self, body: List[ast.stmt]):
        result = []
        for stmt in body:
            item = self.visit(stmt)
            if item is not None:
                result.append(item)
        return result

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name in self.args:
                raise NotImplementedError('Argument reassigning not implemented')
            print()
        return node

    def visit_If(self, node: ast.If) -> Any:
        node.test = self.visit(node.test)
        node.body = self.process_body(node.body)
        if node.orelse:
            node.orelse = self.process_body(node.orelse)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        for arg in node.args.args:
            self.args.add(arg.arg)
        node.body = self.process_body(node.body)
        return node
