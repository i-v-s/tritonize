import ast
from typing import Any, Set


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
