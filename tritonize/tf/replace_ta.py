import ast
from typing import List, Dict, Any

from tritonize import TensorArgument


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
