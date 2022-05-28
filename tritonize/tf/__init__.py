from typing import List, Any
import ast

from tritonize.data import Writer
from tritonize.tf.replace_ta import TAName, replace_tensor_argument
from tritonize.tf.inliner import Inliner
from tritonize.tf.reduction_finder import ReductionFinder
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


