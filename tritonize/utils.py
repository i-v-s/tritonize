import ast


def ast_bin_op(a1, *args, op=None):
    if args:
        a2, *other = args
        return ast_bin_op(ast.BinOp(a1, op, a2), *other, op=op)
    else:
        return a1


def ast_product(*factors):
    return ast_bin_op(*factors, op=ast.Mult())


def ast_sum(*terms):
    return ast_bin_op(*terms, op=ast.Add())


def ast_and(*args):
    return ast_bin_op(*args, op=ast.BitAnd())


def ast_len(c):
    return ast.Call(ast.Name('len', ast.Load()), [c], [])
