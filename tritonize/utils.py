import ast
from typing import Any


def ast_bin_op(a1, *args, op=None):
    if args:
        a2, *other = args
        return ast_bin_op(ast.BinOp(a1, op, a2), *other, op=op)
    else:
        return a1


def ast_product(*factors):
    return ast_bin_op(*factors, op=ast.Mult()) if factors else ast.Constant(1)


def ast_sum(*terms):
    return ast_bin_op(*terms, op=ast.Add())


def ast_and(*args):
    return ast_bin_op(*args, op=ast.BitAnd())


def ast_len(c):
    return ast.Call(ast.Name('len', ast.Load()), [c], [])


def none(*_) -> Any:
    return None


def build_seq(seqs, fn=none):
    seq_map = {}
    for seq in seqs:
        sl = len(seq) - 1
        for i, w in enumerate(seq):
            sm = seq_map.get(w, None)
            pre = seq[i - 1] if i > 0 else None
            nex = seq[i + 1] if i < sl else None
            if sm is None:
                seq_map[w] = [pre, nex]
            else:
                p, n = sm
                if p != pre:
                    sm[0] = fn(p, pre)
                if n != nex:
                    sm[1] = fn(p, pre)
    result = []
    for d, (p, n) in seq_map.items():
        if p is None:
            item = [d]
            while n is not None:
                item.append(n)
                n = seq_map[n][1]
            result.append(tuple(item))
    return result
