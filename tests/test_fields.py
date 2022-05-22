from typing import NamedTuple
from pytest import fixture, raises, mark
import torch
from tritonize import tritonize, NamedTensor


class Fields(NamedTuple):
    a: float
    b: float
    c: float


def sum_to_field(a: NamedTensor('x', Fields), b: NamedTensor('x'), c: NamedTensor('x')):
    a.a = b + c
    a.b = b
    a.c = c


@mark.parametrize('a_ctg,b_ctg', [(False, False), (False, True), (True, False), (True, True)])
def test_fields(a_ctg, b_ctg):
    size = 100
    f = tritonize(DEFAULT_BS=128,
                  anno=dict(
                      a=NamedTensor('x', Fields, need_contiguous=a_ctg),
                      b=NamedTensor('x', need_contiguous=b_ctg),
                      c=NamedTensor('x')))(sum_to_field)
    a = torch.zeros((size, 3), dtype=torch.int, device='cuda')
    b = torch.arange(size, device='cuda')
    c = torch.arange(0, size * size, size, device='cuda')
    f(a, b, c)
    assert (a[:, 0] == b + c).all() and (a[:, 1] == b).all() and (a[:, 2] == c).all()
