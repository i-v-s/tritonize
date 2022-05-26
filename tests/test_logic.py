from pytest import mark
import torch
from tritonize import tritonize, NamedTensor


def store_if(a, b, c):
    if c > 0:
        b[:] = a


@mark.parametrize('a_ctg', [True, False])
@mark.parametrize('b_ctg', [True, False])
@mark.parametrize('c_ctg', [True, False])
@mark.parametrize('size', [15, 128, 130])
def test_store_if(a_ctg, b_ctg, c_ctg, size):
    _store_if = tritonize(
        anno=dict(
            a=NamedTensor('x', need_contiguous=a_ctg),
            b=NamedTensor('x', need_contiguous=b_ctg),
            c=NamedTensor('x', need_contiguous=c_ctg)
        ), DEFAULT_BS=128)(store_if)
    a = torch.arange(size, device='cuda') + 1
    b = torch.zeros(size, device='cuda', dtype=torch.int64)
    c = torch.arange(size, device='cuda') - 10
    _store_if(a, b, c)
    assert (b == torch.where(c > 0, a, torch.zeros(size, device='cuda', dtype=torch.int64))).all()

