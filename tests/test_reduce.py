from pytest import mark, raises
import torch
import triton.language as tl

from tritonize import tritonize, NamedTensor


def calc_sum_or(a, b):
    a[:] = tl.sum(b or 0, axis='x')


# def calc_sum_other(a, b):
#     a[:] = tl.sum(b, axis='x', other=0)


def test_sub_sum():
    bs = 128
    s1, s2 = 5, 10
    a = torch.zeros(bs, device='cuda', dtype=torch.int64)
    b = torch.arange(bs * bs, device='cuda').reshape(bs, bs)
    f = tritonize(DEFAULT_BS=bs, print_result=True,
                  anno=dict(
                      a=NamedTensor('y'),
                      b=NamedTensor('y', 'x')
                  ))(calc_sum_or)
    f(a[:s1], b[:s1, :s2])
    assert torch.all(a[:s1] == b[:s1, :s2].sum(1)).item()


# def calc_sum_if(a, b):
#     t = b or 0
#     if t > 0:
#         r = tl.sum(t, axis='x')
#     else:
#         r = -1
#     a[:] = r
