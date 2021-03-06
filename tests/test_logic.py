from pytest import mark, raises
import torch
from tritonize import tritonize, NamedTensor


def store_if(a, b, c):
    if c > 0:
        b[:] = a


def store_if_else(a, b, c):
    if c > 0:
        b[:] = a
    else:
        b[:] = b


def assign_if(a, b, c):
    t = b
    if c > 0:
        t = a
    b[:] = t


def assign_if_else(a, b, c):
    if c > 0:
        t = a
    else:
        t = b
    b[:] = t


def test_branches():
    size = 15
    fn = tritonize(
        anno=dict(
            a=NamedTensor('x'),
            b=NamedTensor('x'),
            c=NamedTensor('x')
        ), DEFAULT_BS=128)(assign_if_else)
    a = torch.arange(size, device='cuda') + 1
    b = torch.zeros(size, device='cuda', dtype=torch.int64)
    c = torch.arange(size, device='cuda') - 10
    fn(a, b, c)
    assert (b == torch.where(c > 0, a, torch.zeros(size, device='cuda', dtype=torch.int64))).all()


@mark.parametrize('fn', [store_if, assign_if])
@mark.parametrize('a_ctg', [True, False])
@mark.parametrize('b_ctg', [True, False])
@mark.parametrize('c_ctg', [True, False])
@mark.parametrize('size', [15, 128, 130])
def test_if(fn, a_ctg, b_ctg, c_ctg, size):
    fn = tritonize(
        anno=dict(
            a=NamedTensor('x', need_contiguous=a_ctg),
            b=NamedTensor('x', need_contiguous=b_ctg),
            c=NamedTensor('x', need_contiguous=c_ctg)
        ), DEFAULT_BS=128)(fn)
    a = torch.arange(size, device='cuda') + 1
    b = torch.zeros(size, device='cuda', dtype=torch.int64)
    c = torch.arange(size, device='cuda') - 10
    fn(a, b, c)
    assert (b == torch.where(c > 0, a, torch.zeros(size, device='cuda', dtype=torch.int64))).all()


def assign_bad_seq(a, b, c):
    if c > 0:
        t = a
        m = a
        n = b + m
    else:
        t = b
        n = b
        m = a + n
    a[:] = m + n + t


def test_bad_seq():
    nt = NamedTensor('x')
    t = tritonize(anno=dict(a=nt, b=nt, c=nt))
    with raises(AssertionError) as e:
        t(assign_bad_seq)
    assert e.value.args[0] == 'Circular initialization'


def assign_if_many(a, b, c, r):
    a_ = a
    b_ = b
    if c > 0:
        t = a_
        m = b_
        n = t + 2 * m
    elif c > 1:
        t = b_
        m = a_
        n = m + 3 * t
    else:
        t = b_
        m = a_
        n = m + 4 * t
    r[:] = m + n + t


def test_assign_if_many():
    a, b = (torch.tensor([v], device='cuda') for v in [1, 100])

    def wrapper(f, c):
        r = torch.zeros(1, device='cuda')
        f(a, b, c, r)
        return r.item()

    fn = tritonize(
        anno=dict(
            a=NamedTensor('x'),
            b=NamedTensor('x'),
            c=NamedTensor('x'),
            r=NamedTensor('x')
        ))(assign_if_many)
    assert wrapper(assign_if_many, 0) == wrapper(fn, torch.tensor([0], device='cuda'))
    assert wrapper(assign_if_many, 1) == wrapper(fn, torch.tensor([1], device='cuda'))
    assert wrapper(assign_if_many, 2) == wrapper(fn, torch.tensor([2], device='cuda'))


def assign_with_temp(a, b):
    if a > 0:
        t = a
        r = t + a
    else:
        r = -a
    b[:] = r


def assign_with_temp_2(a, b):
    if a > 5:
        t = a
        r = t + a
    elif a > 2:
        t = 2 * a
        r = a * t
    else:
        r = -a
    b[:] = r


@mark.parametrize('size', [100, 200])
@mark.parametrize('fn', [assign_with_temp, assign_with_temp_2])
def test_assign_with_temp(fn, size):
    a = torch.arange(size, device='cuda') - 20
    b = torch.zeros(size, device='cuda')
    tfn = tritonize(anno=dict(a=NamedTensor('x'), b=NamedTensor('x')), DEFAULT_BS=128)(fn)
    tfn(a, b)
    t = torch.zeros(size)
    for i in range(size):
        fn(a[i].item(), t[i:i + 1])
    assert (b.cpu() == t).all()

