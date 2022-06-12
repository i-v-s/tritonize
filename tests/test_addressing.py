from pytest import fixture, raises
import torch

from tritonize import tritonize, NamedTensor


def add(a, b, c):
    a[:] = b + c


def check_add(size=100, **kwargs):
    _add = tritonize(**kwargs)(add)
    a = torch.zeros((size,), dtype=torch.int, device='cuda')
    b = torch.arange(size, device='cuda')
    c = torch.arange(0, size * size, size, device='cuda')
    _add(a, b, c)
    assert (a == b + c).all()


@fixture
def tensor_x():
    return NamedTensor('x')


def test_add(tensor_x):
    check_add(size=100, anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x}, X_BS=128)


def test_one_block(tensor_x):
    for size in [1, 63, 64]:
        check_add(size=size, anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x}, X_BS=64, one_block=['x'])


def test_one_block_fail(tensor_x):
    with raises(AssertionError):
        check_add(size=100, anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x}, X_BS=64, one_block=['x'])


def test_no_mask_fail(tensor_x):
    with raises(AssertionError):
        check_add(size=100, anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x}, X_BS=128, no_mask=['x'])


def test_no_mask(tensor_x):
    for size in [128, 256]:
        check_add(size=size, anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x}, X_BS=128, no_mask=['x'])


def test_add_blocks(tensor_x):
    check_add(size=1000, anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x}, X_BS=128)


def test_add_contiguous(tensor_x):
    check_add(anno={'a': tensor_x, 'b': tensor_x, 'c': tensor_x.contiguous()}, X_BS=128)
