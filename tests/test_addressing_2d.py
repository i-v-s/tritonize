import torch

from tests.test_addressing import add
from tritonize import tritonize, NamedTensor


def check_add_2d(size_x=100, size_y=100, **kwargs):
    _add = tritonize(**kwargs)(add)
    a = torch.ones((size_x, size_y), dtype=torch.int, device='cuda')
    b = torch.arange(size_x, device='cuda')
    c = torch.arange(0, size_y * size_y, size_y, device='cuda')
    _add(a, b, c)
    print('a:', a)
    assert (a == torch.unsqueeze(b, 1) + torch.unsqueeze(c, 0)).all()


def test_add_xy_trivial():
    check_add_2d(size_x=16, size_y=8, anno={'a': NamedTensor('x', 'y'), 'b': NamedTensor('x'), 'c': NamedTensor('y')},
                 no_mask=['x', 'y'], one_block=['x', 'y'],
                 DEFAULT_BS=16, Y_BS=8)


def test_add_xy():
    check_add_2d(size_x=10, size_y=20, anno={'a': NamedTensor('x', 'y'), 'b': NamedTensor('x'), 'c': NamedTensor('y')},
                 DEFAULT_BS=32)
