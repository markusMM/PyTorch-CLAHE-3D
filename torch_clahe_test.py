import torch
from torch_clahe import compute_clahe


class TestCLAHE3D:

    x1 = torch.randn(2, 1, 9, 32, 32, 32)
    x2 = torch.randn(2, 1, 128, 128, 128)

    def test_shape(self):
        x1 = self.x1
        x1 = compute_clahe(x1, torch.ones_like(x1))
        assert x1.shape[0] == 2
        assert x1.shape[1] == 1
        assert x1.shape[2] == 9
        assert x1.shape[3] == 32
        assert x1.shape[4] == 32
        assert x1.shape[5] == 32

    def test_inequality(self):
        x1 = self.x1
        x1 = compute_clahe(x1, torch.ones_like(x1))
        assert (x1 == self.x1).sum() < 128

    def test_pipeline(self):
        try:
            from folding3d import widow_3d
            x2 = window_3d(self.x2, kernel_size=32)
            x2 = compute_clahe(x2, torch.ones_like(x2))
            assert x2.shape[-1] == 32
            assert x2.shape[-2] == 32
            assert x2.shape[-3] == 32
        except ImportError:
            pass
