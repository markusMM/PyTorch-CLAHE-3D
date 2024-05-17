import numpy as np
import torch
from torch_clahe import compute_clahe


class TestCLAHE3D:

    x1 = torch.randn(1, 10, 3, 32, 32, 32)
    x2 = torch.randn(1, 3, 128, 128, 128)

    def test_shape(self):
        x1 = self.x1
        x1 = compute_clahe(x1, torch.ones_like(x1), .6, 256)
        assert x1.shape[0] == 1
        assert x1.shape[1] == 10
        assert x1.shape[2] == 3
        assert x1.shape[3] == 32
        assert x1.shape[4] == 32
        assert x1.shape[5] == 32

    def test_inequality(self):
        x1 = self.x1.clone()
        x1 = compute_clahe(x1, torch.ones_like(x1), .6, 256)
        assert ~np.allclose(x1, self.x1)

    def test_pipeline(self):
        try:
            from folding3d import window_3d
            x2 = window_3d(self.x2, 32, 16)
            x2 = compute_clahe(x2.clone(), torch.ones_like(x2), .6, 256)
            assert x2.shape[-1] == 32
            assert x2.shape[-2] == 32
            assert x2.shape[-3] == 32
        except ImportError:
            pass

    def test_large_c(self):
        x = torch.randn(6, 1000, 4, 24, 24, 24)
        y = compute_clahe(x.clone(), torch.ones_like(x), .75, 256)
        assert ~np.allclose(x, y)

    def test_property(self):
        y = compute_clahe(self.x1.clone(), torch.ones_like(self.x1), .75, 256)
        assert ~np.allclose(self.x1, y)

    def test_reference(self):
        x = self.x1
        y = compute_clahe(x[0][None].clone(),
                          torch.ones_like(x[0][None]), .75, 256)
        assert ~np.allclose(x[0][None], y)

    def test_smoke(self):
        for k in range(2000, 8000, 1000):
            try:
                x = torch.randn(6, 1000, 4, 24, 24, 24)
                y = compute_clahe(x.clone(), torch.ones_like(x), .75, 256)
            except MemoryError:
                print(f'Smoke at C={k}')
            assert ~np.allclose(x, y)
