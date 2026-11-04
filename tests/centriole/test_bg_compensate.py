import unittest
import numpy as np
from centriole.bg_compensate import backgr, MODE_AUTO, MODE_DARK, MODE_BRIGHT, MODE_GRAY

class TestBackgroundCompensation(unittest.TestCase):
    def test_01_basic_functionality(self):
        # Create a simple synthetic image with known background
        img = np.zeros((100, 100), dtype=np.float32)
        img[25:75, 25:75] = 1.0  # Square in the middle
        img += np.random.normal(0, 0.05, img.shape)  # Reduced noise
        
        # Test background compensation
        bg = backgr(img, mode=MODE_AUTO, thresh=2, splinepoints=20, scale=1)
        
        # Verify results
        self.assertEqual(bg.shape, img.shape)
        self.assertTrue(np.all(np.isfinite(bg)))
        
        # The background-subtracted image should have the square visible
        result = img - bg
        middle_intensity = np.mean(result[25:75, 25:75])
        edge_intensity = np.mean(result[0:25, 0:25])
        self.assertGreater(
            middle_intensity,
            edge_intensity, 
            "Background subtraction should make the square more visible"
        )

    def test_02_different_modes(self):
        # Create test image with both bright and dark features
        img = np.zeros((100, 100), dtype=np.float32)
        img[25:50, 25:75] = 1.0  # Bright square
        img[50:75, 25:75] = -1.0  # Dark square
        
        # Test each mode
        for mode in [MODE_AUTO, MODE_DARK, MODE_BRIGHT, MODE_GRAY]:
            bg = backgr(img, mode=mode, thresh=2, splinepoints=5, scale=1)
            self.assertEqual(bg.shape, img.shape)
            self.assertTrue(np.all(np.isfinite(bg)))
            
            # Verify the background is reasonable
            result = img - bg
            if mode == MODE_DARK:
                # Dark mode should preserve dark features
                self.assertLess(np.mean(result[50:75, 25:75]), np.mean(result[25:50, 25:75]))
            elif mode == MODE_BRIGHT:
                # Bright mode should preserve bright features
                self.assertGreater(np.mean(result[25:50, 25:75]), np.mean(result[50:75, 25:75]))

    def test_03_edge_cases(self):
        # Test with uniform image first (simpler case)
        img = np.ones((50, 50), dtype=np.float32)
        bg = backgr(img, mode=MODE_AUTO, thresh=2, splinepoints=5, scale=1)
        self.assertEqual(bg.shape, img.shape)
        self.assertTrue(np.allclose(bg, img, atol=0.1))  # Background should be close to input
        
        # Test with mask that has some True values
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True  # Create a valid mask region
        bg = backgr(img, mask=mask, mode=MODE_AUTO, thresh=2, splinepoints=5, scale=1)
        self.assertEqual(bg.shape, img.shape)
        self.assertTrue(np.all(np.isfinite(bg)))

def test_file_background_uses_pillow_bytes_and_closes_image(tmp_path, monkeypatch):
    import importlib
    import tifffile
    from PIL import Image

    module = importlib.import_module("centriole.bg_compensate")
    pixels = np.arange(256, dtype=np.uint16).reshape(16, 16)
    path = tmp_path / "background.tiff"
    tifffile.imwrite(path, pixels, byteorder="<")
    opened = []
    original_open = Image.open

    def open_image(*args, **kwargs):
        image = original_open(*args, **kwargs)
        opened.append(image)
        return image

    captured = []

    def background(image, *args, **kwargs):
        captured.append(image.copy())
        return np.zeros_like(image)

    monkeypatch.setattr(Image, "open", open_image)
    monkeypatch.setattr(module, "backgr", background)
    module.bg_compensate(str(path), 1, 4, 1)
    expected = pixels.astype(np.float32) / 4095.0
    assert captured[0].dtype == expected.dtype
    assert captured[0].shape == expected.shape
    assert captured[0].tobytes() == expected.tobytes()
    assert opened[0].fp is None
