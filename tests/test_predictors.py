import unittest

import numpy as np
import torch

from prometheo.predictors import Predictors
from prometheo.utils import device


class TestPredictorsDevice(unittest.TestCase):
    def test_move_predictors_to_device_with_numpy(self):
        predictors = Predictors(
            s1=np.ones((2, 1, 1, 12, 2), dtype=np.float32),
            s2=None,
            meteo=np.ones((2, 12, 2), dtype=np.float32),
            dem=None,
            latlon=np.ones((2, 2), dtype=np.float32),
            label=np.zeros((2, 1, 1, 1, 1), dtype=np.float32),
            timestamps=np.array([[1, 1, 2021]] * 2).reshape(2, 1, 3)
        )

        moved = predictors.move_predictors_to_device(device)

        for key, val in moved._asdict().items():
            if isinstance(val, torch.Tensor):
                self.assertEqual(val.device, device)
                self.assertIsInstance(val, torch.Tensor)

    def test_move_predictors_to_device_with_mixed_types(self):
        predictors = Predictors(
            s1=torch.randn(2, 1, 1, 12, 2),
            meteo=np.random.rand(2, 12, 2),
            latlon=None,
            dem=None,
            label=None,
            s2=None,
            timestamps=None
        )
        moved = predictors.move_predictors_to_device()
        self.assertTrue(moved.s1.device == device)
        self.assertTrue(moved.meteo.device == device)

    # Why: Ensures you can safely unpack predictors and avoid silent None propagation.
    def test_as_dict_with_and_without_ignore_nones(self):
        predictors = Predictors(
            s1=torch.rand(1), s2=None, meteo=None, dem=None, latlon=torch.rand(1), label=None, timestamps=None
        )
        d1 = predictors.as_dict(ignore_nones=True)
        d2 = predictors.as_dict(ignore_nones=False)

        self.assertIn("s1", d1)
        self.assertNotIn("s2", d1)
        self.assertIn("s2", d2)
        self.assertIsNone(d2["s2"])

    # Why: NamedTuple is immutable — this ensures you're returning a new instance.
    def test_move_predictors_to_device_returns_new_instance(self):
        predictors = Predictors(
            s1=np.ones((1, 1, 1, 1, 2), dtype=np.float32),
            s2=None, meteo=None, dem=None, latlon=None, label=None, timestamps=None
        )
        moved = predictors.move_predictors_to_device(device)

        self.assertIsNot(predictors, moved)
        self.assertEqual(predictors.s1.dtype, np.float32)
        self.assertTrue(isinstance(moved.s1, torch.Tensor))

    # Why: Makes sure functions like .as_dict() or .move_predictors_to_device() don't crash when all values are None.
    def test_all_fields_none_safe_handling(self):
        predictors = Predictors()  # All default to None
        self.assertEqual(predictors.as_dict(), {})

        moved = predictors.move_predictors_to_device(device)
        self.assertIsInstance(moved, Predictors)

    # Why: Validates custom collate_fn handles variable presence of inputs correctly.
    def test_collate_fn_correctness(self):
        from prometheo.predictors import collate_fn
        batch = [
            Predictors(
                s1=torch.rand(1,1,1,12,2),
                s2=None,
                meteo=torch.rand(1,12,2),
                dem=None,
                latlon=torch.rand(1,2),
                label=torch.rand(1,1,1,1,1),
                timestamps=torch.randint(0, 12, (1,12,3))
            ) for _ in range(3)
        ]
        collated = collate_fn(batch)
        self.assertEqual(collated.s1.shape[0], 3)
        self.assertEqual(collated.meteo.shape[0], 3)

    def test_to_torchtensor_converts_numpy_and_preserves_tensor(self):
        from prometheo.predictors import to_torchtensor

        arr = np.ones((3,))
        tensor = torch.tensor([1.0, 2.0])

        converted = to_torchtensor(arr, device)
        self.assertIsInstance(converted, torch.Tensor)
        self.assertEqual(converted.device, device)

        already_tensor = to_torchtensor(tensor, device)
        self.assertEqual(already_tensor.device, device)

