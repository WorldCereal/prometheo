import unittest
from src.models import Presto
from src.predictors import Predictors, S1_bands, S2_bands, meteo_bands, dem_bands
import numpy as np


class TestPresto(unittest.TestCase):
    def test_forward_from_predictor_with_all(self):
        b, t, h, w = 8, 4, 1, 1

        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_bands)),
            s2=np.random.rand(b, h, w, t, len(S2_bands)),
            meteo=np.random.rand(b, t, len(meteo_bands)),
            dem=np.random.rand(b, h, w, len(dem_bands)),
            latlon=np.random.rand(b, 2),
            month=6,
        )
        model = Presto()
        _ = model(x)

    def test_forward_from_predictor_with_s1_only(self):
        b, t, h, w = 8, 4, 1, 1

        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_bands)),
            latlon=np.random.rand(b, 2),
            month=6,
        )
        model = Presto()
        _ = model(x)
