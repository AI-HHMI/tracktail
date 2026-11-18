"""Regression tests: metric averaging under numpy>=2, where float() on a 1-element
1-D array raises TypeError ('only 0-dimensional arrays can be converted to Python scalars')."""
import math

import numpy as np
import torch

from tracktail.model.eval_metrics import get_delta_x, get_eval_metrics
from tracktail.model.train_utils import _scalar_metric, average_metrics


def test_average_metrics_accepts_one_element_arrays():
    dicts = [{'val/delta_x_1': np.array([0.25]), 'val/mte': 1.0},
             {'val/delta_x_1': np.array([0.75]), 'val/mte': np.float64(3.0)}]
    out = average_metrics(dicts, 'val/')
    assert out == {'val/delta_x_1_avg': 0.5, 'val/mte_avg': 2.0}
    assert all(type(v) is float for v in out.values())


def test_average_metrics_named_nan_safe():
    dicts = [{'val/jaccard_2': np.array([0.5])}, {'val/jaccard_2': np.array([np.nan])}]
    out = average_metrics(dicts, 'val/', name='ds', nan_safe=True)
    assert out == {'val_ds/jaccard_2_avg': 0.5}


def test_scalar_metric_variants():
    assert _scalar_metric('a', np.array([2.0])) == 2.0
    assert _scalar_metric('a', np.array(2.0)) == 2.0
    assert _scalar_metric('a', torch.tensor([2.0])) == 2.0
    assert _scalar_metric('a', 3) == 3.0
    assert _scalar_metric('a', np.array([1.0, np.nan, 3.0])) == 2.0
    assert math.isnan(_scalar_metric('a', np.array([])))


def test_eval_metrics_are_python_floats():
    rng = np.random.default_rng(0)
    B, T, N = 2, 6, 5
    ct = torch.as_tensor(rng.normal(size=(B, T, N, 3)), dtype=torch.float32)
    cp = ct + 0.5 * torch.as_tensor(rng.normal(size=(B, T, N, 3)), dtype=torch.float32)
    vt = torch.ones(B, T, N, 1, dtype=torch.bool)
    vp = torch.full((B, T, N, 1), 3.0)
    m = get_eval_metrics(vp, vt, cp, ct)
    for k, v in m.items():
        assert np.ndim(v) == 0, (k, v)
        float(v)
    assert isinstance(get_delta_x(cp.numpy(), ct.numpy(), vt.numpy(), 1.0), float)
    average_metrics([m, m], 'eval/')
