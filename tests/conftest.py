"""Shared fixtures: the worm dataset as (x, y) landmarks and as complex preshapes."""

import numpy as np
import pytest

from shapekit import load_celegans, preshape, to_complex


@pytest.fixture(scope="session")
def worms():
    points, labels = load_celegans()
    return points, labels


@pytest.fixture(scope="session")
def worm_preshapes(worms):
    points, labels = worms
    return to_complex(preshape(points)), labels
