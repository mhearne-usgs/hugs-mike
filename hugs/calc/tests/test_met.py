#!/usr/bin/env python
"""Test the `met` module."""

from hugs.calc import get_wind_components, get_wind_dir, get_wind_speed, snell_angle

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest


def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.])

    assert_array_almost_equal(true_speed, speed, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3., -4.)
    assert_almost_equal(s, 5., 3)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.])

    assert_array_almost_equal(true_dir, direc, 4)


def test_wind_components():
    """Test scalar values."""
    speed1 = 100
    wdir1 = 90
    u1, v1 = get_wind_components(speed1, wdir1)
    np.testing.assert_almost_equal(u1, -100.0)
    np.testing.assert_almost_equal(v1, -6.123233995736766e-15)

    # test array values
    speed2 = np.array([100, 200])
    wdir2 = np.array([90, 0])
    speed2 = [100, 200]
    wdir2 = [90, 0]
    u2, v2 = get_wind_components(speed2, wdir2)
    np.testing.assert_almost_equal(u2, np.array([-100., -0.]))
    np.testing.assert_almost_equal(
        v2, np.array([-6.123234e-15, -2.000000e+02]))

    # test > 360 directions - catch warning
    wdir3 = 375
    speed3 = 100
    with pytest.warns(UserWarning):
        u2, v2 = get_wind_components(speed3, wdir3)


def test_snell_zero():
    """Test for exceptions."""
    incoming = 0
    vtop = 0
    vbottom = 100
    with pytest.raises(ValueError):
        snell_angle(incoming, vtop, vbottom)


if __name__ == '__main__':
    test_wind_components()
    test_snell_zero()
