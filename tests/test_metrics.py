"""Tests for haversine distance metrics."""

import math

from vessel_tracker.metrics import haversine_distance


def test_haversine_zero_distance():
    """test that the haversine distance between a point and itself is 0"""
    assert haversine_distance(48.0, -5.0, 48.0, -5.0) == 0.0


def test_haversine_known_distance_one_degree_latitude():
    """check formula is ok: 1deg of latitude is approx 111km everywhere"""
    d = haversine_distance(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(d, 111.0, abs_tol=2.0)
