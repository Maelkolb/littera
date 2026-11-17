import math

import numpy as np
from lines_dejavu import render
from PIL import Image

from littera.geometry import GeometryConfig, best_gap, ink_edge, prepare


def degrees(slant):
    return math.degrees(math.atan(slant))


def test_strip_shape_padding_and_round_trip():
    image, cuts, centre = render("Hallo schrieb mir den Tag")
    strip = prepare(image)
    assert strip.height == 96 and strip.width % 4 == 0
    assert strip.image[:, :12].max() < 0.05 and strip.image[:, -12:].max() < 0.05
    xs = np.array([10.0, 80.0, 200.0])
    back, _ = strip.to_crop(strip.strip_u(xs), strip.v_mid)
    assert np.allclose(back, xs)
    assert abs(strip.centre_y.mean() - centre) < 3


def test_slant_is_estimated_and_sheared_out():
    for slant in (20.0, -15.0, 0.0):
        image, cuts, centre = render("Hallo schrieb mir den Tag", slant_deg=slant)
        strip = prepare(image)
        assert abs(degrees(strip.slant) - slant) < 3.0, slant
        x_top, _ = strip.to_crop(100.0, 0.0)
        x_bottom, _ = strip.to_crop(100.0, 96.0)
        if slant:
            assert (x_top - x_bottom) * np.sign(slant) > 0


def test_box_crop_with_neighbouring_lines_keeps_the_main_line():
    plain = prepare(render("Hallo schrieb mir den Tag")[0])
    image, cuts, centre = render("Hallo schrieb mir den Tag", neighbours=True)
    strip = prepare(image)
    assert abs(strip.centre_y.mean() - centre) < 4
    assert 0.75 < strip.scale / plain.scale < 1.35


def test_baseline_and_polygon_guide_a_skewed_line():
    image, cuts, centre = render("Hallo schrieb mir den Tag", neighbours=True)
    angle = 4.0
    rotated = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)
    w, h = image.size
    cx, cy = rotated.width / 2, rotated.height / 2
    theta = math.radians(angle)

    def turn(x, y):
        dx, dy = x - w / 2, y - h / 2
        return cx + dx * math.cos(theta) + dy * math.sin(theta), cy - dx * math.sin(theta) + dy * math.cos(theta)

    baseline_y = centre + 0.25 * 24
    baseline = [turn(5, baseline_y), turn(w - 5, baseline_y)]
    polygon = [turn(0, centre - 16), turn(w, centre - 16), turn(w, centre + 12), turn(0, centre + 12)]
    for kwargs in ({"baseline": baseline}, {"polygon": polygon}, {"polygon": polygon, "baseline": baseline}):
        strip = prepare(rotated, **kwargs)
        for x in (40.0, w - 40.0):
            x_turned, y_turned = turn(x, centre)
            assert abs(np.interp(x_turned, strip.centre_x, strip.centre_y) - y_turned) < 3, kwargs
        slope = (strip.centre_y[-1] - strip.centre_y[0]) / (strip.centre_x[-1] - strip.centre_x[0])
        assert abs(slope + math.tan(theta)) < 0.03, kwargs
    masked = prepare(rotated, polygon=polygon)
    assert masked.image[: int(masked.v_mid - 2.0 * masked.core)].sum() < 0.05 * masked.image.sum()


def test_gap_and_edge_helpers():
    profile = np.array([5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5], dtype=float)
    assert best_gap(profile, 1, 10, 4.5, 0.5) == (3.0, 6.0)
    assert best_gap(np.array([3, 2, 1, 2, 3.0]), 0, 5, 2.0, 0.5) == (2.0, 3.0)
    assert ink_edge(profile, 4.5, 3, 0.5, "start") == 6.0
    assert ink_edge(profile, 4.5, 3, 0.5, "end") == 3.0
    assert GeometryConfig().core == GeometryConfig().height * GeometryConfig().core_ratio
