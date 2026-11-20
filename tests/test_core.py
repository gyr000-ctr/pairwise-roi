import math
import struct

import pytest

import roikit

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[float(r * 4 + c) for c in range(4)] for r in range(4)]


def _r():
    return roikit.Resizer(0.5)


def _crop(img, x, y, w, h):
    return [row[x:x + w] for row in img[y:y + h]]

CASES = [
        ("full_shape", lambda: len(_r().resize(IMG)) == 2),
        ("roi_shape", lambda: len(_r().resize_roi(IMG, (0, 0, 2, 2))) == 1),
        ("roi_matches_crop", lambda: _r().resize_roi(IMG, (0, 0, 2, 2)) == _r().resize(_crop(IMG, 0, 0, 2, 2))),
        ("roi_offset", lambda: _r().resize_roi(IMG, (2, 2, 2, 2)) == _r().resize(_crop(IMG, 2, 2, 2, 2))),
        ("out_of_bounds", lambda: _raises(lambda: _r().resize_roi(IMG, (3, 3, 4, 4)))),
        ("zero_rect", lambda: _raises(lambda: _r().resize_roi(IMG, (0, 0, 0, 2)))),
        ("bad_scale", lambda: _raises(lambda: roikit.Resizer(0.0))),
        ("full_values", lambda: _r().resize(IMG) == [[0.0, 2.0], [8.0, 10.0]]),
        ("roi_values", lambda: _r().resize_roi(IMG, (0, 0, 2, 2)) == [[0.0]]),
        ("type", lambda: isinstance(_r().resize(IMG), list)),
        ("deterministic", lambda: _r().resize(IMG) == _r().resize(IMG)),
        ("no_mutation", lambda: (lambda copy: (_r().resize(IMG), IMG == copy)[1])([row[:] for row in IMG])),
        ("roi_full", lambda: _r().resize_roi(IMG, (0, 0, 4, 4)) == _r().resize(IMG)),
        ("scale_one", lambda: roikit.Resizer(1.0).resize(IMG) == IMG),
        ("negative_rect", lambda: _raises(lambda: _r().resize_roi(IMG, (-1, 0, 2, 2)))),
    ]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
