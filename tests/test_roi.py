"""ROI 重采样的边界、内存、并发与兼容性测试（补充可见用例）。"""

import concurrent.futures
import math
import time
import tracemalloc

import pytest

import roikit
from roikit import EmptyRoiError, RoiBoundsError, RoiError


def _crop(img, x, y, w, h):
    return [row[x:x + w] for row in img[y:y + h]]


IMG = [[float(r * 100 + c) for c in range(37)] for r in range(29)]


@pytest.mark.parametrize("scale", [0.5, 0.3, 0.7, 1.0, 1.5, 2.0, 3.3, 0.01])
@pytest.mark.parametrize(
    "rect",
    [(0, 0, 37, 29), (0, 0, 11, 7), (5, 3, 13, 17),
     (36, 28, 1, 1), (2, 2, 35, 27), (1, 0, 36, 1)],
)
def test_roi_matches_resize_crop(scale, rect):
    r = roikit.Resizer(scale)
    x, y, w, h = rect
    assert r.resize_roi(IMG, rect) == r.resize(_crop(IMG, x, y, w, h))


def test_roi_error_kinds_are_distinct():
    r = roikit.Resizer(0.5)
    with pytest.raises(RoiBoundsError):
        r.resize_roi(IMG, (34, 0, 4, 4))
    with pytest.raises(EmptyRoiError):
        r.resize_roi(IMG, (0, 0, 0, 2))
    # 二者互斥，且都可按 ValueError 统一捕获。
    assert not issubclass(RoiBoundsError, EmptyRoiError)
    assert not issubclass(EmptyRoiError, RoiBoundsError)
    assert issubclass(RoiError, ValueError)


def test_empty_image_raises():
    with pytest.raises(ValueError):
        roikit.Resizer(0.5).resize([])
    with pytest.raises(ValueError):
        roikit.Resizer(0.5).resize_roi([], (0, 0, 1, 1))


def test_roi_does_not_copy_whole_image():
    big = [[float(r * 2000 + c) for c in range(2000)] for r in range(2000)]
    r = roikit.Resizer(0.5)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    out = r.resize_roi(big, (10, 10, 64, 64))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # 峰值增量远小于整图（2000x2000x8 字节 ≈ 30.5 MiB）。
    assert peak - base < 0.05 * (2000 * 2000 * 8)
    assert len(out) == 32 and all(len(row) == 32 for row in out)


def test_concurrent_calls_match_serial():
    big = [[float(r * 2000 + c) for c in range(2000)] for r in range(2000)]
    r = roikit.Resizer(0.5)
    rects = [(i % 60, (i * 7) % 60, 1 + i % 40, 1 + (i * 3) % 40)
             for i in range(200)]
    serial = [r.resize_roi(big, q) for q in rects]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        concurrent_results = list(ex.map(lambda q: r.resize_roi(big, q), rects))
    assert concurrent_results == serial


def test_legacy_resize_unchanged():
    r = roikit.Resizer(0.5)
    snapshot = [row[:] for row in IMG]
    assert r.resize(IMG) == roikit.Resizer(0.5).resize(IMG)
    assert IMG == snapshot
    assert roikit.Resizer(1.0).resize(IMG) == IMG
    assert r.resize_roi(IMG, (0, 0, 37, 29)) == r.resize(IMG)


def test_non_square_roi_shape():
    out = roikit.Resizer(0.5).resize_roi(IMG, (3, 5, 20, 7))
    assert len(out) == math.ceil(7 * 0.5)
    assert all(len(row) == math.ceil(20 * 0.5) for row in out)


def test_huge_rect_rejected_without_allocating():
    start = time.monotonic()
    with pytest.raises(RoiBoundsError):
        roikit.Resizer(0.5).resize_roi(IMG, (0, 0, 1 << 62, 1))
    assert time.monotonic() - start < 1.0
