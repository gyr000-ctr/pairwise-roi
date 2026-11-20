"""roikit 核心 API：整图与区域最邻近重采样。

设计要点
--------
* 像素映射：最邻近，目标像素 ``d`` 取源下标 ``int(d * in / out)``。
* ``resize_roi(image, r)`` 等价于 ``resize(crop(image, r))``：直接按该
  区域的几何计算源坐标，逐像素读取 ``image``，既不物化裁剪副本，也不
  物化整图缩放结果；输出列表大小只与区域相关。
* 并发：``Resizer`` 实例不可变（``__slots__`` + 只读属性），单次调用的
  全部中间状态都是局部变量，因此同一实例并发调用与串行结果一致。
* 错误分类（均可统一按 ``ValueError`` 捕获）：
    - ``ValueError``：图像为空 / 行不齐 / 宽高非法；
    - ``TypeError``：图像不是二维序列；
    - ``RoiError``（``ValueError`` 子类）：矩形结构非法；
    - ``RoiBoundsError``（``RoiError``）：矩形越界或为负；
    - ``EmptyRoiError``（``RoiError``）：矩形宽或高为 0。
  越界与空区域分别使用不同子类，调用方可精确区分。
"""

import math

__all__ = [
    "Resizer",
    "RoiError",
    "RoiBoundsError",
    "EmptyRoiError",
]


class RoiError(ValueError):
    """矩形参数非法（结构或取值）。"""


class RoiBoundsError(RoiError):
    """矩形超出图像范围。"""


class EmptyRoiError(RoiError):
    """矩形宽或高为 0。"""


def _check_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(
            "scale 必须是实数，得到 %r" % (type(scale).__name__,))
    # NaN、正负无穷与 0 一样无法得到正的目标尺寸。
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须是有限正实数，得到 %r" % (scale,))
    return float(scale)


def _check_image(image):
    """返回 ``(height, width)``，只读取尺寸，不复制像素。"""
    if not isinstance(image, (list, tuple)):
        raise TypeError(
            "image 必须是二维序列，得到 %r" % (type(image).__name__,))
    if not image:
        raise ValueError("image 不能为空")
    width = None
    for row in image:
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                "image 的每一行必须是序列，得到 %r"
                % (type(row).__name__,))
        n = len(row)
        if n == 0:
            raise ValueError("image 的每一行必须非空")
        if width is None:
            width = n
        elif n != width:
            raise ValueError("image 各行宽度不一致：%d != %d" % (n, width))
    return len(image), width


def _check_rect(rect):
    """校验矩形结构并返回 ``(x, y, w, h)``。"""
    if isinstance(rect, (str, bytes, bytearray)) or not hasattr(rect, "__len__"):
        raise RoiError(
            "rect 必须是含 4 个整数的序列，得到 %r"
            % (type(rect).__name__,))
    if len(rect) != 4:
        raise RoiError(
            "rect 必须含 4 个整数 (x, y, w, h)，得到长度 %d" % len(rect))
    values = []
    for v in rect:
        if isinstance(v, bool):
            raise RoiError("rect 分量不能是 bool")
        if isinstance(v, int):
            values.append(v)
        elif isinstance(v, float) and math.isfinite(v) and v.is_integer():
            # 兼容 2.0 这类整值浮点，非整值浮点无法对应像素。
            values.append(int(v))
        else:
            raise RoiError(
                "rect 的每个分量必须是整数，得到 %r" % (v,))
    return values[0], values[1], values[2], values[3]


def _source_index(i, in_size, out_size):
    """目标下标 ``i`` 对应的源下标（最邻近，等价于 ``int(i / scale)``）。"""
    j = int(i * in_size / out_size)
    # 浮点误差保护：理论上不会越界，仍显式裁剪。
    if j < 0:
        return 0
    if j >= in_size:
        return in_size - 1
    return j


def _target_size(in_size, scale):
    """向上取整：保证 scale=1 不丢像素，且结果长度符合比例。"""
    return max(1, int(math.ceil(in_size * scale)))


class Resizer:
    """按固定比例缩放图像；实例不可变，可安全并发调用。"""

    __slots__ = ("_scale",)

    def __init__(self, scale):
        """按比例缩放；``scale`` 须为有限正实数。"""
        self._scale = _check_scale(scale)

    @property
    def scale(self):
        return self._scale

    def resize(self, image):
        """整图缩放，返回新的二维列表，不修改 ``image``。"""
        height, width = _check_image(image)
        out_w = _target_size(width, self._scale)
        out_h = _target_size(height, self._scale)
        scale = self._scale
        out = []
        append = out.append
        for i in range(out_h):
            src_row = image[_source_index(i, height, out_h)]
            if scale == 1.0:
                # scale==1 时逐元素原样返回，保持与输入逐字节一致。
                append(list(src_row))
            else:
                append([float(src_row[_source_index(k, width, out_w)])
                        for k in range(out_w)])
        return out

    def resize_roi(self, image, rect):
        """只对 ``rect`` 覆盖区域做重采样。

        结果与 ``crop(resize(image), ...)`` 无关，而是逐元素等价于
        ``resize(image[y:y+h][x:x+w])``：仅读取该区域的源像素，输出大小
        只与区域相关，不会为局部运算复制整图。
        """
        height, width = _check_image(image)
        x, y, w, h = _check_rect(rect)

        # 先定位（负值 / 越界），再判断退化尺寸，使错误分类更符合直觉。
        if x < 0 or y < 0 or w < 0 or h < 0:
            raise RoiBoundsError(
                "区域不能为负（out of bounds）：x=%d, y=%d, w=%d, h=%d"
                % (x, y, w, h))
        if x + w > width or y + h > height:
            raise RoiBoundsError(
                "区域越界（out of bounds）：rect=(%d, %d, %d, %d) 超出 "
                "图像 %dx%d" % (x, y, w, h, width, height))
        if w == 0 or h == 0:
            raise EmptyRoiError(
                "区域宽高必须为正（empty roi）：w=%d, h=%d" % (w, h))

        out_w = _target_size(w, self._scale)
        out_h = _target_size(h, self._scale)

        out = []
        append = out.append
        if self._scale == 1.0:
            # 与整图 scale=1 的整行浅拷贝保持一致，只取区域内的列。
            for i in range(out_h):
                append(list(image[y + _source_index(i, h, out_h)]
                             [x:x + w]))
            return out

        col_src = [x + _source_index(k, w, out_w) for k in range(out_w)]
        for i in range(out_h):
            src_row = image[y + _source_index(i, h, out_h)]
            append([float(src_row[c]) for c in col_src])
        return out
