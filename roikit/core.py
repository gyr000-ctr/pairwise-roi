"""roikit 核心 API（初始未实现）。"""


class Resizer:
    def __init__(self, scale):
        """按比例缩放"""
        raise NotImplementedError

    def resize(self, image):
        """整图缩放"""
        raise NotImplementedError

    def resize_roi(self, image, rect):
        """区域缩放"""
        raise NotImplementedError
