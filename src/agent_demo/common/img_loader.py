import cv2
import base64
import numpy as np
import os
import re
import logging

logger = logging.getLogger(__name__)


class Imgloader:

    @classmethod
    def load_and_resize_to_base64(
        cls,
        path: str,
        target_size: tuple[int, int],  # 目标尺寸 (width, height)
    ) -> tuple[str, str]:
        # 1. 加载图片为 NumPy 数组
        image, type = cls.load_file_as_array(path)
        if image is None:
            raise ValueError(f"Failed to load image from: {path}")
        # 2. 获取原始尺寸和目标尺寸
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size
        # 3. 计算等比例缩放后的新尺寸
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        # 4. 执行缩放（使用 OpenCV 的 INTER_AREA 插值，适合缩小）
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # 5. 转为 Base64
        return cls.to_base64(resized_image, type)

    @classmethod
    def load_file_as_base64(cls, path: str) -> tuple[str, str]:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File does not exist: {abs_path}")
        file_type = os.path.splitext(abs_path)[1].lower().lstrip('.')
        if file_type not in ('png', 'jpg', 'jpeg'):
            raise ValueError(f"Unsupported file type: {abs_path}")
        with open(file=abs_path, mode="rb") as file:
            file_content: bytes = file.read()
            b64_str: str = base64.b64encode(file_content).decode("utf-8")
        logger.debug(f"Loaded file as base64: {abs_path} (type: {file_type})")
        return b64_str, file_type  # 返回 Base64 和文件类型

    @classmethod
    def load_file_as_array(cls, path: str) -> tuple[np.ndarray, str]:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File does not exist: {abs_path}")
        file_type = os.path.splitext(abs_path)[1].lower().lstrip('.')
        if file_type not in ('png', 'jpg', 'jpeg'):
            raise ValueError(f"Unsupported file type: {abs_path}")
        image: np.ndarray = cv2.imread(abs_path)
        logger.debug(f"Loaded file as array: {abs_path} (type: {file_type})")
        return image, file_type  # 返回数组和文件类型

    @classmethod
    def info(cls, image: np.ndarray):
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        color_type = "Grayscale" if channels == 1 else "Color"
        dtype = image.dtype
        pixel_count = width * height
        total_tokens_matrix = pixel_count * channels

        logger.info(f"Resolution: {width} x {height}")
        logger.info(f"Number of channels: {channels}")
        logger.info(f"Image type: {color_type}")
        logger.info(f"Data type: {dtype}")
        logger.info(f"Total pixels: {pixel_count}")
        logger.info(f"Total_tokens_matrix: {total_tokens_matrix}")

    @classmethod
    def to_base64(cls, image: np.ndarray, image_format: str = ".jpg") -> tuple[str, str]:
        if not image_format.startswith("."):
            image_format = "." + image_format
        success, encoded_img = cv2.imencode(image_format, image)
        if not success:
            raise ValueError(f"imencode {image_format} faile")
        b64_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
        return b64_str, image_format

    @classmethod
    def from_base64(cls, b64_str: str) -> np.ndarray:
        pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
        if not pattern.match(b64_str):
            raise ValueError("提供的字符串不是有效的 Base64 编码")
        img_data = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img

    @classmethod
    def is_base64_encoded(cls, b64_str: str) -> bool:
        pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
        if pattern.match(b64_str):
            return True
        else:
            return False
