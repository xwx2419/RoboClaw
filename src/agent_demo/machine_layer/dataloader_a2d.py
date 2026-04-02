from typing import Literal
import cv2
from a2d_sdk.robot import CosineCamera as Camera
from a2d_sdk.robot import RobotDds as Robot
from a2d_sdk.robot import Slam as Slam
from agent_demo.types.machine_layer import A2DData
import numpy as np
import base64
import logging
from .base_dataloader import BaseDataloader
import os

logger = logging.getLogger(__name__)


class DataLoaderA2D(BaseDataloader):
    def __init__(self, format: Literal["jpeg", "png", "jpg"] = "jpeg"):
        self.supported_cameras = ["head", "hand_left", "hand_right"]
        self._robot: Robot = Robot()
        self._slam: Slam = Slam()
        self._camera: Camera = Camera(self.supported_cameras)
        self._frame_id: int = 0
        self._format: Literal["jpeg", "png", "jpg"] = format
        if format.lower() == "jpeg" or format.lower() == "jpg":
            self._ext = ".jpg"
        elif format.lower() == "png":
            self._ext = ".png"
        else:
            raise ValueError("Unsupported format. Use 'jpeg' or 'png'.")

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def slam(self) -> Slam:
        return self._slam

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def frame_id_auto_plus(self) -> int:
        current_frame_id = self._frame_id
        if current_frame_id < 1000000:
            self._frame_id += 1
        else:
            self._frame_id = 0
        return current_frame_id

    def shutdown(self) -> None:  # 必须调用，否则会在后台阻塞程序退出
        self._robot.shutdown()

    async def get_latest_concatenate_image_base64(self, need_save: bool = False) -> A2DData | None:
        a2d_data: A2DData | None = self.get_latest_camera_data()
        if a2d_data is None:
            return None

        concatenate_image_base64: str | None = self.get_concatenate_image_base64(a2d_data)
        if concatenate_image_base64 is None:
            return None

        return a2d_data

    # JPEG/PNG -> (H, W, 3) BGR
    def get_latest_camera_data(self) -> A2DData | None:
        head_image, head_img_ts = self._camera.get_latest_image("head")
        if head_image is None or head_img_ts is None:
            return None
        left_wrist_image, left_img_ts = self._camera.get_image_nearest("hand_left", head_img_ts)
        right_wrist_image, right_img_ts = self._camera.get_image_nearest("hand_right", head_img_ts)

        # 统一转换为 BGR，避免 RGB/BGR 通道错位导致色偏
        head_image = self._ensure_bgr(head_image)
        left_wrist_image = self._ensure_bgr(left_wrist_image)
        right_wrist_image = self._ensure_bgr(right_wrist_image)

        return A2DData(
            frame_id=self.frame_id_auto_plus,
            image_type=self._format,
            image_ts=head_img_ts,
            head_image=head_image,
            left_wrist_image=left_wrist_image,
            right_wrist_image=right_wrist_image,
        )

    def get_concatenate_image_base64(self, a2d_data: A2DData, need_save: bool = False) -> str | None:
        encode_image = self._get_concatenate_encode_image(a2d_data, need_save)  # JPEG|PNG
        if encode_image is None:
            return None
        img_base64: str = base64.b64encode(encode_image.tobytes()).decode('utf-8')
        a2d_data.concatenated_image_base64 = img_base64
        return img_base64

    def _get_concatenate_encode_image(self, a2d_data: A2DData, need_save: bool = False) -> np.ndarray | None:
        if a2d_data.head_image is None or a2d_data.left_wrist_image is None or a2d_data.right_wrist_image is None:
            logger.info("One or more input images are None.")
            return None
        # 获取最小高度
        min_height = min(
            a2d_data.left_wrist_image.shape[0], a2d_data.head_image.shape[0], a2d_data.right_wrist_image.shape[0]
        )
        # 保持宽高比按最小高度进行缩放
        left_resized = self._resize_to_height(a2d_data.left_wrist_image, min_height)
        head_resized = self._resize_to_height(a2d_data.head_image, min_height)
        right_resized = self._resize_to_height(a2d_data.right_wrist_image, min_height)
        # 针对每个图片添加白边和文字标注
        left_annotated = self._annotate_image(left_resized, "Left Wrist")
        head_annotated = self._annotate_image(head_resized, "Head")
        right_annotated = self._annotate_image(right_resized, "Right Wrist")
        # 拼接图像： left_annotated - head_annotated - right_annotated
        concatenated_annotated = cv2.hconcat([left_annotated, head_annotated, right_annotated])
        # 将处理后的图像数据编码为指定的格式
        success, encoded_image = cv2.imencode(self._ext, concatenated_annotated)
        if not success:
            logger.error("Failed to encode concatenated image.")
            return None
        else:
            a2d_data.concatenated_image = concatenated_annotated
        if need_save:
            save_dir = "./img_dir"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"img_{a2d_data.frame_id}{self._ext}"  # 假设 self._ext = '.jpg'
            save_path = os.path.join(save_dir, filename)
            # 使用 cv2.imwrite 保存图像
            cv2.imwrite(save_path, concatenated_annotated)
            logger.info(f"Image saved to {save_path}")
        return encoded_image  # JPEG|PNG

    def show_encoded_image(self, encode_image: np.ndarray, format: str = "jpeg") -> None:
        # 解码为图像（模拟真实图像接收情况）
        decoded = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode encoded image")

        # 显示图像
        cv2.imshow("Encoded Image", decoded)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    def _resize_to_height(self, image: np.ndarray, height):
        h, w = image.shape[:2]
        scale = height / h
        return cv2.resize(image, (int(w * scale), height))

    def _annotate_image(self, image: np.ndarray, label: str, top_border: int = 15, side_border: int = 3) -> np.ndarray:
        # 只在顶部添加较宽白色边框，用于放文字
        image_with_border = cv2.copyMakeBorder(
            image,
            top=top_border,  # 顶部边框较大用于放文字
            bottom=0,
            left=side_border,
            right=side_border,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),  # 白色
        )
        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 0, 0)  # 黑色文字
        # 计算文字尺寸和位置（居中）
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (image_with_border.shape[1] - text_size[0]) // 2
        text_y = (top_border + text_size[1]) // 2
        # 添加文字
        cv2.putText(image_with_border, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        return image_with_border

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        # 有些 SDK 返回 RGB，OpenCV 处理为 BGR，做一次 RGB->BGR 转换以消除偏色
        try:
            if image is None:
                return image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 简单启发式：尝试从 RGB 转到 BGR；若原本就是 BGR，转换会导致颜色互换，因此仅做一次统一转换
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception:
            return image
