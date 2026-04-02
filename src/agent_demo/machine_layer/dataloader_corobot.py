"""
CoRobot DataLoader - 使用 a2d_sdk.CosineCamera 获取机器人数据
"""

from typing import Literal, Optional
import logging
import base64
import os
import subprocess
import sys
import numpy as np
import cv2
from .base_dataloader import BaseDataloader
from agent_demo.types.machine_layer import A2DData

logger = logging.getLogger(__name__)

try:
    from corobot.utils.dds_setting import dds_env_set
except ImportError:
    # 如果无法导入，使用本地实现
    def dds_env_set():
        """设置 DDS 环境变量"""
        logger.info("setting robot dds environment variables...")
        try:
            local_ip = subprocess.check_output(["ip", "-o", "-4", "addr", "list"]).decode("utf-8").strip().splitlines()
            local_ip = [line.split()[3].split("/")[0] for line in local_ip if "10.42.0." in line]
            local_ip = local_ip[0] if local_ip else None

            A2D_LOCATOR_IP = "10.42.0.101"
            if local_ip:
                LOCATOR_IP = local_ip
                AORTA_DISCOVERY_URI = f"http://{A2D_LOCATOR_IP}:2379"
                os.environ["LOCATOR_IP"] = LOCATOR_IP
                os.environ["AORTA_DISCOVERY_URI"] = AORTA_DISCOVERY_URI
                os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
                logger.info(
                    f"[CoRobot] DDS env set: LOCATOR_IP={LOCATOR_IP}, AORTA_DISCOVERY_URI={AORTA_DISCOVERY_URI}"
                )
            else:
                raise RuntimeError("no ip in 10.42.0.* found, can not communicate with robot")
        except Exception as e:
            logger.error(f"Error setting environment variables: {e}")


# 从 corobot 虚拟环境导入 a2d_sdk
corobot_venv_paths = [
    "/home/easter/corobot/lib/python3.10/site-packages",
    os.path.expanduser("~/corobot/lib/python3.10/site-packages"),
]

A2D_SDK_AVAILABLE = False
Camera = None
Robot = None
Slam = None

for venv_path in corobot_venv_paths:
    if os.path.exists(venv_path) and venv_path not in sys.path:
        sys.path.insert(0, venv_path)
        break

try:
    from a2d_sdk.robot import CosineCamera as Camera
    from a2d_sdk.robot import RobotDds as Robot
    from a2d_sdk.robot import Slam as Slam

    A2D_SDK_AVAILABLE = True
    logger.info("[CoRobot] 从 corobot 虚拟环境导入 a2d_sdk 成功")
except ImportError as e:
    logger.warning(f"无法从 corobot 虚拟环境导入 a2d_sdk: {e}")
    A2D_SDK_AVAILABLE = False


class DataLoaderCoRobot(BaseDataloader):
    """
    CoRobot DataLoader - 使用 a2d_sdk.CosineCamera 获取机器人观察数据
    """

    def __init__(
        self, format: Literal["jpeg", "png", "jpg"] = "jpeg", base_url: str = None, env_config: dict | None = None
    ):
        if not A2D_SDK_AVAILABLE:
            raise RuntimeError("a2d_sdk 不可用，无法初始化 DataLoaderCoRobot")

        # 设置 DDS 环境变量
        dds_env_set()

        self.supported_cameras = ["head", "hand_left", "hand_right"]
        # 延迟初始化 Robot 对象，避免订阅不需要的 HAL 话题（减少 DDS 缓冲区溢出警告）
        # Robot 对象主要用于 shutdown，实际使用中主要依赖 Camera 和 Slam
        self._robot: Optional[Robot] = None
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

        logger.info(
            "[CoRobot] DataLoaderCoRobot initialized - 使用 a2d_sdk.CosineCamera (Robot 对象延迟初始化，减少 DDS 订阅)"
        )

    @property
    def robot(self) -> Robot:
        """延迟初始化 Robot 对象（仅在需要时创建，避免订阅不需要的 HAL 话题）"""
        if self._robot is None:
            logger.debug("[CoRobot] 延迟初始化 Robot 对象")
            self._robot = Robot()
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

    def shutdown(self) -> None:
        """关闭连接（必须调用，否则会在后台阻塞程序退出）"""
        # 如果 Robot 对象已初始化，则关闭它
        if self._robot is not None:
            self._robot.shutdown()
            self._robot = None
        logger.info("[CoRobot] DataLoaderCoRobot.shutdown() called")

    async def get_latest_concatenate_image_base64(self, need_save: bool = False) -> Optional[A2DData]:
        """
        从机器人获取观察数据并转换为 A2DData 格式
        """
        a2d_data = self.get_latest_camera_data()
        if a2d_data is None:
            return None

        concatenate_image_base64: str | None = self.get_concatenate_image_base64(a2d_data, need_save)
        if concatenate_image_base64 is None:
            return None

        return a2d_data

    def get_latest_camera_data(self) -> Optional[A2DData]:
        """
        获取最新相机数据
        """
        try:
            # 获取 head 相机的最新图像和时间戳
            head_image, head_img_ts = self._camera.get_latest_image("head")
            if head_image is None or head_img_ts is None:
                return None

            # 使用 head 的时间戳，获取其他相机的时间对齐图像
            left_wrist_image, left_img_ts = self._camera.get_image_nearest("hand_left", head_img_ts)
            right_wrist_image, right_img_ts = self._camera.get_image_nearest("hand_right", head_img_ts)

            # 验证时间对齐的准确性（时间差阈值：1ms = 1e6 纳秒）
            TIME_ALIGNMENT_THRESHOLD_NS = 1e6  # 1ms
            if left_img_ts is not None:
                left_delta = abs(left_img_ts - head_img_ts)
                if left_delta > TIME_ALIGNMENT_THRESHOLD_NS:
                    pass
                    # logger.warning(
                    #     f"[CoRobot] 左手相机时间对齐偏差过大: {left_delta/1e6:.2f}ms (阈值: {TIME_ALIGNMENT_THRESHOLD_NS/1e6:.2f}ms)"
                    # )

            if right_img_ts is not None:
                right_delta = abs(right_img_ts - head_img_ts)
                if right_delta > TIME_ALIGNMENT_THRESHOLD_NS:
                    pass
                    # logger.warning(
                    #     f"[CoRobot] 右手相机时间对齐偏差过大: {right_delta/1e6:.2f}ms (阈值: {TIME_ALIGNMENT_THRESHOLD_NS/1e6:.2f}ms)"
                    # )

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
        except Exception as e:
            logger.error(f"[CoRobot] 获取相机数据出错: {e}")
            return None

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """确保图像是 BGR 格式"""
        # a2d_sdk 返回的是 RGB，需要转换为 BGR
        try:
            if image is None:
                return None
            if len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception:
            return image

    def get_concatenate_image_base64(self, a2d_data: A2DData, need_save: bool = False) -> Optional[str]:
        """
        生成拼接图像的 base64 编码
        """
        encode_image = self._get_concatenate_encode_image(a2d_data, need_save)  # JPEG|PNG
        if encode_image is None:
            return None
        img_base64: str = base64.b64encode(encode_image.tobytes()).decode('utf-8')
        a2d_data.concatenated_image_base64 = img_base64
        return img_base64

    def _get_concatenate_encode_image(self, a2d_data: A2DData, need_save: bool = False) -> Optional[np.ndarray]:
        """
        生成拼接图像的编码格式（JPEG/PNG）
        """
        if a2d_data.head_image is None or a2d_data.left_wrist_image is None or a2d_data.right_wrist_image is None:
            logger.debug("[CoRobot] 部分图像数据缺失，无法生成拼接图像")
            return None

        try:
            # 获取最小高度
            min_height = min(
                a2d_data.left_wrist_image.shape[0], a2d_data.head_image.shape[0], a2d_data.right_wrist_image.shape[0]
            )

            # 缩放图像
            left_resized = self._resize_to_height(a2d_data.left_wrist_image, min_height)
            head_resized = self._resize_to_height(a2d_data.head_image, min_height)
            right_resized = self._resize_to_height(a2d_data.right_wrist_image, min_height)

            # 添加标注
            left_annotated = self._annotate_image(left_resized, "Left Wrist")
            head_annotated = self._annotate_image(head_resized, "Head")
            right_annotated = self._annotate_image(right_resized, "Right Wrist")

            # 拼接
            concatenated = cv2.hconcat([left_annotated, head_annotated, right_annotated])
            a2d_data.concatenated_image = concatenated

            # 编码
            success, encoded_image = cv2.imencode(self._ext, concatenated)
            if not success:
                logger.error("[CoRobot] 图像编码失败")
                return None

            # 保存（如果需要）
            if need_save:
                save_dir = "./img_dir"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"img_{a2d_data.frame_id}{self._ext}"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, concatenated)
                logger.info(f"[CoRobot] 图像已保存到 {save_path}")

            return encoded_image

        except Exception as e:
            logger.error(f"[CoRobot] 生成拼接图像失败: {e}")
            return None

    def _resize_to_height(self, image: np.ndarray, height: int) -> np.ndarray:
        """缩放图像到指定高度"""
        try:
            h, w = image.shape[:2]
            scale = height / h
            return cv2.resize(image, (int(w * scale), height))
        except Exception:
            return image

    def _annotate_image(self, image: np.ndarray, label: str, top_border: int = 15, side_border: int = 3) -> np.ndarray:
        """给图像添加标注"""
        try:
            # 添加边框
            image_with_border = cv2.copyMakeBorder(
                image,
                top=top_border,
                bottom=0,
                left=side_border,
                right=side_border,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            # 添加文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (0, 0, 0)
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = (image_with_border.shape[1] - text_size[0]) // 2
            text_y = (top_border + text_size[1]) // 2
            cv2.putText(image_with_border, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            return image_with_border
        except Exception:
            return image
