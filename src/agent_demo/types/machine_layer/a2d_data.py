from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class A2DData(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    frame_id: int = 0
    image_type: Literal["jpeg", "png", "jpg"]
    image_ts: Optional[int] = None
    head_image: Optional[np.ndarray] = None
    left_wrist_image: Optional[np.ndarray] = None
    right_wrist_image: Optional[np.ndarray] = None
    concatenated_image: Optional[np.ndarray] = None
    concatenated_image_base64: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}  # 允许使用 np.ndarray 等自定义类型

    def img_info(self) -> None:
        if self.head_image is None or self.left_wrist_image is None or self.right_wrist_image is None:
            logger.warning("One or more input images are None.")
            return

        head_h, head_w = self.head_image.shape[:2]
        left_h, left_w = self.left_wrist_image.shape[:2]
        right_h, right_w = self.right_wrist_image.shape[:2]

        logger.info(f"============ {self.frame_id} ============")
        logger.debug(f"head_image image size: width={head_w}, height={head_h}")
        logger.debug(f"left_wrist_image image size: width={left_w}, height={left_h}")
        logger.debug(f"right_wrist_image image size: width={right_w}, height={right_h}")

        if self.concatenated_image is not None:
            concatenated_h, concatenated_w = self.concatenated_image.shape[:2]
            logger.debug(f"concatenated_image image size: width={concatenated_w}, height={concatenated_h}")
        else:
            logger.info("concatenated_image is None.")

        if self.concatenated_image_base64 is not None:
            logger.debug(f"concatenated_image_base64 length: {len(self.concatenated_image_base64)}")
        else:
            logger.info("concatenated_image_base64 is None.")
