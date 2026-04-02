from .json_loader import JSONLoader
from .img_loader import Imgloader
import logging

logger = logging.getLogger(__name__)


class TestTaskLoader:
    def __init__(self, task_config_path: str):
        self.task_config: dict = JSONLoader.load(task_config_path)
        self.task_img_list: list[dict] = []
        for task in self.task_config["test_task_list"]:
            img_base64, img_type = Imgloader.load_file_as_base64(path=task["img_path"])
            img_str = self.task_config["img_str"]
            self.task_img_list.append(
                {
                    "img_str": img_str,
                    "img_base64": img_base64,
                    "img_path": task["img_path"],
                    "img_type": img_type,
                    "evaluation_indicators": task["evaluation_indicators"],
                }
            )

    def info(self):
        for idx, task in enumerate(self.task_img_list):
            logger.info(
                str(idx)
                + "|"
                + task["img_str"]
                + "|"
                + task["img_type"]
                + "|"
                + task["evaluation_indicators"]
                + "|"
                + task["img_path"]
            )
