import yaml
from pathlib import Path


class YAMLLoader:
    """
    通用 YAML 文件读取类
    用法：
        loader = YAMLLoader("/path/to/config.yaml")
        data = loader.load()
    """

    def __init__(self, file_path: str):
        # 初始化，保存路径和 logger
        self.file_path = Path(file_path)

    def load(self) -> dict[str, object]:
        """
        加载 YAML 文件为字典。
        文件不存在或格式错误会抛出异常。
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.file_path}")

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing failed: {e}") from e
