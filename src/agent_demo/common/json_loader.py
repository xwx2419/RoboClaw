import json
from pathlib import Path


class JSONLoader:

    @staticmethod
    def load(file_path: str) -> dict:
        """
        加载并解析指定路径的 JSON 文件。

        :param file_path: JSON 文件的路径
        :return: 返回文件解析后的字典数据
        :raises FileNotFoundError: 如果文件不存在
        :raises ValueError: 如果 JSON 解析失败
        """
        _file_path: Path = Path(file_path)

        if not _file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with _file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load JSON: {e}") from e
