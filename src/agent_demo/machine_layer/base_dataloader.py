from abc import ABC, abstractmethod


class BaseDataloader(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def robot(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def slam(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def camera(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_id_auto_plus(self):
        raise NotImplementedError

    @abstractmethod
    def get_latest_concatenate_image_base64(self, need_save: bool = False):
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):  # 必须调用，否则会在后台阻塞程序退出
        raise NotImplementedError
