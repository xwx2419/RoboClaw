import logging

logger = logging.getLogger(__name__)


class BaseSession:
    def __init__(self):
        pass

    # ========== 优雅退出 ==========
    async def shutdown(self) -> None:
        raise NotImplementedError

    async def terminate(self) -> None:
        raise NotImplementedError

    # ========= 意图分类器 ==========
    async def intention_classification(self):
        pass

    # ========= 工作流路由器 ==========
    async def workflow_routing(self):  # 工作流路由器
        pass
