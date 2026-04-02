# CoRobot MCP Server

这是一个用于连接 CoRobot 系统的 MCP (Model Context Protocol) 服务器。

## 功能

该服务器提供了以下工具，用于控制 CoRobot PolicyTask：

- `start_task`: 启动 PolicyTask 任务
- `stop_task`: 停止 PolicyTask 任务
- `reset_task`: 重置 PolicyTask 到初始状态
- `set_prompt`: 设置任务提示词
- `get_prompt`: 获取当前任务提示词
- `set_evaluate_params`: 设置评估参数（包括策略服务器配置）
- `get_status`: 获取 PolicyTask 的详细状态

## 配置

服务器默认连接到 `http://localhost:8765` 的 CoRobot API。

确保 CoRobot 服务正在运行，并且可以通过该地址访问。

## 使用

该服务器通过 MCP 协议与 Olympus Agent 系统集成，在 `ormcp_services.json` 中配置后即可使用。

