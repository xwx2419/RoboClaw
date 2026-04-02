# Paper-style vanity benchmark — long-horizon skill fields

与 RoboClaw 论文中「梳妆台」评测设定对齐的**可直接填进** `long-horizon-execution` 的字段示例。`prompt` 为送给策略/PolicyTask 的自然语言指令（建议英文，便于与常见 VLA 对齐）；`success_check` 为**可观测**判定句，供执行后验证。

> **使用声明**：本文档仅为字段与流程的**参考示例**，不构成唯一标准。实际执行时请以你的真实场景约束（物体摆位、遮挡、相机视角、抓取稳定性、硬件状态）、实验协议与安全规则为准，按需调整任务顺序、`prompt`、`success_check`、重试与重置策略。

> **顺序说明**：默认采用：①妆前乳+关抽屉（primer）②口红插槽（lipstick）③身体乳放置（lotion）④纸巾擦拭（tissue wipe）。

---

## `global_goal`（总目标一句话）

**中文：** 在带分类标签的收纳环境中，完成梳妆台整理：先将妆前乳放入抽屉目标区并关抽屉，再将口红插入窄槽、将身体乳放到指定标签区，最后用纸巾擦掉桌面上爽肤水的渍迹。

**English (one line):** Complete the vanity-table benchmark in execution order: place primer inside the labeled drawer target and fully close the drawer, then insert lipstick into the labeled narrow slot, place body lotion in its labeled region, and finally wipe the spilled toner area with tissue using stable contact and full coverage.

---

## `completion_criteria`（整任务完成的可观测条件）

同时满足：

1. **妆前乳**：管状妆前乳在抽屉内**目标区域**，抽屉**完全关闭**且无异物阻挡。  
2. **口红**：口红**完全、正确**插入**标定窄槽**（无悬空、无卡在一半）。  
3. **身体乳**：乳液瓶离开桌面初始摆放区，且位于**标定给 body lotion 的标签/格位**内，姿态稳定无倾倒风险。  
4. **擦拭**：纸巾对渍迹区域完成**覆盖式擦拭轨迹**，目标区域内可见液体渍迹已清除或达到实验定义的「可接受残留」标准（与实验协议一致）。

---

## `subtask_plan`（有序子任务）

每项字段：`subtask_id` · `objective` · `prompt` · `success_check` · `reset_after` · `max_retries` · `notes`（可选）。

### 1) Primer Placement（执行顺序 1）

| 字段 | 内容 |
|------|------|
| `subtask_id` | `place-primer-close-drawer` |
| `objective` | 妆前乳放入抽屉内目标区并关抽屉 |
| `prompt` | `Pick up the primer tube, place it in the target region inside the labeled primer drawer without blocking the drawer rails, then fully close the drawer. Ensure nothing protrudes that prevents complete closure.` |
| `success_check` | 妆前乳在抽屉内目标区；抽屉**完全关闭**；无因物体干涉导致的缝隙或反弹开启。 |
| `reset_after` | `true` |
| `max_retries` | `2` |
| `notes` | 遮挡与狭窄空间：若无法确认内部状态，可先部分拉开验证再关（以安全与实验规则为准）。 |

### 2) Lipstick Insertion（执行顺序 2）

| 字段 | 内容 |
|------|------|
| `subtask_id` | `insert-lipstick-slot` |
| `objective` | 高精度对齐并将口红插入窄槽 |
| `prompt` | `Align the lipstick with the labeled narrow insertion slot using precise position and orientation, then insert smoothly until fully seated. Maintain alignment through contact; do not force if misaligned—realign and retry.` |
| `success_check` | 口红**完全插入**标定槽位，无半插入、无歪斜卡死；末端状态稳定。 |
| `reset_after` | `true` |
| `max_retries` | `3` |
| `notes` | 容差小，失败多因对齐；可适当提高重试次数并缩短单次 `timeout` 避免长时间顶死。 |

### 3) Body Lotion Placement（执行顺序 3）

| 字段 | 内容 |
|------|------|
| `subtask_id` | `place-body-lotion` |
| `objective` | 大范围抓取-搬运-放置身体乳至标签区 |
| `prompt` | `Pick up the body lotion bottle from the vanity table, move it with a stable grasp, and place it fully inside the compartment or region labeled for body lotion. Keep the bottle upright and release only when it is stably supported.` |
| `success_check` | 身体乳瓶在**标定 body lotion 区域**内，不再依赖夹爪支撑；桌面原位置无残留该瓶。 |
| `reset_after` | `true` |
| `max_retries` | `2` |
| `notes` | 视角变化大，success_check 应结合**最新帧**与标签可见性。 |

### 4) Tissue Wipe（执行顺序 4）

| 字段 | 内容 |
|------|------|
| `subtask_id` | `wipe-toner-spill` |
| `objective` | 用纸巾对渍迹区做稳定接触与覆盖擦拭 |
| `prompt` | `Using a tissue, wipe the designated toner spill region on the table. Maintain stable contact with the surface and follow a wiping trajectory that covers the entire marked area until the spill is cleared per the experiment protocol.` |
| `success_check` | 标定擦拭区域内轨迹覆盖充分；液体渍迹清除或达到协议定义阈值；纸巾与桌面接触过程无危险打滑失控（按安全判据）。 |
| `reset_after` | `true`（整段 benchmark 结束时常用；若后还有子任务则按协议） |
| `max_retries` | `2` |
| `notes` | 强调**连续运动质量**而非单一终点位姿；可与视觉上的「渍迹面积」对比作为辅助检查。 |

---

## `run_dir` / 日志

| 项 | 建议 |
|------|------|
| `run_dir` | `./runs/vanity_paper_<日期或 run_id>/`（或你机器上可写路径） |
| 进度日志 | `run_dir/logs/subtasks.jsonl`：每行一次尝试，至少含 `subtask_id`、`attempt_index`、`prompt`、`status_text`、`verification_result`、`failure_reason`、时间戳（与 `subtask-plan-template.md` 一致） |
| 最终摘要 | `run_dir/summary.md` 或 `summary.json`：完成/未完成子任务列表、重试次数、人工介入记录 |

---





