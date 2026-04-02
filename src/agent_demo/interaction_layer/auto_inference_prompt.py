"""Periodic auto-inference prompt shared by TUI and Gradio."""

AUTO_INFERENCE_PROMPT = """\
[Automatic system tick — this is NOT a user-typed message.]

1) If there is **no** robot task currently being executed, monitored, or awaiting visual success/failure judgment, respond with **one short plain sentence only** (no JSON, no multi-section report). Prefer the same language as the latest user messages when obvious. Example meaning: "当前没有需要跟踪的机器人任务。"

2) Only when a task is **clearly in progress** or the user **explicitly** asked for execution, tracking, or visual judgment: use the latest and previous-frame images and follow the active robot-task guidance (including structured JSON **only if** that guidance applies).

Do not treat this tick as a new user command; do not restart unrelated long explanations.
"""
