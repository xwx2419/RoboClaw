"""Load optional markdown extensions for agent memory prompts."""

from pathlib import Path

_REFS_DIR = Path(__file__).resolve().parent / "references"
_ROBOCLAW_MD = _REFS_DIR / "RoboClaw.md"


def load_roboclaw_self_knowledge_extension() -> str:
    """Return RoboClaw.md body for appending to SELF_KNOWLEDGE (after template format)."""
    if not _ROBOCLAW_MD.is_file():
        return ""
    return _ROBOCLAW_MD.read_text(encoding="utf-8").strip()
