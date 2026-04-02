from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re

import yaml

from agent_demo.common.yaml_loader import YAMLLoader

logger = logging.getLogger(__name__)

_SKILL_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)
_INLINE_SKILL_TOKEN_RE = re.compile(r"(?<![\w-])\$([A-Za-z0-9][A-Za-z0-9-]*)")


@dataclass(slots=True)
class LocalSkill:
    name: str
    description: str
    skill_dir: Path
    skill_md_path: Path
    body: str
    display_name: str = ""
    short_description: str = ""
    default_prompt: str = ""
    scripts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    linked_skills: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "skill_dir": str(self.skill_dir),
            "skill_md_path": str(self.skill_md_path),
            "display_name": self.display_name,
            "short_description": self.short_description,
            "resource_counts": {
                "scripts": len(self.scripts),
                "references": len(self.references),
                "assets": len(self.assets),
            },
        }

    def detail_dict(self) -> dict[str, object]:
        return {
            **self.summary_dict(),
            "default_prompt": self.default_prompt,
            "body": self.body,
            "scripts": self.scripts,
            "references": self.references,
            "assets": self.assets,
            "linked_skills": self.linked_skills,
        }


@dataclass(slots=True)
class ExpandedSkillMessage:
    message: str
    requested_skills: tuple[str, ...]
    included_skills: tuple[str, ...]
    missing_requested_skills: tuple[str, ...]
    missing_referenced_skills: tuple[str, ...]


class LocalSkillRegistry:
    _REFERENCE_MAX_FILES = 5
    _REFERENCE_MAX_CHARS_PER_FILE = 4000
    _REFERENCE_MAX_TOTAL_CHARS = 12000

    def __init__(self, configured_paths: list[str] | None = None, workspace_root: str | None = None):
        self._configured_paths: list[str] = configured_paths or []
        self._workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self._skills_by_name: dict[str, LocalSkill] = {}

    @property
    def search_roots(self) -> list[Path]:
        roots: list[Path] = []

        env_paths = os.environ.get("OLYMPUS_SKILLS_PATHS", "")
        if env_paths:
            for raw_path in env_paths.split(os.pathsep):
                if raw_path.strip():
                    roots.append(Path(raw_path).expanduser())

        for raw_path in self._configured_paths:
            if raw_path.strip():
                roots.append(Path(raw_path).expanduser())

        roots.extend(
            [
                self._workspace_root / "skills",
                self._workspace_root / "skills" / "public",
            ]
        )

        unique_roots: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                unique_roots.append(resolved)
                seen.add(resolved)
        return unique_roots

    def refresh(self) -> dict[str, LocalSkill]:
        skills: dict[str, LocalSkill] = {}
        seen_dirs: set[Path] = set()

        for root in self.search_roots:
            if not root.exists() or not root.is_dir():
                continue

            for skill_md_path in sorted(root.rglob("SKILL.md")):
                skill_dir = skill_md_path.parent.resolve()
                if skill_dir in seen_dirs:
                    continue
                seen_dirs.add(skill_dir)

                try:
                    skill = self._load_skill(skill_md_path.resolve())
                except Exception as exc:
                    logger.warning("Skip invalid skill %s: %s", skill_md_path, exc)
                    continue

                if skill.name in skills:
                    logger.warning(
                        "Duplicate skill name '%s' found at %s; keeping %s",
                        skill.name,
                        skill.skill_dir,
                        skills[skill.name].skill_dir,
                    )
                    continue

                skills[skill.name] = skill

        self._skills_by_name = skills
        return self._skills_by_name

    def list_skills(self, refresh: bool = True) -> list[LocalSkill]:
        if refresh or not self._skills_by_name:
            self.refresh()
        return list(self._skills_by_name.values())

    def get_skill(self, skill_name: str, refresh: bool = True) -> LocalSkill | None:
        if refresh or not self._skills_by_name:
            self.refresh()
        return self._skills_by_name.get(skill_name)

    def suggest(self, prefix: str = "", refresh: bool = True) -> list[LocalSkill]:
        if refresh or not self._skills_by_name:
            self.refresh()
        normalized_prefix = prefix.strip().lower()
        return [
            skill
            for skill in sorted(self._skills_by_name.values(), key=lambda item: item.name)
            if not normalized_prefix or skill.name.lower().startswith(normalized_prefix)
        ]

    def extract_skill_names(self, text: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for match in _INLINE_SKILL_TOKEN_RE.finditer(text):
            name = match.group(1)
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def expand_inline_request(
        self,
        message: str,
        refresh: bool = True,
        execution_context: str | None = None,
    ) -> ExpandedSkillMessage:
        if refresh or not self._skills_by_name:
            self.refresh()

        requested_skills = self.extract_skill_names(message)
        if not requested_skills:
            return ExpandedSkillMessage(
                message=message,
                requested_skills=(),
                included_skills=(),
                missing_requested_skills=(),
                missing_referenced_skills=(),
            )

        available_requested = [name for name in requested_skills if name in self._skills_by_name]
        missing_requested = [name for name in requested_skills if name not in self._skills_by_name]
        ordered_skills, missing_references = self._resolve_requested_skills(available_requested)
        return ExpandedSkillMessage(
            message=self._build_inline_skill_message(
                original_message=message,
                requested_skills=requested_skills,
                ordered_skills=ordered_skills,
                missing_references=missing_references,
                execution_context=execution_context,
            ),
            requested_skills=tuple(requested_skills),
            included_skills=tuple(skill.name for skill in ordered_skills),
            missing_requested_skills=tuple(missing_requested),
            missing_referenced_skills=tuple(missing_references),
        )

    def build_service_description(self, max_items: int = 6) -> str:
        skills = self.list_skills(refresh=True)
        base = "本地 Skill 服务，负责发现并调用符合 Codex SKILL.md 规范的技能包。"
        if not skills:
            return base + " 当前未发现可用 skill。"

        skill_fragments = [f"{skill.name}: {skill.description}" for skill in skills[:max_items]]
        if len(skills) > max_items:
            skill_fragments.append(f"其余 {len(skills) - max_items} 个 skill 可通过 list_skills 查看")
        return base + " 已发现 skill: " + "；".join(skill_fragments)

    def _resolve_requested_skills(self, requested_skills: list[str]) -> tuple[list[LocalSkill], list[str]]:
        ordered_skills: list[LocalSkill] = []
        missing_references: list[str] = []
        visited: set[str] = set()
        missing_seen: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            skill = self._skills_by_name.get(name)
            if skill is None:
                if name not in missing_seen:
                    missing_references.append(name)
                    missing_seen.add(name)
                return

            ordered_skills.append(skill)
            for linked_skill_name in skill.linked_skills:
                visit(linked_skill_name)

        for name in requested_skills:
            visit(name)
        return ordered_skills, missing_references

    def _build_inline_skill_message(
        self,
        original_message: str,
        requested_skills: list[str],
        ordered_skills: list[LocalSkill],
        missing_references: list[str],
        execution_context: str | None = None,
    ) -> str:
        requested_display = ", ".join(f"${name}" for name in requested_skills)
        included_display = ", ".join(f"${skill.name}" for skill in ordered_skills) or "(none)"
        sections = [
            "The user explicitly requested local skill guidance from the local skill library.",
            "Apply the following skill instructions as task-specific guidance for this turn.",
            f"Requested skills: {requested_display}",
            f"Included skill packages: {included_display}",
            "Do not echo the skill package verbatim unless the user asks for it.",
            "",
        ]

        if execution_context:
            sections.extend(
                [
                    "## Skill Execution Environment",
                    execution_context,
                    "",
                ]
            )

        sections.extend(
            [
                "## User Request",
                self._strip_skill_tokens(original_message)
                or "No extra task text was provided beyond the skill invocation. Ask for missing inputs if needed.",
                "",
                "## Local Skill Package",
            ]
        )

        for skill in ordered_skills:
            sections.extend(
                [
                    f"### Skill `{skill.name}`",
                    f"Source: {skill.skill_md_path}",
                    f"Description: {skill.description}",
                ]
            )
            if skill.default_prompt:
                sections.extend(
                    [
                        "Default prompt:",
                        skill.default_prompt,
                    ]
                )
            sections.extend(
                [
                    "Instructions:",
                    skill.body,
                ]
            )

            reference_sections = self._load_reference_sections(skill)
            if reference_sections:
                sections.extend(
                    [
                        "References:",
                        *reference_sections,
                    ]
                )

            sections.append("")

        if missing_references:
            missing_display = ", ".join(f"${name}" for name in missing_references)
            sections.extend(
                [
                    "## Missing Referenced Skills",
                    f"The following skill references were mentioned inside local skills but were not found: {missing_display}",
                    "",
                ]
            )

        return "\n".join(sections).strip()

    def _load_reference_sections(self, skill: LocalSkill) -> list[str]:
        if not skill.references:
            return []

        sections: list[str] = []
        total_chars = 0
        remaining_slots = self._REFERENCE_MAX_FILES

        for relative_path in skill.references:
            if remaining_slots <= 0 or total_chars >= self._REFERENCE_MAX_TOTAL_CHARS:
                break
            if relative_path.startswith("... ("):
                sections.append(f"Additional references omitted: {relative_path}")
                break

            reference_path = skill.skill_dir / relative_path
            if not reference_path.exists() or not reference_path.is_file():
                sections.append(f"Reference `{relative_path}` is missing.")
                remaining_slots -= 1
                continue

            try:
                raw_text = reference_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                sections.append(f"Reference `{relative_path}` is not UTF-8 text and was skipped.")
                remaining_slots -= 1
                continue
            except OSError as exc:
                sections.append(f"Reference `{relative_path}` could not be read: {exc}.")
                remaining_slots -= 1
                continue

            remaining_budget = self._REFERENCE_MAX_TOTAL_CHARS - total_chars
            allowed_chars = min(self._REFERENCE_MAX_CHARS_PER_FILE, remaining_budget)
            excerpt = raw_text.strip()
            if len(excerpt) > allowed_chars:
                excerpt = excerpt[: max(0, allowed_chars - 3)].rstrip() + "..."

            if not excerpt:
                sections.append(f"Reference `{relative_path}` is empty.")
            else:
                sections.extend(
                    [
                        f"#### Reference `{relative_path}`",
                        excerpt,
                    ]
                )
                total_chars += len(excerpt)

            remaining_slots -= 1

        omitted_files = len(skill.references) - min(len(skill.references), self._REFERENCE_MAX_FILES)
        if omitted_files > 0:
            sections.append(f"Additional references omitted: {omitted_files} more file(s).")

        return sections

    def _load_skill(self, skill_md_path: Path) -> LocalSkill:
        raw_text = skill_md_path.read_text(encoding="utf-8")
        match = _SKILL_FRONTMATTER_RE.match(raw_text)
        if not match:
            raise ValueError("SKILL.md must start with YAML frontmatter")

        frontmatter_text = match.group("frontmatter")
        body = match.group("body").strip()
        metadata = yaml.safe_load(frontmatter_text) or {}
        name = str(metadata.get("name", "")).strip()
        description = str(metadata.get("description", "")).strip()
        if not name or not description:
            raise ValueError("SKILL.md frontmatter must define name and description")

        skill_dir = skill_md_path.parent
        ui_metadata = self._load_openai_yaml(skill_dir / "agents" / "openai.yaml")
        ui_fields = self._normalize_ui_metadata(ui_metadata)
        return LocalSkill(
            name=name,
            description=description,
            skill_dir=skill_dir,
            skill_md_path=skill_md_path,
            body=body,
            display_name=str(ui_fields.get("display_name", "")).strip(),
            short_description=str(ui_fields.get("short_description", "")).strip(),
            default_prompt=str(ui_fields.get("default_prompt", "")).strip(),
            scripts=self._collect_relative_files(skill_dir / "scripts"),
            references=self._collect_relative_files(skill_dir / "references"),
            assets=self._collect_relative_files(skill_dir / "assets"),
            linked_skills=self.extract_skill_names(body),
        )

    def _load_openai_yaml(self, openai_yaml_path: Path) -> dict[str, object]:
        if not openai_yaml_path.exists():
            return {}
        return YAMLLoader(str(openai_yaml_path)).load()

    def _normalize_ui_metadata(self, raw_metadata: dict[str, object]) -> dict[str, object]:
        interface = raw_metadata.get("interface")
        if isinstance(interface, dict):
            return interface
        return raw_metadata

    def _collect_relative_files(self, directory: Path, limit: int = 100) -> list[str]:
        if not directory.exists() or not directory.is_dir():
            return []

        file_paths = [
            path.relative_to(directory.parent).as_posix() for path in sorted(directory.rglob("*")) if path.is_file()
        ]
        if len(file_paths) <= limit:
            return file_paths

        overflow = len(file_paths) - limit
        return [*file_paths[:limit], f"... (+{overflow} more files)"]

    def _strip_skill_tokens(self, text: str) -> str:
        without_tokens = _INLINE_SKILL_TOKEN_RE.sub("", text)
        without_extra_spaces = re.sub(r"[ \t]{2,}", " ", without_tokens)
        without_extra_blank_lines = re.sub(r"\n{3,}", "\n\n", without_extra_spaces)
        return without_extra_blank_lines.strip()
