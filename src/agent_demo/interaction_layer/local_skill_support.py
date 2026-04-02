from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent_demo.agent_layer.agent_components.agent_tools.local_skill_registry import (
    ExpandedSkillMessage,
    LocalSkillRegistry,
)
from agent_demo.types.agent_types import ServiceRegister


@dataclass(frozen=True, slots=True)
class PreparedAgentMessage:
    message: str | None
    expanded_message: ExpandedSkillMessage
    status_message: str | None
    error_message: str | None


def build_skill_execution_context(services: Sequence[ServiceRegister] | None) -> str:
    sections = [
        "This skill executes inside the main Olympus agent instance and shares that agent's MCP tool access.",
        "Any tool from an active service below can be called directly during the skill.",
        "If a required service is inactive, first call `AgentTools___activate_service`, then call that service's tools.",
        "",
        "Service snapshot:",
    ]
    if not services:
        sections.append(
            "- No live service registry snapshot is available yet. Fall back to the Server Registry Block in system context."
        )
        return "\n".join(sections)

    for service in services:
        status = "active" if service.is_activation else "inactive"
        tools = (
            ", ".join(f"`{service.service_name}___{tool.tool_name}`" for tool in service.tools_list)
            or "(no tools discovered)"
        )
        sections.append(f"- `{service.service_name}` [{status}]: {tools}")
    return "\n".join(sections)


def prepare_agent_message(
    message: str,
    skill_registry: LocalSkillRegistry,
    services: Sequence[ServiceRegister] | None = None,
) -> PreparedAgentMessage:
    expanded_message = skill_registry.expand_inline_request(
        message,
        execution_context=build_skill_execution_context(services),
    )
    if not expanded_message.requested_skills:
        return PreparedAgentMessage(
            message=message,
            expanded_message=expanded_message,
            status_message=None,
            error_message=None,
        )

    if expanded_message.missing_requested_skills:
        missing_display = ", ".join(f"${name}" for name in expanded_message.missing_requested_skills)
        available_skills = skill_registry.list_skills(refresh=False)
        available_display = ", ".join(f"${skill.name}" for skill in available_skills[:8]) or "(none)"
        return PreparedAgentMessage(
            message=None,
            expanded_message=expanded_message,
            status_message="Requested local skill not found.",
            error_message=f"Local skill not found: {missing_display}\nAvailable skills: {available_display}",
        )

    requested_display = ", ".join(f"${name}" for name in expanded_message.requested_skills)
    status_message = f"Attached local skill guidance: {requested_display}"
    if expanded_message.missing_referenced_skills:
        missing_refs = ", ".join(f"${name}" for name in expanded_message.missing_referenced_skills[:3])
        suffix = "..." if len(expanded_message.missing_referenced_skills) > 3 else ""
        status_message += f" | missing referenced: {missing_refs}{suffix}"

    return PreparedAgentMessage(
        message=expanded_message.message,
        expanded_message=expanded_message,
        status_message=status_message,
        error_message=None,
    )
