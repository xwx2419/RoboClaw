from agent_demo.types.agent_types import AssistantMessageType, TextParam


NO_RESULT_MESSAGE = "No result returned."


def extract_assistant_message_text(message: AssistantMessageType | None) -> str:
    if message is None:
        return ""

    if message.content and message.content.text:
        return message.content.text

    if message.refusal and message.refusal.refusal:
        return message.refusal.refusal

    return ""


def extract_assistant_text_param(message: AssistantMessageType | None) -> TextParam | None:
    message_text = extract_assistant_message_text(message)
    if not message_text:
        return None
    return TextParam(text=message_text)


def resolve_final_response_text(final_text: str | None, streamed_text: str | None) -> str:
    normalized_final_text = (final_text or "").strip()
    normalized_streamed_text = (streamed_text or "").strip()

    if normalized_final_text and normalized_final_text != NO_RESULT_MESSAGE:
        return final_text or normalized_final_text

    if normalized_streamed_text:
        return streamed_text or normalized_streamed_text

    if normalized_final_text:
        return final_text or normalized_final_text

    return NO_RESULT_MESSAGE
