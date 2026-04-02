from pydantic import BaseModel, Field


class InteractionPackage(BaseModel):
    display_widget: str = Field(default="top_top_log")
    content_type: str = Field(default="unknow")
    agent_id: str
    content: object
