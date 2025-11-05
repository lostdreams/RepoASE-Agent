
from typing import Any, AsyncGenerator, Mapping, Sequence, List, Optional

from autogen_core import CancellationToken, ComponentBase
from pydantic import BaseModel, SerializeAsAny, model_validator,Field
from abc import ABC, abstractmethod
from datetime import datetime
from .types import LLMMessage, AssistantMessage ,ToolCall, ToolExecutionResult, LLMResponse,
from repo_agent.core._base_llm import BaseLLM


class AgentEvent(BaseModel):
    agent_name: str
    create_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

class AgentToolRequestEvent(AgentEvent):
    tool_calls: List[ToolCall]
    tool_execution_results: List[ToolExecutionResult]

class AgentThinkEvent(AgentEvent):
    reason_content: str

class AgentLLMCallEvent(AgentEvent):
    messages: List[LLMMessage]
    response: LLMResponse

class AgentInputEvent(AgentEvent):
    task: str


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM ,

        description : str | None  = None,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.llm = llm
        self.descrition = description
        self.system_prompt = system_prompt
        self._history: list[LLMMessage] = []
    
    @abstractmethod
    def run(self, task: str, **kwargs) -> AgentResponse:
        """运行Agent"""
        pass


class AgentResponse(BaseModel):
    agent_name: str
    response: str | None = None
    event_list: Sequence[AgentEvent]
    message_list: List[LLMMessage]

    @model_validator(mode='after')
    def check_response(self) -> "AgentResponse":
        if self.response is None and self.message_list:
            last_message = self.message_list[-1]
            if isinstance(last_message, AssistantMessage) and isinstance(last_message.content, str):
                self.response = last_message.content
            else:
                self.response = "" 
        return self




class AgentContext:

    pass
