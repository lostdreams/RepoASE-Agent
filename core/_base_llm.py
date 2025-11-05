

from abc import abstractmethod
from pydantic import BaseModel
from typing import AsyncGenerator, Any, Sequence, Union,Optional,Literal,Type
from repo_agent.core.types import (
    LLMResponse,
    LLMMessage,
    Tool,
    ToolCall,
    ToolSchema

)




class BaseLLM:

    model_name: str
    stream_mode: bool
    

    def __init__(
        self,
        model_name: str,
        stream: bool,
    ) -> None:
    
        self.model_name = model_name
        self.stream = stream 


    
    @abstractmethod
    async def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse | AsyncGenerator[LLMResponse, None]:
        pass
    

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Optional[Union[Literal["auto", "none", "any", "required"], str] ] | None = None,
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Optional[Union[Literal["auto", "none", "required"], str]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        pass
