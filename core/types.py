from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
    Protocol,
    runtime_checkable,
    Type,
)

from typing import Union

JSONPrimitive = Union[
    str,
    int,
    float,
    bool,
    None,
]
from abc import abstractmethod
import asyncio
import inspect
import json
import time
import warnings
import typing
import re
from functools import partial
from abc import abstractmethod

from pydantic import BaseModel, Field, create_model
from dataclasses import dataclass
from typing_extensions import Annotated

from openai import AsyncOpenAI,AsyncClient
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
    ParsedChoice,
    ChatCompletionToolMessageParam
)
from openai.types.chat.chat_completion import Choice
from openai.types.shared_params import (
    FunctionDefinition,
    FunctionParameters,
    ResponseFormatJSONObject,
    ResponseFormatText,
)


from ._base_llm import BaseLLM

import logging
logger = logging.getLogger(__name__)


class SystemMessage(BaseModel):
    """系统消息"""
    role: Literal["system"] = "system"
    content: str
    type: Literal["SystemMessage"] = "SystemMessage"


class UserMessage(BaseModel):
    """用户消息"""
    role: Literal["user"] = "user"
    content: str
    type: Literal["UserMessage"] = "UserMessage"


class ToolCall(BaseModel):
    """工具调用"""
    id: str
    arguments: str | BaseModel  # JSON string
    tool_name: str


class AssistantMessage(BaseModel):
    """助手消息"""
    role : str = 'assistant'
    response: str | None = None
    content: Union[str, List[ToolCall]] | None = None
    reason_content: str | None = None
    type: Literal["AssistantMessage"] = "AssistantMessage"


class ToolExecutionResult(BaseModel):
    """工具执行结果"""
    tool_call_id: str
    name: str
    content: str
    raw_result: Any = None
    is_error: bool = False
    duration_seconds: float = 0.0


class ToolMessage(BaseModel):
    """工具执行结果消息"""
    role: Literal["tool"] = "tool"
    results: List[ToolExecutionResult]
    
    type: Literal["ToolResultMessage"] = "ToolResultMessage"

    

        



LLMMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="type")
]


# ==================== 其他类型定义 ====================

@dataclass
class RequestUsage:
    prompt_tokens: int
    completion_tokens: int


FinishReasons = Literal["stop", "length", "tool_calls", "content_filter", "unknown"]


class TopLogprob(BaseModel):
    """Top logprob信息"""
    logprob: float
    bytes: Optional[List[int]] = None


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List[TopLogprob]] = None
    bytes: Optional[List[int]] = None


class LLMResponse(BaseModel):
    finish_reason: FinishReasons
    content: Union[str, List[ToolCall]]
    usage: Optional[RequestUsage]
    reason_content: Optional[str] = None
    structued_content:  BaseModel | None = None


class ParametersSchema(BaseModel):
    type: str
    properties: Dict[str, Any]
    required: Optional[Sequence[str]] = None
    additionalProperties: Optional[bool] = None


class ToolSchema(BaseModel):
    """工具Schema定义"""
    name: str
    description: Optional[str] = None
    parameters: Optional[ParametersSchema] = None
    strict: bool = True





class Tool:
    """工具基类"""
    
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    async def execute(self, args: Mapping[str, Any] | BaseModel) -> ToolExecutionResult:
        """执行工具调用"""
        pass
class FunctionTool(Tool):
    """基于 Python 函数的工具"""
    
    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        # CHANGED: Default strict to False for better compatibility with non-OpenAI endpoints
        strict: bool = False, 
    ):
        self._func = func
        self._name = name or func.__name__
        self._description = description or inspect.getdoc(func) or ""
        self.strict = strict
        self._args_model = self._build_args_model()
        self._signature = inspect.signature(self._func)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def _build_args_model(self) -> type[BaseModel]:
        """根据函数签名构建 Pydantic 参数模型"""
        sig = inspect.signature(self._func)
        type_hints = typing.get_type_hints(self._func)
        
        fields = {}
        for param in sig.parameters.values():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            ):
                raise TypeError(
                    f"Unsupported parameter kind in function '{self.name}': {param.kind}"
                )
            
            param_type = type_hints.get(param.name, Any)
            
            # Use param.name as the description by default for better model understanding
            field_description = param.name

            if param.default is inspect.Parameter.empty:
                # CHANGED: Added description to the Field
                fields[param.name] = (param_type, Field(..., description=field_description))
            else:
                # CHANGED: Added description to the Field
                fields[param.name] = (param_type, Field(default=param.default, description=field_description))
        
        return create_model(f"{self.name}Args", **fields)
    
    @property
    def schema(self) -> ToolSchema:
        """返回工具 schema"""
        params_schema = self._args_model.model_json_schema()
        
        # 清理schema
        params_schema.pop('$defs', None)
        params_schema.pop('$schema', None)
        params_schema.pop('title', None)
        
        params_schema.setdefault('type', 'object')
        params_schema.setdefault('properties', {})

        # CHANGED: Explicitly set additionalProperties to False for stricter schema definition
        # This aligns with AutoGen's schema and is a best practice.
        params_schema.setdefault('additionalProperties', False)
        
        # 提取required字段
        required = params_schema.pop('required', None)
        additional_properties = params_schema.pop('additionalProperties', False)
        
        parameters = ParametersSchema(
            type=params_schema.get('type', 'object'),
            properties=params_schema.get('properties', {}),
            required=required,
            additionalProperties=additional_properties
        )
        
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
            strict=self.strict,
        )
    
  
    async def execute(self, args: ToolCall) -> ToolExecutionResult:
        """执行工具调用"""
        logger.info(f"执行工具 '{self.name}'，调用 ID: {args.id}, \n 参数{args.arguments}")
        
        start_time = time.time()
        
        try:
            args_dict = json.loads(args.arguments)

            # 验证是否妈祖
            validated_args = self._args_model.model_validate(args_dict)
            
            if asyncio.iscoroutinefunction(self._func):
                raw_result = await self._func(**validated_args.model_dump())
            else:
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    None,
                    partial(self._func, **validated_args.model_dump())
                )
            
            duration = time.time() - start_time
            
            return ToolExecutionResult(
                tool_call_id=args.id, # Use args.id
                name=self.name,
                content=json.dumps(raw_result, ensure_ascii=False),
                raw_result=raw_result,
                is_error=False,
                duration_seconds=duration
            )
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"工具 '{self.name}' 执行失败: {e}")
            
            return ToolExecutionResult(
                tool_call_id=args.id, # Use args.id
                name=self.name,
                content=json.dumps({"error": str(e)}, ensure_ascii=False),
                is_error=True,
                duration_seconds=duration
            )
