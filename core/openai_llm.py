""""
当前华为内源的API问题

- 结构化输出接口不能使用, 一些模型不支持function call 



如果需要强制使用tool call
    设置tool_choice为强制调用tool的模型名称

    - 设置required 则肯定会调用tool 
"""
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
from openai import AsyncStream
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
import logging


logger = logging.getLogger(__name__)

from .types import (

    LLMMessage,
    LLMResponse,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    Tool,
    FinishReasons,
    ToolCall,
    ToolExecutionResult,
    ToolMessage,
    ToolSchema,
    RequestUsage,
    ChatCompletionTokenLogprob,
    FunctionTool
)
from repo_agent.core._base_llm import BaseLLM

def normalize_stop_reason(reason: Optional[str]) -> FinishReasons:
    """标准化停止原因"""
    if reason is None:
        return "unknown"
    reason_lower = reason.lower()
    if "stop" in reason_lower:
        return "stop"
    elif "length" in reason_lower or "max" in reason_lower:
        return "length"
    elif "tool" in reason_lower or "function" in reason_lower:
        return "tool_calls"
    elif "content_filter" in reason_lower:
        return "content_filter"
    else:
        return "unknown"


def parse_reasoning_content(content: str) -> tuple[str | None, str]:
    """解析R1模型的思考内容"""
    if "<think>" in content and "</think>" in content:
        start = content.find("<think>")
        end = content.find("</think>")
        thought = content[start + 7:end].strip()
        actual_content = content[end + 8:].strip()
        return thought, actual_content
    return None, content


def assert_valid_name(name: str) -> None:
    """验证工具名称是否有效"""
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError(
            f"Invalid tool name: '{name}'. "
            "Tool names must only contain a-z, A-Z, 0-9, underscores and dashes."
        )


def convert_to_openai_format_message(
    message: LLMMessage,
) -> Sequence[ChatCompletionMessageParam]:
    """将LLMMessage转换为OpenAI格式"""
    if isinstance(message, SystemMessage):
        return [{"role": "system", "content": message.content}]
    elif isinstance(message, UserMessage):
        return [{"role": "user", "content": message.content}]
    elif isinstance(message, AssistantMessage):
        if isinstance(message.content, str):
            return [{"role": "assistant", "content": message.content}]
        elif isinstance(message.content, list):
           # 转为openai的tool格式 是一个list[Dict]的形式
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": tc.arguments
                    }
                }
                for tc in message.content
            ]
            return [{"role": "assistant", "tool_calls": tool_calls}]
        else:
            return [{"role": "assistant", "content": ""}]
    elif isinstance(message, ToolMessage):
        return [
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content
            }
            for result in message.results
        ]
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def _parse_tool_call_from_content(content: str) -> List[ToolCall]:
    """
    适配华为终端云的tool调用机制, 其tool调用结果是采用字符串形式返回, 而不是function
    Example format:
    <tool_call>
    <function=get_weather>
    <parameter=location>
    北京
    </parameter>
    </function>
    </tool_call>
    """
    if not content or "<tool_call>" not in content:
        return []

    parsed_tool_calls = []

    # Find all tool_call blocks
    tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)

    for i, block in enumerate(tool_call_blocks):
        # Find function name
        function_match = re.search(r"<function=(.*?)>", block, re.DOTALL)
        if not function_match:
            continue

        tool_name = function_match.group(1).strip()

        # Find all parameters within the function block
        param_matches = re.findall(r"<parameter=(.*?)>\s*(.*?)\s*</parameter>", block, re.DOTALL)

        arguments_dict = {key.strip(): value.strip() for key, value in param_matches}

        # Generate a unique ID for the tool call
        tool_call_id = f"tool_{tool_name}_{int(time.time())}_{i}"

        parsed_tool_calls.append(
            ToolCall(
                id=tool_call_id,
                tool_name=tool_name,
                arguments=json.dumps(arguments_dict, ensure_ascii=False)
            )
        )

    return parsed_tool_calls



def convert_to_openai_format_tools(
    tools: Sequence[Union[Tool, ToolSchema]],
) -> List[ChatCompletionToolParam]:
    """将工具转换为OpenAI格式"""
    result: List[ChatCompletionToolParam] = []
    for tool in tools:
        if isinstance(tool, Tool):
            tool_schema = tool.schema
        else:
            tool_schema = tool
        
        # 构建parameters
        if tool_schema.parameters is not None:
            parameters_dict = {
                "type": tool_schema.parameters.type,
                "properties": tool_schema.parameters.properties,
            }
            if tool_schema.parameters.required is not None:
                parameters_dict["required"] = list(tool_schema.parameters.required)
            if tool_schema.parameters.additionalProperties is not None:
                parameters_dict["additionalProperties"] = tool_schema.parameters.additionalProperties
            parameters = cast(FunctionParameters, parameters_dict)
        else:
            parameters = cast(FunctionParameters, {})
        
        result.append(
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool_schema.name,
                    description=(tool_schema.description if tool_schema.description is not None else ""),
                    parameters=parameters,
                    strict=(tool_schema.strict if hasattr(tool_schema, 'strict') else False),
                ),
            )
        )
    
    # 验证所有工具名称
    for tool_param in result:
        assert_valid_name(tool_param["function"]["name"])
    
    return result



class OpenAICreateParams(BaseModel):
    """创建请求的参数"""
    messages: List[ChatCompletionMessageParam]
    tools: List[ChatCompletionToolParam]
    response_format: Optional[Type[BaseModel]]
    create_args: Dict[str, Any]

class OpenAILLM(BaseLLM):
    """OpenAI兼容的LLM客户端"""
    
    TOOL_CHOICE_MODES = ["auto", "none", "any", "required"]
    # 更多参数 ref  https://linux.do/t/topic/813639
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        stream_mode: bool = True,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        temperature=0.1, 
        **kwargs: Any,
    ):
        """初始化OpenAI客户端
        
        Args:
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL
            **kwargs: 其他客户端参数
        """
        self.model = model
        self.stream_mode = stream_mode
        
        self.client = AsyncClient(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        
    
    def _build_create_args(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Union[Tool, ToolSchema]],
        tool_choice: Optional[Union[Literal["auto", "none", "required"], str]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """构建OpenAI API调用参数
        
        参照 agentscope 的流程，但保持原有的数据结构
        
        Args:
            messages: 消息列表
            tools: 工具列表
            tool_choice: 工具选择策略
            structured_output: 结构化输出的Pydantic模型
            **kwargs: 其他OpenAI API参数（temperature, max_tokens等）
        
        Returns:
            包含所有API调用参数的字典
        """
        # 检查消息格式
        if not isinstance(messages, (list, tuple)):
            raise ValueError(
                f"OpenAI `messages` field expected type `list`, got `{type(messages)}` instead."
            )
        
        # 1. 转换消息为OpenAI格式
        oai_messages = []
        for msg in messages:
            oai_messages.extend(convert_to_openai_format_message(msg))
        
        # 2. 构建基础参数
        create_args = {
            "model": self.model,
            "messages": oai_messages,
        }
        
        # 3. 添加用户提供的额外参数（temperature, max_tokens等）
        create_args.update(kwargs)
        
        # 4. 处理 structured_output（结构化输出）
        if structured_output is not None and issubclass(structured_output, BaseModel):
            if tools or tool_choice:
                warnings.warn(
                    "当使用结构化输出的时候,tool 参数将不生效",
                    UserWarning,
                    stacklevel=2,
                )
            # 使用beta client的parse API
            create_args["response_format"] = structured_output
            create_args.pop("tools", None)
            create_args.pop("tool_choice", None)
            return create_args
        # 
        # 5. 处理工具（如果没有structured_output）
        if tools:
            converted_tools = convert_to_openai_format_tools(tools)
            create_args["tools"] = converted_tools
            
            # 6. 处理tool_choice
            if tool_choice:
                self._validate_tool_choice(tool_choice, tools)
                create_args["tool_choice"] = self._format_tool_choice(
                    tool_choice,
                    tools,
                )
        
        return create_args
    
    def _validate_tool_choice(
        self,
        tool_choice: Union[Literal["auto", "none", "required"], str],
        tools: Sequence[Union[Tool, ToolSchema]],
    ) -> None:
        """验证tool_choice参数的有效性"""
        if tool_choice in  OpenAILLM.TOOL_CHOICE_MODES:
            return
        
        # 如果是具体的工具名称，检查是否存在
        tool_names = []
        for tool in tools:
            if isinstance(tool, Tool):
                tool_names.append(tool.schema.name)
            else:
                tool_names.append(tool.name)
        
        if tool_choice not in tool_names:
            raise ValueError(
                f"tool_choice '{tool_choice}' 不合法: {tool_names}, 应该是以下其中之一{tool_names + OpenAILLM.TOOL_CHOICE_MODES}"
            )
    
    def _format_tool_choice(
        self,
        tool_choice: Union[Literal["auto", "none", "any", "required"], str],
        tools: Sequence[Union[Tool, ToolSchema]],
    ) -> Union[str, Dict[str, Any]]:
        """格式化tool_choice参数为OpenAI API格式"""
        if tool_choice in OpenAILLM.TOOL_CHOICE_MODES:
            return tool_choice
        
        # 具体的工具名称
        return {
            "type": "function",
            "function": {"name": tool_choice}
        }
        
    async def __call__(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Optional[Union[Literal["auto", "none", "any", "required"], str]] = 'auto',
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncGenerator[LLMResponse, None]:
        """统一的调用接口，根据 stream_mode 决定是否使用流式输出
        
        Args:
            messages: 消息列表
            tools: 工具列表
            tool_choice: 工具选择策略
            structured_output: 结构化输出的Pydantic模型
            **kwargs: OpenAI API参数（temperature, max_tokens等）
        
        Returns:
            如果 stream_mode=False，返回 LLMResponse
            如果 stream_mode=True，返回 AsyncGenerator[LLMResponse, None]
        """
        if self.stream_mode:
            # 返回流式生成器
            return self.stream_chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_output=structured_output,
                **kwargs,
            )
        else:
            # 返回单次响应
            return await self.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_output=structured_output,
                **kwargs,
            )

    async def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Optional[Union[Literal["auto", "none", "any", "required"], str] ] | None = None,
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """非流式聊天接口
        
        Args:
            messages: 消息列表
            tools: 工具列表
            tool_choice: 工具选择策略
            structured_output: 结构化输出的Pydantic模型
            **kwargs: OpenAI API参数（temperature, max_tokens等）
        
        Returns:
            LLMResponse对象
        """
        
        # 构建API参数
        create_args = self._build_create_args(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            structured_output=structured_output,
            stream = False,
            **kwargs,
        )
        
        if structured_output is not None:
            create_args.pop('stream')
            response = await self.client.chat.completions.parse(  
                **create_args
            )
        else:
            response = await self.client.chat.completions.create(**create_args)
        if structured_output is not None:

        
            response = cast(ParsedChatCompletion[Any], response)
        
        logger.info(f'发送 给{self.model}的 参数为{create_args}')
        logger.info(f'返回的类型为{response.model_dump_json(indent= 4)}')

        
        parsed_response : LLMResponse = self._parse_openai_completion_response(
            response=response,
            structured_output=structured_output,
        )
        
        return parsed_response
    
    def _parse_openai_completion_response(
        self,
        response: ChatCompletion,
        structured_output: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        """解析OpenAI非流式响应
        
        参照 agentscope 的解析逻辑
        ChatCompletion对象
            - choices[Choice(
                -finish_reason=
                - index= 
                - message=ChatCompletionMessage(
                    - content 
                    - function_call
                    - object='chat.completion'
                    - total_tokens=337
                )
            )]
            - 
        
        """
        # 1. 提取usage
                
        if not response.choices:
            # choice为空 出错
            return LLMResponse(
                finish_reason="unknown",
                content="",
                usage=None,
                reason_content=None,
            )
        usage = RequestUsage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
        )


        
        choice = response.choices[0]
        thought: Optional[str] = None
        
        
        custom_tool_calls = []
        # 判断当前文本是否是tool call的形式
        if choice.message.content:
            custom_tool_calls = _parse_tool_call_from_content(choice.message.content)
        # 标准的openai格式会在functioncall 或tool_call字段直接返回tool调用的对象 , 如果模型不支持function call 会有
        if choice.message.tool_calls:
            if choice.finish_reason != "tool_calls":
                warnings.warn(f"模型的: {choice.finish_reason} != tool_calls", stacklevel=2)
            
            # 当有toolcall且content不为空, 则当做思考过程
            if choice.message.content:
                thought = choice.message.content
            
            content = [
                ToolCall(
                    id=tc.id,
                    arguments=tc.function.arguments,
                    tool_name=tc.function.name,
                )
                for tc in choice.message.tool_calls
            ]
            finish_reason = "tool_calls"

        # 2. 
        elif custom_tool_calls:

            thought = choice.message.content
            content = custom_tool_calls
            finish_reason = "tool_calls" 

        # 3. Handle regular text response (if no tool calls of any kind)
        else:
            finish_reason = choice.finish_reason
            content = choice.message.content or ""

      
        if structured_output:


            logger.info(f'结构化的类型{ type(choice.message.parsed)}')
            structured_answer = choice.message.parsed
            
            logger.info(f'model_dump后类型为{ structured_answer}')
        else:
            structured_answer = None
        
        response_obj = LLMResponse(
            finish_reason=normalize_stop_reason(finish_reason),
            content=content,
            usage=usage,
            reason_content=thought,
           structued_content= structured_answer,

        )
      
        return response_obj


    async def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Optional[Union[Literal["auto", "none", "required"], str]] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """流式聊天接口
        
        Args:
            messages: 消息列表
            tools: 工具列表
            tool_choice: 工具选择策略
            structured_output: 结构化输出的Pydantic模型
            **kwargs: OpenAI API参数
        
        Yields:
            LLMResponse对象（每个chunk都返回一个完整的LLMResponse）
        """
        
        # 构建API参数
        create_args = self._build_create_args(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            structured_output=structured_output,
            stream=True,
            **kwargs,
        )
        
        # 流式输出需要添加参数
        create_args["stream_options"] = {"include_usage": True}
        
        logger.info(f'发送给 {self.model} 的流式参数为 {create_args}')
        
        # 调用API并获取流
        if structured_output is not None:
            # 使用 beta client 的 stream 方法
            stream = self.client.beta.chat.completions.stream(**create_args)
        else:
            # 使用标准 client 创建流
            stream = await self.client.chat.completions.create(**create_args)
        
        # 解析流式响应
        async for item in self._parse_openai_stream_response(
            stream=stream,
            structured_output=structured_output,
        ):
            yield item


    async def _parse_openai_stream_response(
        self,
        stream: AsyncStream,
        structured_output: Optional[Type[BaseModel]] = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """解析OpenAI流式响应
        
        参照 agentscope 的流式解析逻辑，但适配你的 LLMResponse 结构
        
        Args:
            stream: OpenAI AsyncStream 对象
            structured_output: 结构化输出的Pydantic模型
        
        Yields:
            LLMResponse对象（包含累积的内容）
        """
        # 累积变量
        usage: Optional[RequestUsage] = None
        text = ""
        thought = ""  # reasoning_content
        tool_calls: Dict[int, Dict[str, Any]] = {}  # 使用字典累积 tool calls
        stop_reason: Optional[str] = None
        chunk_count = 0
        
        async with stream as async_stream:
            async for chunk in async_stream:
                chunk_count += 1
                
                # 如果是 structured_output，chunk 类型可能不同
                if structured_output:
                    if hasattr(chunk, 'type') and chunk.type != "chunk":
                        continue
                    actual_chunk = chunk.chunk if hasattr(chunk, 'chunk') else chunk
                else:
                    actual_chunk = chunk
                
                # 1. 提取 usage（通常在最后一个 chunk）
                if hasattr(actual_chunk, 'usage') and actual_chunk.usage:
                    usage = RequestUsage(
                        prompt_tokens=actual_chunk.usage.prompt_tokens,
                        completion_tokens=actual_chunk.usage.completion_tokens,
                    )
                
                # 2. 如果没有 choices，跳过（可能是只有 usage 的 chunk）
                if not actual_chunk.choices:
                    continue
                
                choice = actual_chunk.choices[0]
                
                # 3. 提取 finish_reason（如果有）
                if choice.finish_reason:
                    stop_reason = choice.finish_reason
                
                # 4. 累积 reasoning_content（思考内容）- O1/O3等推理模型
                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    thought += choice.delta.reasoning_content
                
                # 5. 累积普通文本内容（标准流式）
                if hasattr(choice.delta, 'content') and choice.delta.content:
                    text += choice.delta.content
                
                # 6. 累积 tool_calls（标准OpenAI格式）
                if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        if tc.index in tool_calls:
                            # 累积 arguments（流式tool call会分多个chunk发送）
                            if tc.function.arguments:
                                tool_calls[tc.index]["arguments"] += tc.function.arguments
                        else:
                            # 新的 tool call
                            tool_calls[tc.index] = {
                                "id": tc.id,
                                "tool_name": tc.function.name,
                                "arguments": tc.function.arguments or "",
                            }
                
                # 7. 检查华为云的自定义tool call格式（XML格式在文本中）
                # 只在没有标准tool_calls且有文本内容时检查
                if text and not tool_calls:
                    custom_tool_calls = _parse_tool_call_from_content(text)
                    if custom_tool_calls:
                        logger.debug(f"从文本中解析到自定义tool calls: {custom_tool_calls}")
                        # 将自定义格式转换为标准格式
                        for i, tc in enumerate(custom_tool_calls):
                            tool_calls[i] = {
                                "id": tc.id,
                                "tool_name": tc.tool_name,
                                "arguments": tc.arguments,
                            }
                        # 如果文本完全是tool call格式，将其作为thinking
                        if "<tool_call>" in text:
                            thought = text
                            text = ""  # 清空文本，因为它是tool call
                
                # 8. 构建当前的内容
                if tool_calls:
                    # 有工具调用：返回 ToolCall 列表
                    content = [
                        ToolCall(
                            id=tc["id"],
                            tool_name=tc["tool_name"],
                            arguments=tc["arguments"],
                        )
                        for tc in tool_calls.values()
                    ]
                else:
                    # 普通文本响应
                    content = text
                
                # 9. 构建并 yield LLMResponse（每个chunk都返回当前累积状态）
                result = LLMResponse(
                    finish_reason=normalize_stop_reason(stop_reason) if stop_reason else "unknown",
                    content=content,
                    usage=usage,
                    reason_content=thought if thought else None,
                )
                
                yield result
        
        # 流结束后的统计日志
        logger.info(f'流式响应统计: 共收到 {chunk_count} 个chunk, '
                    f'累积文本长度: {len(text)}, '
                    f'tool_calls数量: {len(tool_calls)}')
        
        # 最后再yield一次最终状态（确保包含完整的usage和finish_reason）
        if tool_calls:
            final_content = [
                ToolCall(
                    id=tc["id"],
                    tool_name=tc["tool_name"],
                    arguments=tc["arguments"],
                )
                for tc in tool_calls.values()
            ]
        else:
            final_content = text
        
        final_result = LLMResponse(
            finish_reason=normalize_stop_reason(stop_reason) if stop_reason else "unknown",
            content=final_content,
            usage=usage,
            reason_content=thought if thought else None,
        )
        
        logger.info(f'流式响应完成，最终结果: {final_result}')
