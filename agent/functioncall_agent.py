from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
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
import json
import time
from pydantic import BaseModel,Field
from repo_agent.core.types import (
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
from abc import ABC, abstractmethod
import asyncio
from repo_agent.core.openai_llm import OpenAILLM
from repo_agent.core._base_llm import BaseLLM
from repo_agent.core.agent_types import AgentResponse
from datetime import datetime
from repo_agent.core.agent_types import (
    BaseAgent,
    AgentEvent,
    AgentInputEvent,
    AgentLLMCallEvent,
    AgentResponse,
    AgentThinkEvent,
    AgentToolRequestEvent
)



class FunctionCallAgent(BaseAgent):

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: List[Tool],
        description: str | None = None,
        system_prompt: Optional[str] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 5,
    ) -> None:
        super().__init__(name, llm, system_prompt=system_prompt, description=description)
        self._tool_map = {tool.name: tool for tool in tools} if tools else {}
        self.tools = tools
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations

    def _attach_tool_description_to_system_prompt(self) -> str:
        base_prompt = self.system_prompt or "You are a helpful AI assistant."
        if not self.tools:
            return base_prompt

        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        prompt = base_prompt + "\n\n## Available Tools\n"
        prompt += "When you need external information or to perform actions, you can use the following tools by making function calls:\n"
        prompt += tool_descriptions + "\n"
        prompt += "\nPlease decide whether to call tools and use multiple calls if necessary to obtain a complete answer."
        return prompt

    async def run_stream(self, task: str) -> AsyncGenerator[Union[AgentEvent, AgentResponse], None]:
        """
        Agent's core execution loop, emitting events via async yield.
        """
        start_time = time.time()
        events: List[AgentEvent] = []

        # 1. Yield Input Event
        input_event = AgentInputEvent(
            agent_name=self.name,
            task=task,
            duration_seconds=time.time() - start_time
        )
        yield input_event
        events.append(input_event)

        messages: List[LLMMessage] = [
            SystemMessage(content=self._attach_tool_description_to_system_prompt()),
            UserMessage(content=task)
        ]

        for i in range(self.max_tool_iterations):
            # 2. Call LLM
            llm_start_time = time.time()
            response = await self.llm.chat(messages, tools=self.tools)
            llm_duration = time.time() - llm_start_time

            llm_event = AgentLLMCallEvent(
                agent_name=self.name,
                duration_seconds=llm_duration,
                messages=messages.copy(),
                response=response
            )
            yield llm_event
            events.append(llm_event)
            
            assistant_message = AssistantMessage(content=response.content)
            messages.append(assistant_message)

            if response.finish_reason == 'stop' or isinstance(response.content, str):
                # Final answer received
                think_start_time = time.time()
                final_answer = response.content if isinstance(response.content, str) else ""
                
                think_event = AgentThinkEvent(
                    agent_name=self.name,
                    reason_content=f"Final answer generated: {final_answer[:100]}...",
                    duration_seconds=time.time() - think_start_time
                )
                yield think_event
                events.append(think_event)

                yield AgentResponse(
                    agent_name=self.name,
                    response=final_answer,
                    event_list=events,
                    message_list=messages
                )
                return

            elif response.finish_reason == 'tool_calls' and isinstance(response.content, list):
                # Tool calls requested
                tool_calls: List[ToolCall] = response.content
                
                think_event = AgentThinkEvent(
                    agent_name=self.name,
                    reason_content=f"Planning to call tools: {[tc.tool_name for tc in tool_calls]}"
                )
                yield think_event
                events.append(think_event)

                tool_exec_start_time = time.time()
                tool_tasks = [self._execute_tool(tool_call) for tool_call in tool_calls]
                tool_results: List[ToolExecutionResult] = await asyncio.gather(*tool_tasks)
                tool_exec_duration = time.time() - tool_exec_start_time

                tool_req_event = AgentToolRequestEvent(
                    agent_name=self.name,
                    tool_calls=tool_calls,
                    tool_execution_results=tool_results,
                    duration_seconds=tool_exec_duration
                )
                yield tool_req_event
                events.append(tool_req_event)

                messages.append(ToolMessage(results=tool_results))
                continue

            else:
                # Unexpected response type or finish reason
                break

        # Reached max iterations
        final_response_content = "达到最大轮次的Tool调用数量"
        messages.append(AssistantMessage(content=final_response_content))
        yield AgentResponse(
            agent_name=self.name,
            response=final_response_content,
            event_list=events,
            message_list=messages
        )
        return

    async def run(self, task: str) -> AgentResponse:
        """Runs the agent and returns the final response."""
        final_response = None
        async for message in self.run_stream(task):
            if isinstance(message, AgentResponse):
                final_response = message
        if final_response is None:
            raise RuntimeError("Agent stream finished without producing a final AgentResponse.")
        return final_response

    async def _execute_tool(self, tool_call: ToolCall) -> ToolExecutionResult:
        """
        Executes a single tool call by delegating to the corresponding Tool object.
        """
        tool: Tool | None = self._tool_map.get(tool_call.tool_name)
        if tool is None:
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                name=tool_call.tool_name,
                content=f"Error: Tool '{tool_call.tool_name}' not found.",
                is_error=True
            )
        
        try:
            #等待结果
            result = await tool.execute(tool_call)
            return result
        except Exception as e:
             return ToolExecutionResult(
                tool_call_id=tool_call.id,
                name=tool_call.tool_name,
                content=f"An unexpected error occurred in the agent's tool executor: {e}",
                is_error=True,
            )











