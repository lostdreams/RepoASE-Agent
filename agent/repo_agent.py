"""
RepoAgent 修正版实现

核心修改：
1. 明确分离四步流程，第二步和第三步完全独立
2. 第一步只判断是否能回答，不制定 action plan
3. 第二步只能用执行工具（仓库+额外），不包括 update_context
4. 第三步只能用 update_context 工具，必须调用
5. 移除第四步（决策已合并到第一步）
6. 添加 execution_timeout 超时机制
7. 添加 enable_compress_max_length 上下文压缩机制
"""

import json
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Sequence
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime

from repo_agent.core.agent_types import (
    AgentResponse,
    BaseAgent,
    AgentEvent,
    AgentInputEvent,
    AgentThinkEvent,
    AgentLLMCallEvent,
    AgentToolRequestEvent
)
from repo_agent.core.types import (
    LLMMessage,
    Tool,
    FunctionTool,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
    ToolExecutionResult
)
from repo_agent.core._base_llm import BaseLLM
from repo_agent.tool.repo_sandbox import RepoSandbox

logger = logging.getLogger(__name__)

DEFAULT_REPO_MEMORY_SAVE_DIR = '/usr1/Rtos_Coding/data/repo_cache/repo_memory'



class RepoMemory(BaseModel):
    """仓库长期记忆机制"""
    
    repo_name: str
    context: str = ""
    subrepo_context: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}
    _save_path: Optional[Path] = None
    
    @classmethod
    def load(cls, repo_name: str, save_dir: Optional[Union[str, Path]] = None) -> "RepoMemory":
        if save_dir is None:
            save_dir = Path(DEFAULT_REPO_MEMORY_SAVE_DIR)
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{repo_name}_memory.json"
        
        if save_path.exists():
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                memory = cls(**data)
                memory._save_path = save_path
                logger.info(f"从 {save_path} 加载记忆成功")
                return memory
            except Exception as e:
                logger.warning(f"加载记忆失败: {e}，创建新记忆")
        
        logger.info(f"未找到 {save_path}，创建新的空记忆")
        memory = cls(repo_name=repo_name)
        memory._save_path = save_path
        memory.save_to_disk()
        logger.info(f"已创建空记忆文件: {save_path}")
        return memory
    
    def save_to_disk(self) -> None:
        if self._save_path is None:
            save_dir = Path.home() / '.repo_agent_memory'
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_path = save_dir / f"{self.repo_name}_memory.json"
        
        try:
            data = self.model_dump(exclude={'_save_path'})
            with open(self._save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"记忆已保存到 {self._save_path}")
        except Exception as e:
            logger.error(f"保存记忆失败: {e}")
    
    def update_repo_context(self, new_context: str) -> None:
        self.context = new_context
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def update_subrepo_context(self, subrepo_path: str, new_context: str) -> None:
        self.subrepo_context[subrepo_path] = new_context
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def get_memory_summary(self) -> str:
        summary_parts = []
        if self.context:
            summary_parts.append(f"## {self.repo_name} 仓库概览\n{self.context}")
        if self.subrepo_context:
            summary_parts.append("\n## 模块/目录说明")
            for path, desc in self.subrepo_context.items():
                summary_parts.append(f"- `{path}`: {desc}")
        return "\n\n".join(summary_parts) if summary_parts else ""



class AgentContext(BaseModel):
    """Agent 上下文管理器"""
    
    context: str = ""
    history_messages: List[LLMMessage] = Field(default_factory=list)
    _history_messages_num: int = 5
    
    model_config = {"arbitrary_types_allowed": True}
    
    def insert_to_agent(self, base_system_prompt: str) -> List[LLMMessage]:
        messages = []
        
        system_content_parts = [base_system_prompt]
        if self.context:
            system_content_parts.append("\n\n## 当前构建的上下文\n")
            system_content_parts.append(self.context)
        
        system_message = SystemMessage(content="\n".join(system_content_parts))
        messages.append(system_message)
        
        recent_messages = self.history_messages[-self._history_messages_num:] if len(
            self.history_messages) > self._history_messages_num else self.history_messages
        messages.extend(recent_messages)
        
        return messages
    
    def update_context(self, new_content: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = f"\n\n--- 更新于 {timestamp} ---\n"
        
        if self.context:
            self.context += separator + new_content
        else:
            self.context = new_content
    
    def add_message(self, message: LLMMessage) -> None:
        self.history_messages.append(message)
    
    def get_context_length(self) -> int:
        return len(self.context)
    
    def get_history_length(self) -> int:
        return len(self.history_messages)



class PlanningDecision(BaseModel):
    """"""
    thinking : str = Field(description="思考当前上下文是否可以支持完成回答问题, 仔细思考, 确认当前上下文是否足够充足, 思考足够全面")

    can_answer: bool = Field(description="当前上下文是否足够回答问题？true 表示可以，false 表示不够")

    next_exploration: Optional[str] = Field(
        description="如果 can_answer=false 说明下一步要探索什么内容, 为True, 则可以为空 ",
        default=None
    )
    


class ContextUpdateSummary(BaseModel):
    """上下文更新汇总"""


    summary: str = Field(description="对本轮执行结果的汇总，包括：发现了什么、哪些成功、哪些失败、得到了什么结论, 即使当前工具执行失败,也要记录下失败的调用方式")

    key_findings: List[str] = Field(description=" 关键发现, 重点是在哪个代码仓的文件下(完整路径) 存在和当前任务相关的内容, 或者和全局任务相关的内容 ")




class ContextRepoAgent(BaseAgent):
    """
    
    修正后的流程：
    1. Planning: 判断 context 是否足够 → 是则回答，否则说明要探索什么
    2. Execution: 只用执行工具（仓库+额外）收集信息
    3. Context Update: 只用 update_context 工具汇总并更新上下文（必须执行）
    4. 循环回到 Step 1
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        repo_sandbox: RepoSandbox,  

        structed_output : Optional[BaseModel] = None ,

        context: Optional[AgentContext] = None,
        ext_tools: Optional[List[Tool]] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,

        
        using_repo_memory: bool = True,
        tool_names: Optional[List[str]] = None,
        max_rounds: int = 50,
        max_tool_each_round: int = 30,
        history_window_size: int = 5,
        execution_timeout: float = 300.0,  # 等待模型回复的时间
        enable_compress_max_length: int = 0, # 默认不开启上下文的压缩
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            description=description or "代码仓库智能体",
            system_prompt=system_prompt
        )
        
        self.repo_sandbox = repo_sandbox
        self.using_repo_memory = using_repo_memory
        self.max_rounds = max_rounds
        self.max_tool_each_round = max_tool_each_round
        self.execution_timeout = execution_timeout
        self.enable_compress_max_length = enable_compress_max_length
        self.repo_memories: Dict[str, RepoMemory] = {}

        if using_repo_memory:
            for repo_name in repo_sandbox.repo_names_list:
                memory = RepoMemory.load(repo_name)
                self.repo_memories[repo_name] = memory
        
        if context is None:
            self.agent_context = AgentContext()
            self.agent_context._history_messages_num = history_window_size
        else:
            self.agent_context = context
        
        self.repo_tools = repo_sandbox.get_sandbox_tools(tool_names)
        self.ext_tools = ext_tools or []


        self.execution_tools = self.repo_tools + self.ext_tools
        
        self.context_tools = [self._create_update_context_tool()]
        
        self.base_system_prompt = self._build_base_system_prompt()
        
        self._current_round = 0
        
        logger.info(f"ContextRepoAgent '{name}' 初始化完成")
        logger.info(f"  - 执行工具: {len(self.execution_tools)} 个")
        logger.info(f"  - 上下文工具: {len(self.context_tools)} 个")
        logger.info(f"  - 执行超时: {execution_timeout} 秒")
        logger.info(f"  - 压缩阈值: {enable_compress_max_length} 字符 ({'启用' if enable_compress_max_length > 0 else '禁用'})")
    
    def _create_update_context_tool(self) -> FunctionTool:
        """创建更新上下文的工具"""
        
        def update_context(summary: str, key_findings: List[str]) -> Dict[str, Any]:
            """
            将本轮的执行结果汇总并添加到上下文
            
            Args:
                summary: 对本轮执行结果的整体汇总
                key_findings: 关键发现列表
            
            Returns:
                更新结果
            """
            logger.info(f"[update_context] 汇总: {summary[:100]}...")
            logger.info(f"[update_context] 关键发现: {len(key_findings)} 条")
            
            update_content_parts = [
                f"**本轮汇总**: {summary}",
                "",
                "**关键发现**:"
            ]
            for i, finding in enumerate(key_findings, 1):
                update_content_parts.append(f"{i}. {finding}")
            
            update_content = "\n".join(update_content_parts)
            
            self.agent_context.update_context(update_content)
            
            return {
                "status": "success",
                "message": f"成功更新上下文（汇总 {len(summary)} 字符，{len(key_findings)} 条发现）",
                "current_context_length": len(self.agent_context.context)
            }
        
        return FunctionTool(
            func=update_context,
            name="update_context",
            description="汇总本轮执行结果并更新上下文。必须调用此工具来保存本轮的发现。"
        )
    
    def _build_base_system_prompt(self) -> str:
        """构建基础系统提示词"""
        parts = []
        
        if self.system_prompt:
            parts.append(self.system_prompt)
        else:
            parts.append("""你是一个专业的代码仓库分析助手，同时也是一个 **Context Engineer**。

## 你的核心任务
持续构建和维护上下文，直到能够回答用户问题。

## 工作流程（三步循环）

### Step 1: Planning（规划）
**目标**: 判断当前 context 是否足够回答问题

如果**足够** → 直接给出答案，结束任务
如果**不够** → 说明下一步要探索什么内容

注意：
- 只需要说"要探索什么"，不需要制定具体的 action plan
- 例如："需要了解 XXX 模块的实现细节"，而不是"调用 read_file 读取 xxx.py"

### Step 2: Execution（执行）
**目标**: 使用工具收集信息

可用工具：
- 仓库探索工具（list_repo_dir, search_files, search_code, read_file 等）
- 额外工具（如果有）

注意：
- 这一步**不能**调用 `update_context`
- 只负责收集信息，不负责汇总
- 执行结果可能成功、失败、或部分成功

### Step 3: Context Update（上下文更新）
**目标**: 汇总本轮结果并更新上下文

可用工具：
- **只有** `update_context`

注意：
- **必须**调用 `update_context`，无论 Step 2 是否成功
- 即使工具调用失败，失败信息本身也有价值
- 需要提炼汇总，不要原封不动复制大量代码
- 要说明：发现了什么、哪些成功、哪些失败

### 循环
完成 Step 3 后，回到 Step 1 继续下一轮。

## 关键原则

1. **上下文是累积的知识库**
   - 会跨轮次保留
   - 是回答问题的最终依据
   - 每轮只能看到最近 5 条消息，但能看到完整上下文

2. **三步分离**
   - Step 1 只判断，不执行
   - Step 2 只执行，不汇总
   - Step 3 只汇总，不执行

3. **错误也有价值**
   - 工具调用失败了？记录下来
   - 文件不存在？说明架构可能不同
   - 这些信息都应该进入上下文

4. **不要重复工作**
   - 检查上下文，避免重复收集已有信息""")
        
        repo_info = self.repo_sandbox.get_repo_info()
        parts.append(f"\n## 可用仓库")
        parts.append(f"你可以访问 {repo_info['total_repos']} 个代码仓库：")
        for repo in repo_info['repos']:
            parts.append(f"- **{repo['name']}**")
            if repo['description']:
                parts.append(f"  描述: {repo['description']}")
        
        if self.using_repo_memory:
            memory_parts = []
            for repo_name, memory in self.repo_memories.items():
                memory_summary = memory.get_memory_summary()
                if memory_summary:
                    memory_parts.append(memory_summary)
            
            if memory_parts:
                parts.append("\n## 仓库长期知识")
                parts.extend(memory_parts)
        
        return "\n".join(parts)
    
    async def run_stream(
        self,
        task: str
    ) -> AsyncGenerator[Union[AgentEvent, AgentResponse], None]:
        """
        流式运行 Agent（修正版）
        
        三步循环：
        1. Planning: 判断是否能回答 → 能则结束，不能则说明要探索什么
        2. Execution: 使用执行工具收集信息
        3. Context Update: 使用 update_context 工具汇总结果
        """
        start_time = time.time()
        events: List[AgentEvent] = []
        
        input_event = AgentInputEvent(
            agent_name=self.name,
            task=task,
            duration_seconds=0
        )
        yield input_event
        events.append(input_event)
        
        self.agent_context.add_message(UserMessage(content=task))
        
        for round_num in range(1, self.max_rounds + 1):
            self._current_round = round_num
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_num}/{self.max_rounds}")
            logger.info(f"{'='*60}")
            
            try:
                planning = None
                async for event in self._planning_step(task, events):
                    if isinstance(event, PlanningDecision):
                        planning = event
                    else:
                        yield event
                        events.append(event)
                
                if planning is None:
                    raise RuntimeError("Planning 阶段未产生 PlanningDecision")
                
                if planning.can_answer:
                    logger.info("✅ Agent 认为可以回答问题，生成最终答案")
                    
                    final_answer = None
                    async for event in self.generate_final_answer(task, events):
                        if isinstance(event, str):
                            final_answer = event
                        else:
                            yield event
                            events.append(event)
                    
                    if final_answer is None:
                        final_answer = "生成最终答案失败"
                    
                    final_response = AgentResponse(
                        agent_name=self.name,
                        response=final_answer,
                        event_list=events,
                        message_list=[]
                    )
                    yield final_response
                    return
                
                logger.info(f"➡️  需要继续探索: {planning.next_exploration}")
                
                exec_summary = None
                async for event in self._execution_step(planning, events):
                    if isinstance(event, str):
                        exec_summary = event
                    else:
                        yield event
                        events.append(event)
                
                if exec_summary is None:
                    exec_summary = "本轮执行未产生结果"
                
                context_updated = False
                async for event in self._context_update_step(exec_summary, events):
                    if isinstance(event, bool):
                        context_updated = event
                    else:
                        yield event
                        events.append(event)
                
                if not context_updated:
                    think_event = AgentThinkEvent(
                        agent_name=self.name,
                        reason_content="⚠️  Context Update 失败，未能更新上下文",
                        duration_seconds=0
                    )
                    yield think_event
                    events.append(think_event)
                else:
                    async for event in self._compress_context_if_needed(events):
                        yield event
                        events.append(event)
                
                continue
            
            except Exception as e:
                logger.error(f"执行出错: {e}", exc_info=True)
                error_msg = f"执行过程中发生错误: {str(e)}"
                
                final_response = AgentResponse(
                    agent_name=self.name,
                    response=error_msg,
                    event_list=events,
                    message_list=[]
                )
                yield final_response
                return
        
        warning = f"已达到最大轮次 ({self.max_rounds})，尝试基于当前上下文回答。"
        logger.warning(warning)
        
        force_answer_prompt = f"""你已经完成了 {self.max_rounds} 轮上下文构建。
现在请基于当前的上下文，给出尽可能完整的答案。

原始任务：{task}

如果信息不足，请说明还缺少哪些信息。"""
        
        messages = self.agent_context.insert_to_agent(self.base_system_prompt)
        messages.append(UserMessage(content=force_answer_prompt))
        
        response = await self.llm.chat(messages=messages)
        final_answer = f"{warning}\n\n{response.content}"
        
        final_response = AgentResponse(
            agent_name=self.name,
            response=final_answer,
            event_list=events,
            message_list=[]
        )
        yield final_response
    
    async def _planning_step(
        self,
        original_task: str,
        events: List[AgentEvent]
    ) -> AsyncGenerator[Union[AgentThinkEvent, PlanningDecision], None]:
        """
        Planning 步骤（修正版）
        
        只判断：当前 context 是否足够回答？
        - 是 → can_answer=true（后续会调用 generate_final_answer）
        - 否 → 说明要探索什么
        """
        logger.info("=== Step 1: Planning ===")
        start_time = time.time()
        
        planning_prompt = f"""## Step 1: Planning

**原始任务**: {original_task}

**当前轮次**: {self._current_round}/{self.max_rounds}

**你的任务**: 判断当前构建的上下文是否足够回答用户问题。

查看"当前构建的上下文"（在系统消息中），然后判断：

1. **如果上下文足够** → 设置 can_answer=true（后续会生成最终答案）
2. **如果上下文不够** → 设置 can_answer=false，说明 next_exploration

注意：
- 第一轮时上下文可能是空的，这是正常的
- next_exploration 只需说"要探索什么"，不要制定具体 action plan
- 例如："需要了解 driver 模块的初始化流程"
- 不要说："调用 search_files 查找 driver 目录"（太具体了）"""
        
        messages = self.agent_context.insert_to_agent(self.base_system_prompt)
        messages.append(UserMessage(content=planning_prompt))
        
        response = await self.llm.chat(
            messages=messages,
            structured_output=PlanningDecision
        )
        
        planning = response.structued_content
        assert isinstance(planning, PlanningDecision)
        
        duration = time.time() - start_time
        
        if planning.can_answer:
            reason = f"[Planning] ✅ 可以回答: {planning.thinking[:100]}"
        else:

            if planning.next_exploration is not None:
                reason = f"[Planning] ➡️  需要探索: {planning.next_exploration[:100]}"

            else:
                reason = '模型输出为空'
        
        think_event = AgentThinkEvent(
            agent_name=self.name,
            reason_content= reason,
            duration_seconds=duration
        )
        yield think_event
        
        yield planning
    
    async def generate_final_answer(
        self,
        original_task: str,
        events: List[AgentEvent]
    ) -> AsyncGenerator[Union[AgentEvent, str], None]:
        """
        生成最终答案
        
        当 Planning 判断可以回答问题时调用此方法
        """
        logger.info("=== 生成最终答案 ===")
        start_time = time.time()
        
        final_prompt = f"""## 生成最终答案

**原始任务**: {original_task}

根据你构建的上下文（见系统消息中的"当前构建的上下文"部分），现在请给出完整、准确的答案。

要求：
1. 基于上下文中的信息回答
2. 回答应该完整、结构清晰
3. 如果有代码示例，请给出
4. 如果有多个相关点，请逐一说明

现在请给出你的最终答案："""
        
        messages = self.agent_context.insert_to_agent(self.base_system_prompt)
        messages.append(UserMessage(content=final_prompt))
        
        response = await self.llm.chat(messages=messages)
        
        duration = time.time() - start_time
        
        llm_event = AgentLLMCallEvent(
            agent_name=self.name,
            duration_seconds=duration,
            messages=messages.copy(),
            response=response
        )
        yield llm_event
        
        think_event = AgentThinkEvent(
            agent_name=self.name,
            reason_content="[最终答案] ✅ 已生成基于上下文的完整答案",
            duration_seconds=0
        )
        yield think_event
        
        yield response.content if isinstance(response.content, str) else str(response.content)


    async def _execution_step(
        self,
        planning: PlanningDecision,
        events: List[AgentEvent]
    ) -> AsyncGenerator[Union[AgentEvent, str], None]:
        """
        Execution 步骤（修正版）
        
        只使用执行工具（仓库+额外），不包括 update_context
        """
        logger.info("=== Step 2: Execution ===")
        start_time = time.time()
        
        execution_prompt = f"""## Step 2: Execution

**当前轮次目标**: {planning.next_exploration}

**你的任务**: 使用工具收集信息

可用的工具：
- 仓库探索工具（list_repo_dir, search_files_in_repo, search_code_in_repo, read_repo_file 等）
- 额外工具（如果有）

注意：

- 只负责收集信息
- 可以多次调用工具
- 最终你的输出是

现在开始探索！"""
        
        messages = self.agent_context.insert_to_agent(self.base_system_prompt)
        messages.append(UserMessage(content=execution_prompt))
        
        tool_call_count = 0
        execution_results = []
        timeout_reached = False
        
        for iteration in range(self.max_tool_each_round):
            try:
                response = await asyncio.wait_for(
                    self.llm.chat(
                        messages=messages,
                        tools=self.execution_tools
                    ),
                    timeout=self.execution_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"执行超时 ({self.execution_timeout}秒)")
                timeout_reached = True
                break
            
            llm_event = AgentLLMCallEvent(
                agent_name=self.name,
                duration_seconds=0,
                messages=messages.copy(),
                response=response
            )
            yield llm_event
            
            assistant_msg = AssistantMessage(
                content=response.content if isinstance(response.content, str) else response.content
            )
            messages.append(assistant_msg)
            self.agent_context.add_message(assistant_msg)
            
            if  isinstance(response.content, str):
                duration = time.time() - start_time
                
                summary = f"完成执行，共 {tool_call_count} 次工具调用"
                if isinstance(response.content, str):
                    summary += f"\n模型回复: {response.content[:200]}"
                
                think_event = AgentThinkEvent(
                    agent_name=self.name,
                    reason_content=f"[Execution] {summary}",
                    duration_seconds=duration
                )
                yield think_event
                
                exec_summary = self._generate_execution_summary(
                    execution_results,
                    response.content if isinstance(response.content, str) else ""
                )
                yield exec_summary
                return
            
            elif  isinstance(response.content, list):
                tool_calls: List[ToolCall] = response.content
                tool_call_count += len(tool_calls)
                
                tool_start = time.time()
                tool_results = []
                for tc in tool_calls:
                    result = await self._execute_tool(tc)
                    tool_results.append(result)
                    execution_results.append({
                        "tool": tc.tool_name,
                        "success": not result.is_error,
                        "content": result.content[:200] + "..." if len(result.content) > 200 else result.content
                    })
                
                tool_duration = time.time() - tool_start
                
                tool_event = AgentToolRequestEvent(
                    agent_name=self.name,
                    tool_calls=tool_calls,
                    tool_execution_results=tool_results,
                    duration_seconds=tool_duration
                )
                yield tool_event
                
                tool_msg = ToolMessage(results=tool_results)
                messages.append(tool_msg)
                self.agent_context.add_message(tool_msg)
                
                continue
            else:

                logger.warning(f'模型输出卡住, 未有输出内容')
                break
        
        duration = time.time() - start_time
        
        if timeout_reached:
            warning = f"执行超时 ({self.execution_timeout}秒)"
        else:
            warning = f"达到最大工具调用次数 ({self.max_tool_each_round})"
        
        think_event = AgentThinkEvent(
            agent_name=self.name,
            reason_content=f"[Execution] {warning}",
            duration_seconds=duration
        )
        yield think_event
        
        exec_summary = self._generate_execution_summary(execution_results, warning)
        yield exec_summary
    
    def _generate_execution_summary(
        self,
        execution_results: List[Dict[str, Any]],
        additional_info: str = ""
    ) -> str:
        """生成执行摘要"""
        if not execution_results:
            return f"本轮未执行任何工具。{additional_info}"
        
        summary_parts = [f"本轮执行了 {len(execution_results)} 次工具调用："]
        
        success_count = sum(1 for r in execution_results if r['success'])
        fail_count = len(execution_results) - success_count
        
        summary_parts.append(f"- 成功: {success_count} 次")
        summary_parts.append(f"- 失败: {fail_count} 次")
        
        tools_used = {}
        for r in execution_results:
            tool_name = r['tool']
            tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
        
        summary_parts.append("\n调用的工具:")
        for tool, count in tools_used.items():
            summary_parts.append(f"  - {tool}: {count} 次")
        
        if additional_info:
            summary_parts.append(f"\n{additional_info}")
        
        return "\n".join(summary_parts)
    
    async def _context_update_step(
        self,
        execution_summary: str,
        events: List[AgentEvent]
    ) -> AsyncGenerator[Union[AgentEvent, bool], None]:
        """
        Context Update 步骤（修正版）
        
        只使用 update_context 工具，必须调用
        """
        logger.info("=== Step 3: Context Update ===")
        start_time = time.time()
        
        update_prompt = f"""## Step 3: Context Update

**本轮执行摘要**:
{execution_summary}

**你的任务**: 汇总本轮的执行结果，并调用 `update_context` 工具更新上下文

可用的工具：
- **只有** `update_context`

你需要：
1. 查看本轮的执行结果（在上面的消息历史中）
2. 提炼关键信息
3. 调用 `update_context(summary="...", key_findings=["...", "..."])`

注意：
- **必须**调用 `update_context`
- 无论工具调用成功还是失败，都要汇总
- 失败的信息也有价值（说明什么不存在、什么路径不对等）
- 不要原封不动复制大量代码，要提炼
- summary 应该简洁（100-300字）
- key_findings 应该是 3-5 条关键发现

现在开始汇总！"""
        
        messages = self.agent_context.insert_to_agent(self.base_system_prompt)
        messages.append(UserMessage(content=update_prompt))
        
        response = await self.llm.chat(
            messages=messages,
            tools=self.context_tools,
            tool_choice= 'required',
        )
        
        llm_event = AgentLLMCallEvent(
            agent_name=self.name,
            duration_seconds=0,
            messages=messages.copy(),
            response=response
        )
        yield llm_event
        
        assistant_msg = AssistantMessage(
            content=response.content if isinstance(response.content, str) else response.content
        )
        messages.append(assistant_msg)
        self.agent_context.add_message(assistant_msg)
        
        if  isinstance(response.content, list):
            tool_calls: List[ToolCall] = response.content
            
            tool_start = time.time()
            tool_results = []
            for tc in tool_calls:

                result = await self._execute_tool(tc)
                tool_results.append(result)
            
            tool_duration = time.time() - tool_start
            
            tool_event = AgentToolRequestEvent(
                agent_name=self.name,
                tool_calls=tool_calls,
                tool_execution_results=tool_results,
                duration_seconds=tool_duration
            )
            yield tool_event
            
            tool_msg = ToolMessage(results=tool_results)
            messages.append(tool_msg)
            self.agent_context.add_message(tool_msg)
            
            success = any(not r.is_error and r.name == "update_context" for r in tool_results)
            
            duration = time.time() - start_time
            if success:
                think_event = AgentThinkEvent(
                    agent_name=self.name,
                    reason_content="[Context Update] ✅ 成功更新上下文",
                    duration_seconds=duration
                )
            else:
                think_event = AgentThinkEvent(
                    agent_name=self.name,
                    reason_content="[Context Update] ❌ 更新上下文失败",
                    duration_seconds=duration
                )
            
            yield think_event
            yield success
            return
        
        else:
            duration = time.time() - start_time
            
            think_event = AgentThinkEvent(
                agent_name=self.name,
                reason_content="[Context Update] ⚠️  模型未调用 update_context",
                duration_seconds=duration
            )
            yield think_event
            yield False
    
    async def _compress_context_if_needed(
        self,
        events: List[AgentEvent]
    ) -> AsyncGenerator[AgentThinkEvent, None]:
        """
        上下文压缩机制
        
        当上下文长度超过阈值时，调用 LLM 进行压缩
        """
        if self.enable_compress_max_length <= 0:
            return
        
        current_length = self.agent_context.get_context_length()
        
        if current_length <= self.enable_compress_max_length:
            return
        
        logger.info(f"上下文长度 ({current_length}) 超过阈值 ({self.enable_compress_max_length})，开始压缩")
        start_time = time.time()
        
        compress_prompt = f"""## 上下文压缩任务

当前上下文已经达到 {current_length} 字符，需要进行压缩以提高效率。

**你的任务**: 将当前的上下文进行压缩和精炼

**原则**:
1. **保留所有关键信息**：文件路径、函数名、重要发现、结论等
2. **记录成功的探索**：哪些工具调用成功了、发现了什么
3. **记录失败的探索**：哪些路径不存在、哪些方法不work（避免重复错误）
4. **删除冗余内容**：重复的信息、过于详细的代码片段
5. **结构化组织**：按模块/主题分类整理

**目标长度**: 压缩到原长度的 50-60%

**当前上下文**:
{self.agent_context.context}

**请输出压缩后的上下文**（直接输出内容，不要有额外说明）:"""
        
        messages = [
            SystemMessage(content="你是一个专业的上下文压缩专家，能够提炼关键信息、保留重要路径、记录失败经验。"),
            UserMessage(content=compress_prompt)
        ]
        
        response = await self.llm.chat(messages=messages)
        
        compressed_context = response.content if isinstance(response.content, str) else str(response.content)
        
        old_length = current_length
        new_length = len(compressed_context)
        compression_ratio = (1 - new_length / old_length) * 100 if old_length > 0 else 0
        
        self.agent_context.context = compressed_context
        
        duration = time.time() - start_time
        
        think_event = AgentThinkEvent(
            agent_name=self.name,
            reason_content=f"[Context Compression] 压缩完成：{old_length} → {new_length} 字符（压缩率 {compression_ratio:.1f}%）",
            duration_seconds=duration
        )
        
        logger.info(f"上下文压缩完成：{old_length} → {new_length} 字符（压缩率 {compression_ratio:.1f}%）")
        
        yield think_event
    
    async def _execute_tool(self, tool_call: ToolCall) -> ToolExecutionResult:
        """执行工具调用"""
        all_tools = self.execution_tools + self.context_tools
        
        tool = next((t for t in all_tools if t.name == tool_call.tool_name), None)
        if tool is None:
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                name=tool_call.tool_name,
                content=f"错误：工具 '{tool_call.tool_name}' 不存在",
                is_error=True
            )
        
        try:
            result = await tool.execute(tool_call)
            return result
        except Exception as e:
            logger.error(f"工具执行错误: {e}", exc_info=True)
            return ToolExecutionResult(
                tool_call_id=tool_call.id,
                name=tool_call.tool_name,
                content=f"工具执行错误: {e}",
                is_error=True
            )
    
    async def run(self, task: str) -> AgentResponse:
        """同步运行"""
        final_response = None
        async for message in self.run_stream(task):
            if isinstance(message, AgentResponse):
                final_response = message
        
        if final_response is None:
            raise RuntimeError("Agent 未产生最终响应")
        
        return final_response
    
    def save_memories(self) -> None:
        """保存所有仓库记忆"""
        if not self.using_repo_memory:
            return
        
        for memory in self.repo_memories.values():
            memory.save_to_disk()
        
        logger.info(f"已保存 {len(self.repo_memories)} 个仓库的记忆")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """获取上下文摘要"""
        return {
            "context_length": self.agent_context.get_context_length(),
            "history_length": self.agent_context.get_history_length(),
            "current_round": self._current_round,
            "context_preview": self.agent_context.context[:500] + "..." if self.agent_context.context else "(空)"
        }
