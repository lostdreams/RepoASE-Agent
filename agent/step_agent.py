"""
传统的FunctionCallAgent
-  不断的tool call
- 


step_agent执行策略:
    输入参数    
        - steps
        - 
    每次选择一个step， 
    
    step选择后，执行step 直到 模型实现当前的agent的内容
    

    - 使用单个function call agent执行step任务, 
    - 执行完毕后, response的内容

"""

from typing import List, Sequence, Optional
from pydantic import BaseModel, Field

from repo_agent.core.agent_types import (
    AgentResponse,
    BaseAgent,
    AgentEvent,
    AgentInputEvent,
    AgentThinkEvent,
    AgentToolRequestEvent
    
)

from repo_agent.core.types import (
    BaseLLM,
    LLMMessage,
    Tool,
    ToolCall,
    ToolExecutionResult
)

from repo_agent.agent  import FunctionCallAgent

import logging
logger = logging.getLogger()



class AgentStep(BaseModel):
    
    step_name: str
    step_prompt : str
    step_tools : List[Tool] | None
    step_description : str  = Field(description= '该步骤主要面向的问题类型,和决策场景')
    expect_answer_format : str | BaseModel  = ""


class StepAgentConfig:
    
    step_space : Sequence[AgentStep]
    first_step_name: str | None = None  # 如果没有指定，则由llm动态生成
    finish_step_name :  str 



class AgentStepTask: 
    step_object :  AgentStep
    step_task : str 



class AgentStepTaskAnswer(AgentStepTask): 
    
    step_answer : str
    step_response :  AgentResponse  




class AgentStepEvent(AgentEvent):

    pass


class AgentS

    
class StepAgent(BaseAgent):


    def __init__(self,
                name: str,   
                llm: BaseLLM, 
                step_agent_config : StepAgentConfig,
                tools : List[Tool],
                description: str | None = None, 
                
                system_prompt: Optional[str] = None,

        
            ):
        super().__init__(name, llm, description, system_prompt)
        #

        self.step_history : List[Dict[str, AgentStepTaskAnswer]]= {} # 记录历史的step的结果
    
        self._all_tools = tools

        

        self._all_chooseable_step_names : [name for  ]

        self.step_history_messages : Dict[str, List[LLMMessage]]= {} # step_name ： 

    
        self.global_messages : List[LLMMessage] = []

    
    

    def _get_tool_list_by_step(self, step: AgentStep) -> List[Tool]: 

        pass
    


    def _build_step_system_prompt(self, step: AgentStep) -> str :

        """
        
        构建单个step的system prompt 从实现的
        """


    def _build_step_decision_prompt_messages(self, overall_task: str) -> str:
        """
            将 system_prompt ,当前的step_history 内容， 和 step的名称，描述等信息拼接起来

        """
        pass
    
    async  def _decide_next_step_using_llm(self) -> AgentStepTask: 
        """
        从构建的prompt中，要求模型选择一个step,  
        
        """
    
        # 要求模型标准化输出
        class NextStep(BaseModel):
            thinking: str
            next_step_task: str
            next_step : Literal[self._all_chooseable_step_names] 
            
    
    

        llm_response = await self.llm.chat(messages= , structured_output= NextStep)

        assert isinstance(llm_response.structued_content , NextStep)


   

    async def get_current_step(self): 
        pass
    
    
    

    async def run(self, task: str) -> AgentResponse:
        final_response = None
        async for message in self.run_stream(task):
            if isinstance(message, AgentResponse):
                final_response = message
        if final_response is None:
            raise RuntimeError("Agent stream finished without producing a final AgentResponse.")
        return final_response
    

    async def run_stream(self, task: str) -> AsyncGenerator[Union[AgentEvent, AgentResponse], None]:
        """"
        全部的执行流程
        
        """
        pass
    
    




    async def _execute_single_step(self, current_step : AgentStepTask)  -> AgentStepTaskAnswer: 

        """
        使用functioncall agent执行

        
        """

    
