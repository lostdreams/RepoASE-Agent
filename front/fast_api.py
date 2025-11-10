"""
Agent Chat Monitor - Multi-Agent Backend Server
支持多个Agent实例的后端服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, Dict, List
import json
import asyncio
import os
from datetime import datetime

# 导入你的Agent基类
from repo_agent.core.agent_types import BaseAgent, AgentResponse
from repo_agent.core.openai_llm import OpenAILLM
from repo_agent.core.types import FunctionTool

app = FastAPI(title="Multi-Agent Chat Monitor API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Agent注册中心 =============
# 存储所有可用的Agent实例
AGENT_REGISTRY: Dict[str, BaseAgent] = {}

def register_agent(agent: BaseAgent) -> None:
    """注册一个Agent实例"""
    AGENT_REGISTRY[agent.name] = agent
    print(f"✅ 已注册Agent: {agent.name}")

def get_agent(agent_name: str) -> BaseAgent:
    """获取指定的Agent实例"""
    agent = AGENT_REGISTRY.get(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return agent

# ============= 初始化示例Agent =============
def initialize_agents():
    """初始化并注册所有Agent实例"""
    from repo_agent.agent.functioncall_agent import FunctionCallAgent
    from repo_agent.test.test_agent import mlops_llm, qwne_480b_coder_llm, TestAgent
    
  
    register_agent(TestAgent().get_hm_kernel_agent())


# ============= API端点 =============

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化Agent"""
    print("=" * 70)
    print("🚀 正在初始化Agent...")
    initialize_agents()
    print(f"✅ 已注册 {len(AGENT_REGISTRY)} 个Agent")
    print("=" * 70)

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端HTML界面"""
    html_file = os.path.join(os.path.dirname(__file__), "agent_chat.html")
    
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        # 自动替换HTML中的后端地址为当前服务器地址
        html_content = html_content.replace(
            'value="http://localhost:8000"',
            'value=""'
        )
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content=f"""
            <html>
            <head><title>Agent Chat Monitor</title></head>
            <body style="font-family: sans-serif; padding: 40px; max-width: 800px; margin: 0 auto;">
                <h1>⚠️ HTML文件未找到</h1>
                <p>请确保 <code>agent_chat_monitor.html</code> 与后端文件在同一目录下</p>
                <p><strong>当前目录:</strong> {os.path.dirname(__file__)}</p>
                <hr>
                <h2>📚 可用端点:</h2>
                <ul>
                    <li><a href="/health">/health</a> - 健康检查</li>
                    <li><a href="/agents">/agents</a> - Agent列表</li>
                    <li><a href="/docs">/docs</a> - API文档</li>
                </ul>
                <hr>
                <h2>🤖 已注册的Agent ({len(AGENT_REGISTRY)}个):</h2>
                <ul>
                    {''.join(f'<li><strong>{name}</strong>: {getattr(agent, "description", "无描述")}</li>' for name, agent in AGENT_REGISTRY.items())}
                </ul>
            </body>
            </html>
        """)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "Agent backend is running",
        "agents_count": len(AGENT_REGISTRY)
    }

@app.get("/agents")
async def list_agents():
    """获取所有可用的Agent列表"""
    agents_info = []
    for name, agent in AGENT_REGISTRY.items():
        agents_info.append({
            "name": agent.name,
            "description": getattr(agent, 'description', None) or "无描述",
            "system_prompt": getattr(agent, 'system_prompt', None),
            "tools_count": len(getattr(agent, 'tools', []))
        })
    
    return {"agents": agents_info}

@app.get("/agent/stream")
async def agent_stream(task: str, agent_name: str):
    """
    SSE流式端点 - 使用指定的Agent处理任务
    
    参数:
        task: 用户任务
        agent_name: 要使用的Agent名称
    """
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 获取指定的Agent
            agent = get_agent(agent_name)
            
            # 使用Agent的run_stream方法
            async for event in agent.run_stream(task):
                # 将事件转换为SSE格式
                event_data = event.model_dump() if hasattr(event, 'model_dump') else event.dict()
                
                # 添加事件类型标识
                event_type = event.__class__.__name__
                event_data['event_type'] = event_type
                
                # 发送SSE事件
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                
                # 如果是最终响应,发送特殊事件
                if event_type == 'AgentResponse':
                    yield f"event: agent_response\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                    
        except HTTPException as e:
            # Agent不存在的错误
            error_event = {
                "event_type": "error",
                "error": e.detail,
                "create_time": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            # 其他错误
            error_event = {
                "event_type": "error",
                "error": str(e),
                "create_time": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/agent/run")
async def agent_run(task: str, agent_name: str):
    """
    非流式端点 - 等待Agent完成后返回完整结果
    """
    try:
        agent = get_agent(agent_name)
        result = await agent.run(task)
        return result.model_dump()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 使用说明 =============
"""
如何添加你自己的Agent:

1. 创建你的Agent类（继承BaseAgent）:
   
   class MyCustomAgent(BaseAgent):
       def __init__(self, name, llm, ...):
           super().__init__(name, llm, ...)
       
       async def run_stream(self, task):
           # 实现你的逻辑
           yield AgentInputEvent(...)
           # ... 其他事件
           yield AgentResponse(...)

2. 在initialize_agents()函数中注册:
   
   my_agent = MyCustomAgent(
       name="MyAgent",
       llm=llm,
       description="我的自定义Agent"
   )
   register_agent(my_agent)

3. 启动服务后，前端会自动显示你的Agent在下拉列表中

4. 用户选择你的Agent后，所有对话都会路由到你的Agent的run_stream方法
"""

if __name__ == "__main__":
    import uvicorn
    import socket
    
    port = int(os.getenv("PORT", 8000))
    
    try:
        container_ip = socket.gethostbyname(socket.gethostname())
    except:
        container_ip = "unknown"
    
    print("=" * 70)
    print("🚀 启动Multi-Agent Chat Monitor后端服务...")
    print("=" * 70)
    print(f"📡 监听地址: http://0.0.0.0:{port}")
    print(f"🖥️  容器IP: {container_ip}")
    print("=" * 70)
    print("📚 可用端点:")
    print(f"   - 首页: /")
    print(f"   - 健康检查: /health")
    print(f"   - Agent列表: /agents")
    print(f"   - API文档: /docs")


    print(f'前端页面为 http://7.220.179.129:32633')
    print(f"   - Agent流式: /agent/stream?task=xxx&agent_name=xxx")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
