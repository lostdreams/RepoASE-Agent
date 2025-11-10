# RepoASE  Agent

一个面向在多个代码仓下的进行自动化软件工程的repo level coding Agent, 本项目还在持续迭代中, 欢迎共同开发

## 🌟 核心特性
- **不使用多Agent协作**, 面向单Agent 的能力提升, 以 Agentcontext（已实现）, Agentstep （已实现） , subagent(待实现）等概念作为构建Agent的核心机制
- **ContextAgent** ： 在context engineering视角下, 围绕任务的context构建来重新构建Agent的执行流程
- **StepAgent**: 放弃多Agent架构，使用Step-Agent来实现在复杂长步骤下的agent的可控执行
- **RepoAgent**: 放弃向量检索，采用Agent阅读文件方式实现repo理解，使用基础的文件阅读，目录打开，文本搜索等机制来实现在repo级别下编码
- **Repo长期记忆**: 将Agent在阅读repo过程中的内容保存下来，作为Agent长期记忆，并不断更新
- **沙盒化执行环境**: 对于Multi-Repo任务，通过沙盒环境提供给Agent，并基于基础的文件查询、Linux命令执行、Git命令执行等权限
- **Event观测机制**: 通过事件流观测Agent内部工作流，实时追踪执行过程
  

---

## 🗓️ 开发计划
- [ ] repo random walk : 在repo级别上进行随机游走， 预先构建repoMemory
- [ ] 使用stepagent 重构repoagent, 并设定repo需求开发，issue解决等模版repoagent
- [ ] 实现SubAgent机制



## 📋 最新进展
- ✅ 实现Repo Agent的context 压缩特性  - *- 2025年11月10日*
- ✅ 实现Repo Agent，已支持在多个repo下进行代码编写任务， 实测可以在30+ tool调用下稳定运行  - *- 2025年11月10日*
- ✅ 实现Step Agent，类似Plan Agent和Agent Graph的实现，定义场景中所有的Step对象 *- 2025年11月5日*
- ✅ 实现基础的Function Call Agent
- ✅ 实现Web可视化界面，支持实时查看Agent执行过程


---

## 🚀 快速开始

### 安装依赖

```bash
pip install fastapi openai uvicorn
```

### 方式一：命令行使用（无前端）

#### 1. 配置 LLM

```python
from repo_agent.core.openai_llm import OpenAILLM

# 初始化LLM
llm = OpenAILLM(
    model="your-model-name",
    api_key="your-api-key",
    base_url="your-base-url",
    stream_mode=False,
)
```

#### 2. 配置代码仓库

```python
from repo_agent.tool.repo import Repo

# 配置单个或多个代码仓库
repo1 = Repo(local_path='/path/to/your/repo1')
repo2 = Repo(local_path='/path/to/your/repo2')
```

#### 3. 创建沙箱环境

```python
from repo_agent.tool.repo_sandbox import RepoSandbox

# 将多个repo组合到一个沙箱中
sandbox = RepoSandbox(repos=[repo1, repo2])
```

#### 4. 创建并运行 Agent

```python
from repo_agent.agent.repo_agent import ContextRepoAgent
import asyncio

# 创建Agent
agent = ContextRepoAgent(
    name='my_agent',
    description='My custom repo agent',
    system_prompt='Your system prompt here',
    llm=llm,
    repo_sandbox=sandbox,
)

# 运行Agent
async def main():
    agent_stream = agent.run_stream(
        task='你的任务描述'
    )
    
    # 处理流式输出
    async for event in agent_stream:
        print(event)

asyncio.run(main())
```

#### 完整示例

```python
from repo_agent.core.openai_llm import OpenAILLM
from repo_agent.tool.repo import Repo
from repo_agent.tool.repo_sandbox import RepoSandbox
from repo_agent.agent.repo_agent import ContextRepoAgent
import asyncio

# 1. 配置LLM
llm = OpenAILLM(
    model="gpt-4",
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
    stream_mode=False,
)

# 2. 配置代码仓库
repo1 = Repo(local_path='/path/to/repo1')
repo2 = Repo(local_path='/path/to/repo2')

# 3. 创建沙箱
sandbox = RepoSandbox(repos=[repo1, repo2])

# 4. 创建Agent
agent = ContextRepoAgent(
    name='code_agent',
    description='Code analysis agent',
    system_prompt='You are a helpful coding assistant.',
    llm=llm,
    repo_sandbox=sandbox,
)

# 5. 运行
async def main():
    agent_stream = agent.run_stream(
        task='分析repo1中的main函数'
    )
    
    async for event in agent_stream:
        print(event)
        if hasattr(event, 'response'):
            print(f'模型最终结果{event.response}')
            print(event.response)

asyncio.run(main())
```

### 方式二：Web界面使用（有前端）

#### 1. 在 `fast_api.py` 中注册你的Agent

```python
def initialize_agents():
    """在启动时注册所有Agent"""
    from repo_agent.core.openai_llm import OpenAILLM
    from repo_agent.tool.repo import Repo
    from repo_agent.tool.repo_sandbox import RepoSandbox
    from repo_agent.agent.repo_agent import ContextRepoAgent
    
    # 配置LLM
    llm = OpenAILLM(
        model="your-model",
        api_key="your-key",
        base_url="your-url",
        stream_mode=False,
    )
    
    # 配置代码仓库和沙箱
    repo1 = Repo(local_path='/path/to/repo1')
    repo2 = Repo(local_path='/path/to/repo2')
    sandbox = RepoSandbox(repos=[repo1, repo2])
    
    # 创建Agent
    my_agent = ContextRepoAgent(
        name='MyAgent',
        description='我的自定义Agent',
        system_prompt='Your prompt',
        llm=llm,
        repo_sandbox=sandbox,
    )
    
    # 注册Agent
    register_agent(my_agent)
```

#### 2. 启动服务器

```bash
python agent_chat_server.py
```

#### 3. 访问Web界面

打开浏览器访问：
```
http://localhost:8000
```

在界面中：
- 从下拉菜单选择你注册的Agent
- 输入任务描述
- 实时查看Agent的执行过程和结果

---

## 💡 核心概念

| 概念 | 说明 |
|------|------|
| **Repo** | 代码仓库对象，提供文件搜索、读取等工具 |
| **RepoSandbox** | 沙箱环境，可包含多个Repo，统一管理多仓库 |
| **Agent** | 智能代理，使用LLM和工具完成任务 |
| **Stream模式** | 实时流式输出Agent的思考和执行过程 |
| **Event** | 事件对象，记录Agent执行过程中的各类事件 |
| **RepoMemory** | 长期记忆系统，存储Agent对repo的理解 |

---

## 📁 项目结构

```
repo-agent/
├── repo_agent/
│   ├── core/                 # 核心模块
│   │   ├── agent_types.py    # Agent基类和事件定义
│   │   ├── openai_llm.py     # LLM接口封装
│   │   └── types.py          # 基础类型定义
│   ├── agent/                # Agent实现
│   │   ├── repo_agent.py     # Repo Agent
│   │   ├── step_agent.py     # Step Agent
│   │   └── functioncall_agent.py  # Function Call Agent
│   ├── tool/                 # 工具模块
│   │   ├── repo.py           # Repo工具
│   │   └── repo_sandbox.py   # 沙箱环境
│   └── util/                 # 工具函数
└── README.md
```

---

## 🔧  基础使用

### 使用RepoMemory

```python
from repo_agent.agent.repo_agent import RepoMemory
# 注意，这里需要修改默认的repo_memory的存放地址
# 加载已有记忆
memory = RepoMemory.load(repo_name='my-repo')

# 访问记忆内容
print(memory.context)
```

### 使用Step Agent

```python
from repo_agent.agent.step_agent import StepAgent, AgentStep

# 定义步骤
steps = [
    AgentStep(name='analyze', description='分析代码'),
    AgentStep(name='generate', description='生成代码'),
]

# 创建Step Agent
step_agent = StepAgent(
    name='multi_step_agent',
    llm=llm,
    steps=steps,
)
```

### 使用Function Call Agent

```python
from repo_agent.agent.functioncall_agent import FunctionCallAgent
from repo_agent.core.types import FunctionTool

# 定义工具函数
def my_tool(param: str) -> str:
    return f"处理: {param}"

# 创建Function Call Agent
fc_agent = FunctionCallAgent(
    name='fc_agent',
    llm=llm,
    tools=[FunctionTool(function=my_tool)],
)
```

---

## 🎯 使用场景

- **代码分析**: 分析大型代码仓库的结构和逻辑
- **代码生成**: 基于现有代码库生成新代码
- **跨仓库操作**: 在多个相关仓库之间协同工作
- **代码重构**: 理解代码上下文并进行智能重构
- **文档生成**: 自动生成代码文档和说明

---

## 📝 API文档

启动服务器后，访问自动生成的API文档：
```
http://localhost:8000/docs
```

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 📧 联系方式

如有问题或建议，欢迎通过Issue联系我们。

---

**⭐ 如果这个项目对你有帮助，欢迎 Star！**
