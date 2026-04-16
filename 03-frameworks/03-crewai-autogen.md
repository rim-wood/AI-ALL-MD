# CrewAI / AutoGen

> 多 Agent 协作框架：让多个 AI Agent 像团队一样协作完成复杂任务

## 学习目标

通过本章学习，你将能够：

- 理解多 Agent 协作的核心动机与常见协作模式
- 掌握 CrewAI 的 Agent / Task / Crew 抽象，快速搭建角色化 Agent 团队
- 掌握 AutoGen (AG2) 的异步事件驱动架构与 GroupChat 多 Agent 对话
- 了解 OpenAI Agents SDK 的 Handoff 与 Guardrails 机制
- 能够根据项目需求选择合适的多 Agent 框架
- 完成一个完整的多 Agent 研究团队实战项目

---

## 1. 多 Agent 协作概述

### 1.1 为什么需要多 Agent？

单个 LLM Agent 在处理复杂任务时面临明显瓶颈：

| 挑战 | 单 Agent 局限 | 多 Agent 方案 |
|------|--------------|--------------|
| 任务复杂度 | 单一 Prompt 难以覆盖所有子任务 | 每个 Agent 专注一个子任务 |
| 专业性 | 一个 Agent 难以同时扮演多种角色 | 不同 Agent 拥有不同专业背景 |
| 上下文窗口 | 所有信息挤在一个上下文中 | 各 Agent 维护独立上下文 |
| 可靠性 | 单点失败影响全局 | Agent 之间可以互相校验 |
| 可维护性 | 巨型 Prompt 难以调试 | 模块化设计，独立迭代 |

**核心思想**：将复杂任务分解为多个子任务，由不同的专业化 Agent 协作完成——就像一个高效团队中的不同角色各司其职。

### 1.2 多 Agent 协作模式

```
┌─────────────────────────────────────────────────────┐
│                  多 Agent 协作模式                     │
├─────────────┬──────────────┬────────────────────────┤
│   顺序模式   │   层级模式    │      对话模式           │
│ Sequential  │ Hierarchical │   Conversational       │
│             │              │                        │
│ A → B → C   │     Boss     │   A ←→ B ←→ C          │
│             │    ↙   ↘    │                        │
│ 流水线式执行  │   W1    W2   │  Agent 之间自由对话      │
│ 上一步输出    │              │  通过讨论达成共识         │
│ 作为下一步输入 │ 管理者分配任务 │  适合头脑风暴/辩论       │
│             │ 汇总结果      │                        │
└─────────────┴──────────────┴────────────────────────┘
```

**顺序模式 (Sequential)**：Agent 按固定顺序执行，前一个 Agent 的输出作为后一个的输入。适合流水线式任务，如"调研 → 撰写 → 审校"。

**层级模式 (Hierarchical)**：一个管理者 Agent 负责分配任务、汇总结果，工作者 Agent 执行具体子任务。适合需要动态调度的复杂项目。

**对话模式 (Conversational)**：多个 Agent 在群聊中自由交流，通过多轮讨论达成共识。适合需要多视角分析的场景。

---

## 2. CrewAI

CrewAI（⭐ 44,500+）是目前最流行的角色化多 Agent 框架之一，以"团队协作"为核心隐喻，让开发者像组建真实团队一样定义 Agent。

### 2.1 核心概念

CrewAI 的四大核心抽象：

```
┌──────────────────────────────────────────┐
│                  Crew                     │
│  (团队：编排 Agent 和 Task 的执行流程)      │
│                                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │
│  │ 研究员   │  │ 写作者   │  │ 审校员   │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │        │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  │
│  │ Task 1  │  │ Task 2  │  │ Task 3  │  │
│  │ 调研主题 │  │ 撰写文章 │  │ 审校润色 │  │
│  └─────────┘  └─────────┘  └─────────┘  │
│                                          │
│  Process: Sequential / Hierarchical      │
└──────────────────────────────────────────┘
```

| 概念 | 说明 | 类比 |
|------|------|------|
| **Agent** | 拥有角色、目标、背景故事的智能体 | 团队成员 |
| **Task** | 具体的工作任务，有描述和期望输出 | 工作任务单 |
| **Crew** | 编排 Agent 和 Task 的团队 | 项目团队 |
| **Process** | 任务执行流程（顺序/层级） | 工作流程 |

### 2.2 安装与配置

```bash
pip install crewai crewai-tools
```

设置环境变量：

```bash
export OPENAI_API_KEY="sk-..."
# 或使用其他模型提供商
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2.3 Agent 定义

Agent 是 CrewAI 的核心，每个 Agent 都有明确的角色定位：

```python
from crewai import Agent

# 定义一个研究员 Agent
researcher = Agent(
    role="高级研究分析师",
    goal="发现关于 {topic} 的最新趋势和关键洞察",
    backstory="""你是一位经验丰富的研究分析师，擅长从海量信息中
    提取关键洞察。你以严谨的态度和敏锐的洞察力著称，
    总能发现别人忽略的重要趋势。""",
    verbose=True,
    allow_delegation=False,  # 是否允许委派任务给其他 Agent
    max_iter=5,              # 最大推理迭代次数
)

# 定义一个写作者 Agent
writer = Agent(
    role="技术内容撰写专家",
    goal="将研究成果转化为引人入胜的技术文章",
    backstory="""你是一位资深技术写作者，擅长将复杂的技术概念
    用通俗易懂的语言表达出来。你的文章既有深度又有可读性，
    深受读者喜爱。""",
    verbose=True,
    allow_delegation=False,
)
```

**Agent 关键属性说明**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `role` | str | Agent 的角色名称，影响其行为风格 |
| `goal` | str | Agent 的目标，支持 `{variable}` 模板变量 |
| `backstory` | str | 背景故事，为 Agent 提供上下文和个性 |
| `tools` | list | Agent 可使用的工具列表 |
| `allow_delegation` | bool | 是否允许将任务委派给其他 Agent |
| `max_iter` | int | 单次任务最大推理迭代次数 |
| `memory` | bool | 是否启用记忆功能 |
| `llm` | str | 指定使用的 LLM 模型 |

### 2.4 Task 定义

Task 描述了 Agent 需要完成的具体工作：

```python
from crewai import Task

# 研究任务
research_task = Task(
    description="""对 {topic} 进行全面深入的研究分析。
    要求：
    1. 识别该领域的 5 个最新趋势
    2. 分析每个趋势的影响和前景
    3. 提供数据支撑和案例佐证""",
    expected_output="""一份结构化的研究报告，包含：
    - 5 个关键趋势及其详细分析
    - 每个趋势的数据支撑
    - 未来发展预测""",
    agent=researcher,
)

# 写作任务 —— 依赖研究任务的输出
writing_task = Task(
    description="""基于研究报告，撰写一篇面向技术读者的深度文章。
    要求：
    1. 文章长度 1500-2000 字
    2. 包含引言、正文（分段论述）、总结
    3. 语言通俗易懂，避免过度学术化""",
    expected_output="一篇完整的技术博客文章，Markdown 格式",
    agent=writer,
    context=[research_task],  # 将研究任务的输出作为上下文
    output_file="output/article.md",  # 输出到文件
)
```

### 2.5 Crew 编排

Crew 将 Agent 和 Task 组合在一起，定义执行流程：

```python
from crewai import Crew, Process

# 创建团队 —— 顺序执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
    memory=True,  # 启用团队记忆
)

# 启动执行，传入模板变量
result = crew.kickoff(inputs={"topic": "2026 年 AI Agent 发展趋势"})
print(result)
```

**层级模式**需要指定管理者 LLM：

```python
from crewai import Crew, Process

hierarchical_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,  # 层级执行
    manager_llm="gpt-4o",         # 管理者使用的模型
    verbose=True,
)
```

### 2.6 工具集成

CrewAI 提供了丰富的内置工具，也支持自定义工具：

```python
from crewai_tools import (
    SerperDevTool,      # Google 搜索
    WebsiteSearchTool,  # 网站内容搜索
    FileReadTool,       # 文件读取
    PDFSearchTool,      # PDF 搜索
)

# 使用内置工具
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

researcher_with_tools = Agent(
    role="高级研究分析师",
    goal="发现关于 {topic} 的最新趋势",
    backstory="你是一位经验丰富的研究分析师...",
    tools=[search_tool, web_tool],  # 赋予工具
    verbose=True,
)
```

**自定义工具**：

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class StockPriceInput(BaseModel):
    symbol: str = Field(description="股票代码，如 AAPL")


class StockPriceTool(BaseTool):
    name: str = "stock_price"
    description: str = "查询指定股票的当前价格"
    args_schema: type[BaseModel] = StockPriceInput

    def _run(self, symbol: str) -> str:
        # 实际项目中调用真实 API
        prices = {"AAPL": 198.50, "GOOGL": 175.30, "MSFT": 420.10}
        price = prices.get(symbol.upper(), None)
        if price:
            return f"{symbol.upper()} 当前价格: ${price}"
        return f"未找到 {symbol} 的价格数据"
```

### 2.7 记忆与学习

CrewAI 内置了多层记忆系统，让 Agent 能够积累经验：

```
┌─────────────────────────────────────────┐
│            CrewAI 记忆系统                │
├─────────────┬─────────────┬─────────────┤
│  短期记忆     │  长期记忆     │  实体记忆    │
│ Short-term  │ Long-term   │  Entity     │
│             │             │             │
│ 当前任务执行  │ 跨任务经验    │ 关键实体信息  │
│ 过程中的上下文 │ 持久化存储    │ 人物/组织等   │
│             │ 用于未来参考  │             │
└─────────────┴─────────────┴─────────────┘
```

```python
crew_with_memory = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    memory=True,            # 启用记忆
    embedder={              # 自定义 Embedding 模型
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
        },
    },
)
```

启用记忆后，Agent 在后续执行中可以：
- **短期记忆**：在同一次 Crew 执行中，后续 Task 可以参考前序 Task 的详细过程
- **长期记忆**：跨多次执行积累经验，提升任务完成质量
- **实体记忆**：记住关键实体（人物、组织、概念）的信息，保持一致性

---

## 3. AutoGen (AG2)

AutoGen（⭐ 54,700+）由 Microsoft 开源，现已演进为 AG2（Microsoft Agent Framework）。v0.4 版本采用全新的异步事件驱动架构，提供从高层到底层的多层 API。

> **品牌说明**：AutoGen 现在也被称为 **Microsoft Agent Framework**，获得了微软的企业级支持，并与 Azure 生态深度集成（Azure AI Foundry、Azure OpenAI Service 等）。如果你在企业环境中评估多 Agent 框架，AutoGen 的微软背书和 Azure 原生集成是重要的加分项。

### 3.1 架构分层与 v0.4 重新设计

AutoGen v0.4 是一次彻底的架构重新设计，围绕**异步、事件驱动、Actor 模型**重新构建了整个 agentic AI 框架，并提供分层 API 以适应不同复杂度的需求。

**v0.4 核心设计理念**：

| 设计维度 | 说明 |
|---------|------|
| **异步优先** | 所有 Agent 交互基于 `async/await`，天然支持高并发 |
| **事件驱动** | Agent 之间通过事件（Event）通信，解耦发送者和接收者 |
| **Actor 模型** | 每个 Agent 是独立的 Actor，拥有自己的状态和消息队列 |
| **分层 API** | AgentChat（高层）→ Core（底层）→ Extensions（扩展），按需选择抽象层级 |

AG2 社区在此基础上推出了 **AG2 Beta**（`autogen.beta`），进一步增强了以下能力：

- **Streaming 支持**：Agent 响应支持流式输出，提升用户体验
- **事件驱动架构**：完整的事件生命周期管理，便于监控和调试
- **多 Provider LLM 支持**：统一接口对接 OpenAI、Anthropic、Google、本地模型等
- **依赖注入**：通过 DI 容器管理 Agent 依赖，提升可测试性
- **Typed Tools**：工具定义支持完整的类型标注，编译时检查参数类型
- **First-class Testing**：内置测试工具和 Mock 支持，Agent 行为可单元测试

```
┌─────────────────────────────────────────┐
│              AutoGen v0.4 架构            │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  AgentChat（高层 API）              │  │
│  │  快速构建多 Agent 对话              │  │
│  │  AssistantAgent / GroupChat       │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │  Core（核心层）                     │  │
│  │  异步消息传递 / 事件驱动             │  │
│  │  Agent 运行时 / 消息路由            │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │  Extensions（扩展层）               │  │
│  │  代码执行 / 工具集成 / 模型客户端    │  │
│  └───────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

| 层级 | 说明 | 适用场景 |
|------|------|---------|
| **AgentChat** | 高层 API，快速构建对话式 Agent | 快速原型、标准多 Agent 对话 |
| **Core** | 异步事件驱动的底层运行时 | 自定义 Agent 行为、复杂编排 |
| **Extensions** | 可插拔的扩展组件 | 代码执行、外部工具集成 |

### 3.2 安装与配置

```bash
# 安装 AgentChat 高层 API（推荐入门）
pip install autogen-agentchat autogen-ext[openai,docker]
```

### 3.3 Agent 类型

AutoGen 提供多种预置 Agent 类型：

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 创建模型客户端
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# AssistantAgent —— 最常用的 Agent 类型
assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="""你是一个有帮助的 AI 助手，擅长回答技术问题。
    回答时要准确、简洁、有条理。""",
)
```

**使用工具的 Agent**：

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# 定义工具函数
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，22°C",
        "深圳": "阵雨，28°C",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


model_client = OpenAIChatCompletionClient(model="gpt-4o")

weather_agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],  # 注册工具
    system_message="你是天气查询助手，使用工具查询天气信息。",
)
```

### 3.4 双 Agent 对话

AutoGen 最经典的模式是两个 Agent 之间的对话：

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# 定义两个 Agent
primary_agent = AssistantAgent(
    name="primary",
    model_client=model_client,
    system_message="""你是一个技术方案设计师。
    针对用户需求提出技术方案，并根据评审意见改进。
    当方案获得认可后，回复 APPROVE 结束讨论。""",
)

critic_agent = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="""你是一个严格的技术评审专家。
    审查技术方案的可行性、性能、安全性。
    如果方案合理，回复 APPROVE。否则提出改进建议。""",
)

# 终止条件：当出现 APPROVE 时结束
termination = TextMentionTermination("APPROVE")

# 创建轮询式群聊
team = RoundRobinGroupChat(
    participants=[primary_agent, critic_agent],
    termination_condition=termination,
    max_turns=10,
)


async def main():
    result = await team.run(task="设计一个支持 10 万并发的实时聊天系统")
    for message in result.messages:
        print(f"[{message.source}]: {message.content[:200]}")
        print("---")


asyncio.run(main())
```

### 3.5 GroupChat 多 Agent 对话

GroupChat 允许多个 Agent 在同一个对话中协作：

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# 定义多个专业 Agent
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="""你是项目规划师，负责：
    1. 分析需求，拆解任务
    2. 制定执行计划
    3. 协调团队成员""",
)

coder = AssistantAgent(
    name="coder",
    model_client=model_client,
    system_message="""你是高级程序员，负责：
    1. 根据计划编写代码
    2. 实现具体功能
    3. 代码需要完整可运行""",
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="""你是代码审查专家，负责：
    1. 审查代码质量和安全性
    2. 提出改进建议
    3. 确认代码符合要求后回复 APPROVE""",
)

termination = TextMentionTermination("APPROVE")

# SelectorGroupChat —— 由模型动态选择下一个发言者
team = SelectorGroupChat(
    participants=[planner, coder, reviewer],
    model_client=model_client,  # 用于选择下一个发言者
    termination_condition=termination,
    max_turns=15,
)


async def main():
    result = await team.run(
        task="用 Python 实现一个简单的 LRU Cache，支持 get 和 put 操作"
    )
    for msg in result.messages:
        print(f"[{msg.source}]: {msg.content[:300]}")
        print("---")


asyncio.run(main())
```

### 3.6 代码执行

AutoGen 的一大特色是内置安全的代码执行环境：

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")


async def main():
    # Docker 沙箱代码执行器（推荐生产环境使用）
    code_executor = DockerCommandLineCodeExecutor(
        image="python:3.12-slim",
        timeout=60,
        work_dir="/tmp/code_exec",
    )

    # 代码编写 Agent
    coder = AssistantAgent(
        name="coder",
        model_client=model_client,
        system_message="""你是 Python 专家。编写代码解决问题。
        代码用 ```python 代码块包裹。
        任务完成后回复 TERMINATE。""",
    )

    # 代码执行 Agent —— 自动执行代码并返回结果
    executor = CodeExecutorAgent(
        name="executor",
        code_executor=code_executor,
    )

    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[coder, executor],
        termination_condition=termination,
        max_turns=10,
    )

    result = await team.run(
        task="分析以下数据并绘制柱状图：{'Python': 35, 'JavaScript': 28, 'Go': 15, 'Rust': 12}"
    )
    print(result.messages[-1].content)


asyncio.run(main())
```

> **安全提示**：生产环境中务必使用 Docker 沙箱执行代码，避免恶意代码对宿主机造成损害。本地开发可使用 `LocalCommandLineCodeExecutor`，但不建议在生产环境使用。

---

## 4. OpenAI Agents SDK

OpenAI Agents SDK 是 OpenAI 官方推出的轻量级 Agent 框架，核心理念是"足够简单，足够强大"。它围绕三个核心概念构建：Agent、Handoff、Guardrails。

### 4.1 核心概念

```
┌─────────────────────────────────────────────┐
│          OpenAI Agents SDK 核心概念           │
│                                             │
│  ┌─────────┐  Handoff   ┌─────────┐        │
│  │ Agent A │ ─────────→ │ Agent B │        │
│  │ 分诊助手 │            │ 技术支持 │        │
│  └─────────┘            └─────────┘        │
│       │                      │              │
│       │ Guardrails           │ Guardrails   │
│       ▼                      ▼              │
│  ┌─────────┐           ┌─────────┐         │
│  │ 输入检查 │           │ 输出检查 │         │
│  │ 安全过滤 │           │ 质量保证 │         │
│  └─────────┘           └─────────┘         │
└─────────────────────────────────────────────┘
```

| 概念 | 说明 |
|------|------|
| **Agent** | 拥有指令、工具和模型配置的智能体 |
| **Handoff** | Agent 之间的任务移交机制 |
| **Guardrails** | 输入/输出的安全检查与质量保证 |

### 4.2 安装与配置

```bash
pip install openai-agents
```

### 4.3 基础 Agent

```python
from agents import Agent, Runner
import asyncio

agent = Agent(
    name="技术助手",
    instructions="""你是一个专业的技术助手。
    回答用户的技术问题，提供准确、实用的建议。
    如果不确定，坦诚告知而非编造答案。""",
    model="gpt-4o",
)


async def main():
    result = await Runner.run(agent, "解释 Python 中的 GIL 是什么？")
    print(result.final_output)


asyncio.run(main())
```

### 4.4 多 Agent Handoff

Handoff 是 Agents SDK 最核心的特性——让 Agent 之间可以无缝移交任务：

```python
import asyncio
from agents import Agent, Runner

# 定义专业 Agent
billing_agent = Agent(
    name="billing_agent",
    instructions="""你是账单专家，处理所有与账单、付款、退款相关的问题。
    提供清晰的账单说明和操作指引。""",
    model="gpt-4o",
)

tech_agent = Agent(
    name="tech_agent",
    instructions="""你是技术支持专家，处理所有技术问题。
    包括 Bug 排查、功能使用指导、API 集成等。""",
    model="gpt-4o",
)

# 分诊 Agent —— 根据用户问题路由到合适的专业 Agent
triage_agent = Agent(
    name="triage_agent",
    instructions="""你是客服分诊助手。分析用户问题的类型：
    - 账单相关问题 → 转交给 billing_agent
    - 技术相关问题 → 转交给 tech_agent
    - 其他问题 → 自己回答""",
    handoffs=[billing_agent, tech_agent],  # 可以移交的目标 Agent
    model="gpt-4o",
)


async def main():
    # 用户提出技术问题 → triage_agent 自动路由到 tech_agent
    result = await Runner.run(triage_agent, "我的 API 调用一直返回 429 错误")
    print(f"最终回答者: {result.last_agent.name}")
    print(f"回答内容: {result.final_output}")


asyncio.run(main())
```

### 4.5 Guardrails 安全护栏

Guardrails 为 Agent 添加输入/输出的安全检查：

```python
import asyncio
from agents import (
    Agent,
    Runner,
    InputGuardrail,
    GuardrailFunctionOutput,
    input_guardrail,
)
from pydantic import BaseModel


class SafetyCheck(BaseModel):
    is_safe: bool
    reason: str


safety_agent = Agent(
    name="safety_checker",
    instructions="判断用户输入是否安全合规。检查是否包含有害、违法或不当内容。",
    output_type=SafetyCheck,
    model="gpt-4o-mini",
)


@input_guardrail
async def check_safety(ctx, agent, input_text):
    """输入安全检查 Guardrail"""
    result = await Runner.run(safety_agent, input_text, context=ctx.context)
    safety = result.final_output_as(SafetyCheck)
    return GuardrailFunctionOutput(
        output_info=safety,
        tripwire_triggered=not safety.is_safe,  # 不安全时触发拦截
    )


# 主 Agent 配置 Guardrail
main_agent = Agent(
    name="main_assistant",
    instructions="你是一个有帮助的助手，回答用户问题。",
    input_guardrails=[check_safety],  # 输入安全检查
    model="gpt-4o",
)


async def main():
    try:
        result = await Runner.run(main_agent, "帮我写一个 Python 排序算法")
        print(result.final_output)
    except Exception as e:
        print(f"请求被拦截: {e}")


asyncio.run(main())
```

---

## 5. 框架对比与选型

### 5.1 核心对比

| 维度 | CrewAI | AutoGen (AG2) | OpenAI Agents SDK |
|------|--------|---------------|-------------------|
| **GitHub Stars** | 44,500+ | 54,700+ | 较新 |
| **核心理念** | 角色化团队协作 | 异步事件驱动对话 | 极简 Agent 编排 |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **生产就绪** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **异步支持** | 基础 | 原生异步 | 原生异步 |
| **状态持久化** | 内置记忆系统 | 需自行实现 | 需自行实现 |
| **代码执行** | 通过工具 | 内置 Docker 沙箱 | 通过工具 |
| **模型支持** | 多模型 | 多模型 | 仅 OpenAI |
| **学习曲线** | 低 | 中高 | 低 |
| **社区生态** | 活跃 | 非常活跃 | 成长中 |

### 5.2 性能基准测试

以下是基于真实场景（4-Agent 研究流水线任务）的性能基准数据，供选型参考：

| 指标 | CrewAI | AutoGen (AG2) | LangGraph |
|------|--------|---------------|-----------|
| **完成时间** | 45-60 秒（Sequential） | 30-40 秒（Async） | 25-35 秒（并行节点） |
| **执行模式** | 顺序流水线 | 异步事件驱动 | 状态图并行执行 |
| **适合场景** | 流程固定、角色明确 | 动态对话、高并发 | 复杂状态流转、需要并行 |

> **说明**：时间差异主要来自执行模式——CrewAI 的 Sequential 模式严格串行，AutoGen 的异步架构允许部分并行，LangGraph 的图结构天然支持无依赖节点并行执行。

**企业级案例数据**：Deloitte 在客户项目中使用 CrewAI 构建多 Agent 系统，报告了 **89% 的任务成功率**，平均每次查询成本约 **$0.12**。这表明 CrewAI 在成本敏感的企业场景中具有良好的性价比。

### 5.3 选型决策树

```
你的项目需要什么？
│
├─ 快速原型 / MVP
│  └─→ CrewAI（最快上手，角色化定义直观）
│
├─ 复杂的多 Agent 对话 / 辩论
│  └─→ AutoGen（GroupChat 灵活，支持动态发言者选择）
│
├─ 需要安全的代码执行
│  └─→ AutoGen（内置 Docker 沙箱）
│
├─ 仅使用 OpenAI 模型 + 需要简洁 API
│  └─→ OpenAI Agents SDK（官方支持，Handoff 机制优雅）
│
├─ 需要 Agent 记忆 / 经验积累
│  └─→ CrewAI（内置短期/长期/实体记忆）
│
├─ 高并发 / 生产级部署
│  └─→ AutoGen（原生异步，事件驱动架构）
│
└─ 需要输入/输出安全护栏
   └─→ OpenAI Agents SDK（内置 Guardrails）
```

### 5.4 快速决策流程图

如果你不想逐条对比，可以用这个简化流程图快速决策：

```
┌─────────────────────────────────────────────────────────┐
│                    你的核心需求是什么？                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  "我需要快速出原型"                                       │
│   └──→ ✅ CrewAI                                        │
│        角色化定义直观，10 分钟搭建 MVP                      │
│                                                         │
│  "我需要可扩展的多 Agent 对话"                              │
│   └──→ ✅ AutoGen (AG2)                                 │
│        原生异步 + 事件驱动，GroupChat 灵活编排               │
│                                                         │
│  "我需要生产级有状态工作流"                                  │
│   └──→ ✅ LangGraph                                     │
│        类型化状态、检查点、人工审批、可视化调试               │
│                                                         │
│  "我还不确定"                                             │
│   └──→ 🟡 从 CrewAI 开始                                │
│        遇到瓶颈时迁移到 LangGraph                         │
│        （CrewAI 的 Task 抽象容易映射到 LangGraph 节点）     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

> **迁移建议**：CrewAI → LangGraph 的迁移路径相对清晰——每个 CrewAI Task 对应一个 LangGraph 节点，Agent 的 role/goal/backstory 转化为节点的 system prompt。主要工作量在于将隐式的 Task 输出传递改为显式的 State 定义。

### 5.5 状态流转机制对比

三大框架在**状态如何在 Agent 之间流转**这一核心问题上采用了截然不同的设计，这直接影响调试、测试和扩展的难度：

```
┌─────────────────────────────────────────────────────────────┐
│                    状态流转机制对比                            │
├───────────────┬─────────────────────────────────────────────┤
│               │                                             │
│   CrewAI      │  Task Output 传递                           │
│               │  Task A 的输出 → 作为 Task B 的 context      │
│               │  隐式传递，框架自动管理                        │
│               │                                             │
│               │  research_task ──output──→ writing_task     │
│               │                (context=[research_task])    │
│               │                                             │
├───────────────┼─────────────────────────────────────────────┤
│               │                                             │
│   AutoGen     │  Message History 对话历史                    │
│               │  所有 Agent 共享同一个对话上下文               │
│               │  通过消息历史传递信息                          │
│               │                                             │
│               │  Agent A ──msg──→ [Chat History] ──msg──→ B │
│               │                                             │
├───────────────┼─────────────────────────────────────────────┤
│               │                                             │
│   LangGraph   │  Typed State Dict 类型化状态字典              │
│               │  每个节点读写同一个强类型 State 对象            │
│               │  状态变更显式、可追踪、可序列化                 │
│               │                                             │
│               │  Node A ──write──→ {State} ──read──→ Node B │
│               │                                             │
└───────────────┴─────────────────────────────────────────────┘
```

| 维度 | CrewAI (Task Output) | AutoGen (Message History) | LangGraph (Typed State) |
|------|---------------------|--------------------------|------------------------|
| **调试** | 查看 Task 输出即可 | 需要翻阅完整对话记录 | State diff 精确定位变更 |
| **测试** | Mock Task 输出 | Mock 消息序列 | 直接构造 State 快照 |
| **持久化** | 内置记忆系统 | 需自行序列化消息 | 内置 Checkpoint 机制 |
| **可预测性** | 中（依赖 LLM 输出格式） | 低（对话走向不确定） | 高（状态变更显式定义） |
| **扩展性** | 中 | 高（消息天然可分发） | 高（状态可分片） |

> **实践建议**：如果你的项目需要严格的可审计性（如金融、医疗场景），LangGraph 的 Typed State 机制是最佳选择——每一步状态变更都可追踪和回放。如果是快速迭代的内部工具，CrewAI 的隐式传递足够简单高效。

### 5.6 组合使用建议

在实际项目中，这些框架并非互斥。常见的组合策略：

- **CrewAI + 自定义工具**：用 CrewAI 快速搭建 Agent 团队，通过自定义工具接入企业内部系统
- **AutoGen 做底层 + 自定义高层封装**：利用 AutoGen Core 的灵活性构建自定义编排逻辑
- **OpenAI Agents SDK 做入口路由**：用 Handoff 机制做请求分发，后端对接不同的专业系统

---

## 6. 实战：研究团队 Agent

本节用 CrewAI 构建一个完整的"研究团队"多 Agent 系统，模拟真实的研究工作流：调研 → 分析 → 撰写 → 审校。

### 6.1 需求设计

```
用户输入研究主题
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  调研员       │ ──→ │  分析师       │ ──→ │  撰写者       │ ──→ │  审校员       │
│  Researcher  │     │  Analyst     │     │  Writer      │     │  Reviewer    │
│              │     │              │     │              │     │              │
│ · 搜索资料    │     │ · 提取洞察    │     │ · 撰写报告    │     │ · 审校质量    │
│ · 收集数据    │     │ · 趋势分析    │     │ · 结构化输出  │     │ · 提出修改    │
│ · 整理来源    │     │ · 对比评估    │     │ · Markdown   │     │ · 最终确认    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
                                                              最终研究报告
```

### 6.2 完整代码实现

```python
"""
研究团队 Agent —— 基于 CrewAI 的多 Agent 协作示例

功能：输入一个研究主题，自动完成调研、分析、撰写、审校的完整流程
依赖：pip install crewai crewai-tools
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# ── 环境配置 ──────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-...")
os.environ.setdefault("SERPER_API_KEY", "...")  # 搜索工具需要

# ── 工具 ──────────────────────────────────────────────────
search_tool = SerperDevTool()

# ── Agent 定义 ────────────────────────────────────────────

researcher = Agent(
    role="高级研究调研员",
    goal="对 {topic} 进行全面的资料收集和信息整理",
    backstory="""你是一位资深研究调研员，拥有 10 年的行业研究经验。
    你擅长从多个来源收集信息，交叉验证数据的准确性，
    并整理出结构清晰的调研素材。你特别注重数据的时效性和可靠性。""",
    tools=[search_tool],
    verbose=True,
    max_iter=5,
)

analyst = Agent(
    role="数据分析专家",
    goal="从调研素材中提取关键洞察和趋势分析",
    backstory="""你是一位数据分析专家，擅长从大量信息中发现模式和趋势。
    你的分析总是有数据支撑，逻辑严密，能够给出有价值的预测和建议。
    你善于用对比分析和 SWOT 分析等方法论来组织你的分析。""",
    verbose=True,
    max_iter=5,
)

writer = Agent(
    role="技术报告撰写专家",
    goal="将分析结果转化为高质量的研究报告",
    backstory="""你是一位技术写作专家，擅长将复杂的分析结果
    转化为结构清晰、逻辑严密的研究报告。你的报告既有深度又有可读性，
    善于使用图表描述、数据引用和案例分析来增强说服力。""",
    verbose=True,
    max_iter=5,
)

reviewer = Agent(
    role="质量审校专家",
    goal="确保研究报告的准确性、完整性和专业性",
    backstory="""你是一位严格的质量审校专家，拥有丰富的学术审稿经验。
    你会从事实准确性、逻辑一致性、论述完整性、语言规范性等多个维度
    审查报告，并给出具体的修改建议。""",
    verbose=True,
    max_iter=5,
)

# ── Task 定义 ─────────────────────────────────────────────

research_task = Task(
    description="""对 {topic} 进行全面调研，要求：
    1. 搜索并整理该领域最新的 5-8 条重要信息/动态
    2. 收集关键数据和统计数字
    3. 记录信息来源以便后续引用
    4. 关注最近 6 个月内的最新发展""",
    expected_output="""结构化的调研素材文档，包含：
    - 关键信息点列表（附来源）
    - 重要数据和统计
    - 主要参与者/公司/项目
    - 时间线和里程碑事件""",
    agent=researcher,
)

analysis_task = Task(
    description="""基于调研素材，进行深度分析：
    1. 识别 3-5 个核心趋势
    2. 对每个趋势进行 SWOT 分析
    3. 评估各趋势的影响力和发展前景
    4. 给出数据支撑的预测""",
    expected_output="""分析报告，包含：
    - 核心趋势识别与排序
    - 每个趋势的详细分析（优势/劣势/机会/威胁）
    - 影响力评估矩阵
    - 未来 1-2 年发展预测""",
    agent=analyst,
    context=[research_task],
)

writing_task = Task(
    description="""基于分析结果，撰写一份完整的研究报告：
    1. 包含摘要、引言、正文（分章节）、结论、参考来源
    2. 正文按趋势分章节论述
    3. 每个章节包含数据引用和案例
    4. 总字数 2000-3000 字
    5. 使用 Markdown 格式""",
    expected_output="完整的 Markdown 格式研究报告",
    agent=writer,
    context=[research_task, analysis_task],
)

review_task = Task(
    description="""审校研究报告，检查以下维度：
    1. 事实准确性：数据和信息是否准确
    2. 逻辑一致性：论述是否前后一致
    3. 完整性：是否覆盖了所有关键趋势
    4. 可读性：语言是否流畅，结构是否清晰
    5. 如有问题，直接在报告中修改并标注修改原因
    6. 输出最终版本的完整报告""",
    expected_output="审校后的最终版研究报告（Markdown 格式）",
    agent=reviewer,
    context=[writing_task],
    output_file="output/research_report.md",
)

# ── Crew 编排 ─────────────────────────────────────────────

research_crew = Crew(
    agents=[researcher, analyst, writer, reviewer],
    tasks=[research_task, analysis_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True,
    memory=True,
)


# ── 执行 ──────────────────────────────────────────────────

def run_research(topic: str) -> str:
    """运行研究团队"""
    result = research_crew.kickoff(inputs={"topic": topic})
    print(f"\n{'='*60}")
    print(f"研究主题: {topic}")
    print(f"Token 使用: {result.token_usage}")
    print(f"报告已保存到: output/research_report.md")
    print(f"{'='*60}")
    return result.raw


if __name__ == "__main__":
    report = run_research("2026 年大语言模型 Agent 技术发展趋势")
    print(report[:500])
```

### 6.3 运行与调试技巧

**1. 调试模式**：设置 `verbose=True` 可以看到每个 Agent 的完整推理过程。

**2. 成本控制**：

```python
# 不同 Agent 使用不同模型，平衡成本和质量
researcher = Agent(
    role="调研员",
    goal="...",
    backstory="...",
    llm="gpt-4o-mini",  # 调研用小模型，降低成本
    tools=[search_tool],
)

writer = Agent(
    role="撰写者",
    goal="...",
    backstory="...",
    llm="gpt-4o",  # 写作用大模型，保证质量
)
```

**3. 错误处理**：

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    max_rpm=10,           # 限制每分钟请求数，避免 Rate Limit
    share_crew=False,     # 不共享匿名使用数据
)

try:
    result = crew.kickoff(inputs={"topic": "AI Agent"})
except Exception as e:
    print(f"执行失败: {e}")
    # 可以实现重试逻辑
```

### 6.4 AutoGen 版本对比

同样的研究团队用 AutoGen 实现，展示两种框架的风格差异：

```python
"""研究团队 —— AutoGen 版本"""

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    system_message="""你是研究调研员。收到研究主题后：
    1. 列出该领域 5 个最新趋势
    2. 为每个趋势提供简要说明和数据支撑
    完成后说"调研完成，请分析师接手"。""",
)

analyst = AssistantAgent(
    name="analyst",
    model_client=model_client,
    system_message="""你是数据分析师。收到调研素材后：
    1. 对每个趋势进行深度分析
    2. 评估影响力和发展前景
    完成后说"分析完成，请撰写者接手"。""",
)

writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="""你是报告撰写者。收到分析结果后：
    1. 撰写完整的 Markdown 格式研究报告
    2. 包含摘要、正文、结论
    完成后说"报告撰写完成，请审校者检查"。""",
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="""你是质量审校者。收到报告后：
    1. 检查准确性和完整性
    2. 如果合格，回复 APPROVE
    3. 如果需要修改，提出具体建议""",
)

termination = TextMentionTermination("APPROVE")

team = SelectorGroupChat(
    participants=[researcher, analyst, writer, reviewer],
    model_client=model_client,
    termination_condition=termination,
    max_turns=20,
)


async def main():
    result = await team.run(
        task="研究 2026 年大语言模型 Agent 技术发展趋势"
    )
    # 输出最终报告
    for msg in result.messages:
        if msg.source == "writer":
            print(msg.content)
            break


asyncio.run(main())
```

**两种实现的关键差异**：

| 维度 | CrewAI 版本 | AutoGen 版本 |
|------|------------|-------------|
| 编排方式 | 显式定义 Task 依赖链 | 通过对话自然流转 |
| 流程控制 | Process.sequential 严格顺序 | SelectorGroupChat 动态选择 |
| 输出管理 | Task 级别 output_file | 需从消息中提取 |
| 代码量 | 较多（显式定义） | 较少（对话驱动） |
| 可预测性 | 高（固定流程） | 中（动态路由） |

---

## 练习

### 练习 1：CrewAI 客服团队（基础）

用 CrewAI 构建一个客服团队，包含以下 Agent：
- **分诊 Agent**：判断用户问题类型（技术/账单/一般咨询）
- **技术支持 Agent**：处理技术问题
- **账单 Agent**：处理账单问题

要求：使用 `Process.hierarchical`，让分诊 Agent 作为管理者分配任务。

### 练习 2：AutoGen 代码审查（中级）

用 AutoGen 构建一个代码审查系统：
- **开发者 Agent**：编写代码
- **审查者 Agent**：审查代码质量、安全性
- **测试者 Agent**：编写并执行测试用例

要求：使用 `SelectorGroupChat`，让 Agent 之间自然对话直到代码通过审查。

### 练习 3：多框架对比实验（进阶）

选择一个具体任务（如"生成一份竞品分析报告"），分别用 CrewAI、AutoGen、OpenAI Agents SDK 实现，对比：
- 代码复杂度
- 执行时间
- 输出质量
- Token 消耗

将对比结果整理成表格，总结各框架的优劣势。

### 练习 4：带记忆的研究助手（进阶）

扩展本章的研究团队实战项目：
1. 启用 CrewAI 的长期记忆功能
2. 对同一主题运行两次，观察第二次执行是否利用了第一次的经验
3. 添加自定义工具（如数据库查询、API 调用）
4. 实现错误重试和降级策略

---

## 延伸阅读

### 官方文档

- [CrewAI 官方文档](https://docs.crewai.com/) —— 最权威的 CrewAI 使用指南，包含完整 API 参考
- [AutoGen (AG2) 官方文档](https://microsoft.github.io/autogen/) —— AutoGen v0.4 架构说明和教程
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) —— Agents SDK 快速入门和 API 参考

### 推荐教程

- [CrewAI Crash Course - Multi AI Agent Systems](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) —— DeepLearning.AI 出品的 CrewAI 短课程，Andrew Ng 推荐
- [AutoGen: Enabling Next-Gen LLM Applications](https://microsoft.github.io/autogen/docs/Getting-Started/) —— Microsoft 官方入门教程，覆盖从基础到高级用法
- [Building Agentic RAG with LlamaIndex & CrewAI](https://www.deeplearning.ai/short-courses/) —— 将 RAG 与多 Agent 结合的实战课程

### GitHub 仓库

- [crewAI](https://github.com/crewAIInc/crewAI) ⭐ 44,500+ —— CrewAI 源码，包含大量示例
- [autogen](https://github.com/microsoft/autogen) ⭐ 54,700+ —— AutoGen 源码，v0.4 分支为最新架构
- [openai-agents-python](https://github.com/openai/openai-agents-python) —— OpenAI Agents SDK 源码

### 深度文章

- [The Landscape of Multi-Agent Frameworks (2026)](https://blog.langchain.dev/) —— LangChain 博客对多 Agent 框架的全景分析
- [When to Use Multi-Agent Systems](https://www.anthropic.com/research) —— Anthropic 关于何时使用多 Agent 系统的研究报告
