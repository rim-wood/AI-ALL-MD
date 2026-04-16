# LangChain / LangGraph

> 最流行的 LLM 应用开发框架与工作流编排引擎

## 学习目标

- 掌握 LangChain v1.2+ 核心抽象与 LCEL（LangChain Expression Language）组合模式
- 理解 LangGraph v1.0+ 状态机架构与工作流编排
- 能够使用 LangSmith 进行链路追踪、评估与 Prompt 管理
- 构建一个完整的多步骤 Agent 并部署到生产环境

---

## 1. LangChain 核心概念

### 1.1 架构概览

LangChain 在 v1.0 之后进行了彻底的模块化重构，形成了清晰的四层架构：

```
┌─────────────────────────────────────────────────────┐
│              第三方集成包 (langchain-openai,          │
│         langchain-anthropic, langchain-aws ...)      │
├─────────────────────────────────────────────────────┤
│           langchain-community (社区集成)              │
│         数百个第三方服务的连接器与适配器                 │
├─────────────────────────────────────────────────────┤
│               langchain (高级链与 Agent)              │
│          预构建的 Chain、Agent、检索策略                │
├─────────────────────────────────────────────────────┤
│             langchain-core (核心抽象)                 │
│    Runnable 接口、LCEL、BaseMessage、BaseTool ...     │
└─────────────────────────────────────────────────────┘
```

| 包名 | 职责 | 安装 | 更新频率 |
|------|------|------|----------|
| `langchain-core` | 核心抽象、LCEL、Runnable 接口 | `pip install langchain-core` | 稳定，少量更新 |
| `langchain` | 高级 Chain、Agent、检索策略 | `pip install langchain` | 中等频率 |
| `langchain-community` | 社区贡献的集成 | `pip install langchain-community` | 频繁更新 |
| `langchain-openai` 等 | 官方维护的第三方集成 | `pip install langchain-openai` | 跟随上游 API |

> **设计原则**：`langchain-core` 零外部依赖（仅依赖 `pydantic`），确保核心抽象的稳定性。具体的模型提供商、向量数据库等集成通过独立包安装，避免依赖膨胀。

安装推荐：

```bash
# 最小安装 — 仅核心 + OpenAI
pip install langchain-core langchain-openai

# 完整安装 — 包含高级功能
pip install langchain langchain-openai langchain-anthropic

# LangGraph 安装
pip install langgraph langgraph-checkpoint-postgres
```

### 1.2 核心抽象

LangChain 围绕以下核心抽象构建，所有抽象都实现了统一的 `Runnable` 接口：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ PromptTemplate│───▶│  ChatModel   │───▶│ OutputParser  │
│  构造提示词    │    │  调用 LLM    │    │  解析输出     │
└──────────────┘    └──────────────┘    └──────────────┘
                           │
                    ┌──────┴──────┐
                    │    Tools    │
                    │  外部工具    │
                    └─────────────┘
```

**核心抽象一览：**

| 抽象 | 基类 | 职责 | 示例 |
|------|------|------|------|
| ChatModel | `BaseChatModel` | 封装 LLM API 调用 | `ChatOpenAI`, `ChatAnthropic` |
| PromptTemplate | `BasePromptTemplate` | 构造和格式化提示词 | `ChatPromptTemplate` |
| OutputParser | `BaseOutputParser` | 解析模型输出为结构化数据 | `PydanticOutputParser` |
| Tool | `BaseTool` | 封装外部工具调用 | `@tool` 装饰器定义 |
| Retriever | `BaseRetriever` | 从数据源检索相关文档 | `VectorStoreRetriever` |
| Memory | `BaseMemory` | 管理对话历史（已逐步迁移至 LangGraph） | `ConversationBufferMemory` |

> **注意**：传统的 `Memory` 抽象已逐步被 LangGraph 的状态管理取代。新项目建议直接使用 LangGraph 管理对话状态，详见 [第 3 节](#3-langgraph)。

### 1.3 LCEL（LangChain Expression Language）

LCEL 是 LangChain v1.0+ 的核心组合模式，使用管道操作符 `|` 将多个 `Runnable` 组件串联成链。它取代了早期的 `LLMChain`、`SequentialChain` 等显式 Chain 类。

**LCEL 的核心优势：**

- **统一接口**：所有组件实现 `Runnable`，支持 `invoke`、`stream`、`batch`、`ainvoke` 等方法
- **自动流式**：管道中的流式传输自动传播，无需额外配置
- **原生异步**：每个 `Runnable` 同时提供同步和异步接口
- **自动并行**：`RunnableParallel` 自动并行执行独立分支
- **可观测性**：与 LangSmith 深度集成，自动记录每一步的输入输出

**基础示例 — 构建一个翻译链：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义组件
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，将用户输入翻译成{target_language}。"),
    ("human", "{text}")
])
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# 使用 | 操作符组合成链
chain = prompt | model | parser

# 调用
result = chain.invoke({
    "target_language": "英文",
    "text": "LangChain 是最流行的 LLM 应用开发框架"
})
print(result)
# Output: LangChain is the most popular LLM application development framework
```

**Runnable 接口的统一方法：**

```python
# 同步调用
result = chain.invoke({"text": "你好"})

# 流式输出
for chunk in chain.stream({"text": "你好"}):
    print(chunk, end="", flush=True)

# 批量调用
results = chain.batch([
    {"text": "你好", "target_language": "英文"},
    {"text": "Hello", "target_language": "日文"},
])

# 异步调用
import asyncio
result = asyncio.run(chain.ainvoke({"text": "你好"}))
```

**RunnableParallel — 并行执行：**

```python
from langchain_core.runnables import RunnableParallel

# 同时生成摘要和关键词
summarize_chain = summary_prompt | model | parser
keywords_chain = keywords_prompt | model | parser

parallel_chain = RunnableParallel(
    summary=summarize_chain,
    keywords=keywords_chain,
)

# 两个链并行执行，结果合并为字典
result = parallel_chain.invoke({"text": "一篇很长的文章..."})
# {"summary": "...", "keywords": "..."}
```

**RunnablePassthrough 与 RunnableLambda：**

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# RunnablePassthrough — 透传输入，常用于 RAG
chain = RunnableParallel(
    context=retriever,                    # 检索相关文档
    question=RunnablePassthrough(),       # 透传用户问题
) | prompt | model | parser

# RunnableLambda — 将普通函数包装为 Runnable
def word_count(text: str) -> dict:
    return {"text": text, "word_count": len(text.split())}

chain = RunnableLambda(word_count) | some_next_step
```

---

## 2. 常用组件

### 2.1 Chat Models

LangChain 为所有主流 LLM 提供统一的 `BaseChatModel` 接口，切换模型只需更换一行代码：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
gpt = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Anthropic
claude = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)

# 统一接口 — 两者用法完全一致
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个有帮助的助手。"),
    HumanMessage(content="什么是 RAG？"),
]

response = gpt.invoke(messages)
print(response.content)
```

**流式调用：**

```python
for chunk in gpt.stream(messages):
    print(chunk.content, end="", flush=True)
```

**绑定工具（Function Calling）：**

关于 Function Calling 的详细原理，请参考 [Function Calling 章节](../02-core-tech/03-function-calling.md)。

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """获取指定城市的天气信息"""
    city: str = Field(description="城市名称")

# 将工具绑定到模型
model_with_tools = gpt.bind_tools([GetWeather])
response = model_with_tools.invoke("北京今天天气怎么样？")
print(response.tool_calls)
# [{'name': 'GetWeather', 'args': {'city': '北京'}, 'id': 'call_xxx'}]
```

### 2.2 Prompt Templates

`ChatPromptTemplate` 是构造多轮对话提示词的核心工具：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 基础模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，请用{style}的风格回答问题。"),
    ("human", "{question}"),
])

# 带对话历史的模板
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 动态插入历史消息
    ("human", "{question}"),
])

# 使用
from langchain_core.messages import HumanMessage, AIMessage

result = prompt_with_history.invoke({
    "chat_history": [
        HumanMessage(content="我叫小明"),
        AIMessage(content="你好小明！有什么可以帮你的？"),
    ],
    "question": "你还记得我的名字吗？",
})
```

### 2.3 Output Parsers

Output Parser 将模型的文本输出解析为结构化数据。在 LangChain v1.2+ 中，推荐优先使用模型原生的 **Structured Output**（`with_structured_output`），仅在模型不支持时回退到 Output Parser。

**方式一：`with_structured_output`（推荐）**

```python
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分 1-10")
    summary: str = Field(description="一句话总结")

structured_model = gpt.with_structured_output(MovieReview)
review = structured_model.invoke("评价一下电影《星际穿越》")
print(review.title)    # 星际穿越
print(review.rating)   # 9.2
print(review.summary)  # 一部关于爱与时间的科幻史诗
```

**方式二：PydanticOutputParser（兼容方案）**

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "按照以下格式输出：\n{format_instructions}"),
    ("human", "{query}"),
])

chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | model | parser

review = chain.invoke({"query": "评价一下电影《星际穿越》"})
```

### 2.4 Tools & Toolkits

工具是 Agent 与外部世界交互的桥梁。LangChain 提供 `@tool` 装饰器快速定义工具：

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息。

    Args:
        query: 搜索关键词
    """
    # 实际实现中调用搜索 API
    return f"搜索结果：关于 '{query}' 的最新信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 数学表达式，如 '2 + 3 * 4'
    """
    try:
        result = eval(expression)  # 生产环境应使用安全的表达式解析器
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# 查看工具信息
print(search_web.name)         # search_web
print(search_web.description)  # 搜索互联网获取最新信息。
print(search_web.args_schema.model_json_schema())
```

**使用 Pydantic 定义复杂工具参数：**

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大返回结果数")
    language: str = Field(default="zh", description="搜索语言")

@tool(args_schema=SearchParams)
def advanced_search(query: str, max_results: int = 5, language: str = "zh") -> str:
    """高级搜索，支持指定结果数量和语言。"""
    return f"搜索 '{query}'，返回 {max_results} 条 {language} 结果"
```

### 2.5 Retrievers

Retriever 是 RAG 流程的核心组件，负责从数据源检索相关文档。关于 RAG 的完整介绍请参考 [RAG 章节](../02-core-tech/04-rag.md) 和 [Embedding 与向量搜索章节](../02-core-tech/05-embeddings-and-vector-search.md)。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 创建向量存储
docs = [
    Document(page_content="LangChain 是一个 LLM 应用开发框架", metadata={"source": "docs"}),
    Document(page_content="LangGraph 用于构建有状态的工作流", metadata={"source": "docs"}),
    Document(page_content="RAG 通过检索增强生成质量", metadata={"source": "blog"}),
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# 转换为 Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 在 LCEL 中使用
results = retriever.invoke("什么是 LangGraph？")
for doc in results:
    print(doc.page_content)
```

**构建完整的 RAG 链：**

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题。如果上下文中没有相关信息，请说明。\n\n上下文：\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | rag_prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("LangGraph 有什么用？")
print(answer)
```

---

## 3. LangGraph

### 3.1 为什么需要 LangGraph

LangChain 的 LCEL 非常适合构建线性的处理管道（A → B → C），但现实中的 AI 应用往往需要：

- **循环**：Agent 反复调用工具直到获得满意结果
- **条件分支**：根据模型输出决定下一步操作
- **状态管理**：跨多个步骤维护和更新共享状态
- **人工干预**：在关键节点暂停等待人工审批
- **错误恢复**：从失败点恢复而非从头开始

```
LCEL（线性管道）：          LangGraph（状态图）：

A → B → C → D              A → B ──→ C
                                ↑     │
                                └─ D ←┘  （循环 + 条件分支）
```

**自主性 vs 可靠性的权衡（Autonomy vs Reliability Tradeoff）**

在构建 AI 应用时，存在一个核心的设计光谱：

```
可靠但僵化                                          强大但混乱
◄──────────────────────────────────────────────────────►
  Router          Chain          Agent          Autonomous
 (规则路由)      (固定流程)     (工具调用循环)      Agent
                                              (完全自主决策)
```

- **光谱左端 — Router（路由器）**：完全基于规则的分支逻辑，100% 可预测，但无法处理意外情况。例如根据用户意图关键词将请求分发到不同的处理链。
- **光谱中间 — Chain / Agent**：模型拥有一定的决策权（选择工具、决定循环次数），但在预定义的框架内运行。
- **光谱右端 — Autonomous Agent（自主 Agent）**：模型完全自主决定下一步行动，能力强大但行为难以预测，可能陷入无限循环、产生高额 API 费用，或执行危险操作。

传统方法迫使开发者在这两端之间做取舍：要么牺牲灵活性换取可靠性，要么接受不可预测性来获得更强的能力。

**LangGraph 的核心价值在于"弯曲这条曲线"** — 通过以下机制，让 Agent 在拥有更多自主权的同时保持可靠性：

- **结构化决策（Structured Decision-Making）**：通过 `StateGraph` 定义明确的状态转移规则，Agent 的每一步决策都在图结构约束内进行
- **状态持久化（State Preservation）**：Checkpointing 机制确保每一步状态都被保存，任何时刻都可以回溯和审计
- **优雅恢复（Graceful Recovery）**：从失败节点恢复执行，而非从头开始；支持 Human-in-the-Loop 在关键节点介入纠偏
- **可控循环**：通过条件边和最大迭代次数限制，防止 Agent 陷入无限循环

简而言之，LangGraph 不是让你在 Router 和 Autonomous Agent 之间二选一，而是让你精确控制每个节点的自主程度 — 某些节点完全确定性执行，某些节点允许模型自由决策，关键节点要求人工审批。

> **重要变更**：LangChain 的 `AgentExecutor` 已被标记为废弃（EOL 2026 年 12 月），所有新的 Agent 开发应使用 LangGraph。LangGraph 提供了更细粒度的控制、更好的可观测性和生产级的可靠性。

### 3.2 核心概念

LangGraph 基于**有向图**模型，核心概念包括：

| 概念 | 说明 | 类比 |
|------|------|------|
| **StateGraph** | 有状态的有向图，定义整个工作流 | 状态机 |
| **State** | 图的共享状态，所有节点可读写 | 全局变量 |
| **Node** | 图中的处理节点，执行具体逻辑 | 函数 |
| **Edge** | 节点之间的连接，定义执行顺序 | 箭头 |
| **Conditional Edge** | 根据状态动态选择下一个节点 | if/else |
| **START / END** | 特殊节点，标记图的入口和出口 | 入口/出口 |

**最小示例 — 构建一个简单的对话图：**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# 1. 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages 自动追加消息

# 2. 定义节点
model = ChatOpenAI(model="gpt-4o")

def chatbot(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 3. 构建图
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 4. 编译并运行
app = graph.compile()

result = app.invoke({
    "messages": [{"role": "user", "content": "什么是 LangGraph？"}]
})
print(result["messages"][-1].content)
```

**带工具调用的 Agent 图：**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    weather_data = {"北京": "晴天 25°C", "上海": "多云 22°C"}
    return weather_data.get(city, f"{city}：暂无数据")

# 绑定工具到模型
tools = [get_weather]
model_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

# 条件边：如果模型调用了工具，转到 tools 节点；否则结束
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # 自动判断是否有 tool_calls
graph.add_edge("tools", "agent")  # 工具执行完毕后回到 agent

app = graph.compile()

result = app.invoke({
    "messages": [{"role": "user", "content": "北京和上海今天天气怎么样？"}]
})
print(result["messages"][-1].content)
```

上述 Agent 的执行流程如下：

```
         ┌──────────┐
         │  START    │
         └────┬─────┘
              ▼
         ┌──────────┐
    ┌───▶│  agent   │───── 无工具调用 ────▶ END
    │    └────┬─────┘
    │         │ 有工具调用
    │         ▼
    │    ┌──────────┐
    └────│  tools   │
         └──────────┘
```

### 3.3 状态管理

LangGraph 的状态是图中所有节点共享的数据结构。每个节点接收当前状态，返回状态更新。

**使用 TypedDict 定义状态：**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，自动追加
    search_results: list[str]               # 每次覆盖
    step_count: Annotated[int, add]         # 使用 add 累加
    final_answer: str                       # 每次覆盖
```

**Annotated 与 Reducer 函数：**

`Annotated[type, reducer]` 中的 reducer 决定了状态字段的更新方式：

| Reducer | 行为 | 适用场景 |
|---------|------|----------|
| `add_messages` | 智能追加消息（支持按 ID 更新） | 对话历史 |
| `operator.add` | 列表拼接或数值累加 | 计数器、日志收集 |
| 无 Reducer | 直接覆盖 | 最终结果、配置项 |

**使用 Pydantic 定义状态（带验证）：**

```python
from pydantic import BaseModel, Field

class ResearchState(BaseModel):
    query: str = Field(description="研究问题")
    sources: list[str] = Field(default_factory=list, description="信息来源")
    draft: str = Field(default="", description="草稿内容")
    review_passed: bool = Field(default=False, description="是否通过审核")
    revision_count: int = Field(default=0, description="修改次数")
```

### 3.4 检查点与持久化

检查点（Checkpointing）是 LangGraph 的核心能力之一，它在每个节点执行后自动保存状态快照，支持：

- **断点恢复**：从任意节点重新执行
- **对话持久化**：跨请求保持对话状态
- **时间旅行**：回溯到历史状态

**MemorySaver（开发环境）：**

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 使用 thread_id 标识对话
config = {"configurable": {"thread_id": "user-123"}}

# 第一轮对话
result1 = app.invoke(
    {"messages": [{"role": "user", "content": "我叫小明"}]},
    config=config,
)

# 第二轮对话 — 自动恢复上下文
result2 = app.invoke(
    {"messages": [{"role": "user", "content": "你还记得我的名字吗？"}]},
    config=config,
)
print(result2["messages"][-1].content)  # 你叫小明！
```

**PostgresSaver（生产环境）：**

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/langgraph"

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()  # 自动创建表
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "session-456"}}
        result = await app.ainvoke(
            {"messages": [{"role": "user", "content": "你好"}]},
            config=config,
        )
```

### 3.5 Human-in-the-loop

在生产环境中，某些关键操作需要人工审批。LangGraph 通过 `interrupt_before` / `interrupt_after` 实现中断：

```python
from langgraph.graph import StateGraph, START, END

class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order_details: dict
    approved: bool

def collect_order(state: OrderState) -> dict:
    """收集订单信息"""
    return {"order_details": {"item": "GPU 服务器", "price": 50000}}

def execute_order(state: OrderState) -> dict:
    """执行订单"""
    return {"messages": [{"role": "assistant", "content": "订单已提交！"}]}

graph = StateGraph(OrderState)
graph.add_node("collect", collect_order)
graph.add_node("execute", execute_order)
graph.add_edge(START, "collect")
graph.add_edge("collect", "execute")
graph.add_edge("execute", END)

# 在 execute 节点之前中断，等待人工审批
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"],
)

config = {"configurable": {"thread_id": "order-789"}}

# 第一次调用 — 在 execute 之前暂停
result = app.invoke(
    {"messages": [{"role": "user", "content": "我要买一台 GPU 服务器"}]},
    config=config,
)
print("订单详情:", result["order_details"])
print("等待人工审批...")

# 人工审批后，使用 None 恢复执行
# （在实际应用中，这里会等待人工通过 API 确认）
result = app.invoke(None, config=config)
print(result["messages"][-1].content)  # 订单已提交！
```

**中断模式对比：**

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `interrupt_before=["node"]` | 在节点执行**之前**中断 | 审批、确认操作 |
| `interrupt_after=["node"]` | 在节点执行**之后**中断 | 查看中间结果、人工修正 |

### 3.6 从 AgentExecutor 迁移到 LangGraph

`AgentExecutor` 将于 **2026 年 12 月正式停止维护（EOL）**。如果你的项目仍在使用 `initialize_agent()` 或 `AgentExecutor`，现在是迁移的最佳时机。

**旧模式 — AgentExecutor（已废弃）：**

```python
# ❌ 旧模式：隐式状态管理，黑盒执行
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[search_web, calculate],
    llm=ChatOpenAI(model="gpt-4o"),
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# AgentExecutor 内部使用隐藏的 scratchpad 字符串管理中间状态
# 开发者无法控制循环逻辑、无法插入中间步骤、难以调试
result = agent.run("搜索 LangGraph 最新版本并计算它的版本号乘以 2")
```

`AgentExecutor` 的核心问题：

- **隐式 Scratchpad**：中间推理过程以字符串形式隐藏在内部，开发者无法直接访问或修改
- **黑盒循环**：工具调用的循环逻辑不透明，难以添加自定义控制（如最大重试次数、超时、人工审批）
- **调试困难**：出错时只能看到最终结果，无法定位是哪一步出了问题

**新模式一 — `create_react_agent()`（简单场景）：**

```python
# ✅ 新模式（简单）：一行代码创建 ReAct Agent
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search_web, calculate],
)

# 状态是显式的 TypedDict，每一步都可追踪
result = agent.invoke({
    "messages": [{"role": "user", "content": "搜索 LangGraph 最新版本"}]
})
```

**新模式二 — `StateGraph`（自定义编排）：**

```python
# ✅ 新模式（高级）：完全自定义的状态图
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 显式定义状态 Schema — 取代隐藏的 scratchpad
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int       # 可以追踪工具调用次数
    should_escalate: bool      # 可以添加自定义控制字段

def agent_node(state: AgentState) -> dict:
    # 完全控制 Agent 逻辑
    if state.get("tool_call_count", 0) >= 5:
        return {"should_escalate": True}  # 超过 5 次工具调用，升级到人工
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response], "tool_call_count": state.get("tool_call_count", 0) + 1}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
# ... 添加边和条件逻辑
```

**迁移核心变化总结：**

| 维度 | AgentExecutor（旧） | LangGraph（新） |
|------|---------------------|-----------------|
| 状态管理 | 隐藏的 scratchpad 字符串 | 显式 `TypedDict` Schema |
| 循环控制 | `max_iterations` 参数 | 条件边 + 自定义路由函数 |
| 人工干预 | 不支持 | `interrupt_before` / `interrupt_after` |
| 持久化 | 不支持 | Checkpointer（Memory / Postgres） |
| 可观测性 | `verbose=True` 打印日志 | LangSmith 全链路追踪 |
| 错误恢复 | 从头重试 | 从失败节点恢复 |

### 3.7 LangGraph vs 其他框架

选择 Agent 框架时，没有"最好"的框架，只有最适合场景的框架：

| 框架 | 最佳场景 | 核心优势 | 局限性 |
|------|----------|----------|--------|
| **LangGraph** | 复杂编排、生产级 Agent | Human-in-the-Loop、Time-Travel Debugging、状态持久化、细粒度控制 | 学习曲线较陡，简单场景略显繁重 |
| **PydanticAI** | 简单的类型安全 Agent | 极简 API、原生 Pydantic 集成、类型推断优秀 | 不支持复杂工作流编排和状态持久化 |
| **OpenAI Agents SDK** | 快速原型、OpenAI 生态 | 托管状态管理、部署简单、与 OpenAI 模型深度集成 | 锁定 OpenAI 生态，自定义能力有限 |

**选型建议：**

- 如果你的 Agent 只需要简单的工具调用循环，且重视类型安全 → **PydanticAI**
- 如果你主要使用 OpenAI 模型，且希望最快上线 → **OpenAI Agents SDK**
- 如果你需要复杂的多步骤工作流、Human-in-the-Loop 审批、从故障中恢复、或多 Agent 协作 → **LangGraph**

---

## 4. LangSmith

LangSmith 是 LangChain 官方的可观测性与评估平台，为 LLM 应用提供全链路追踪、自动化评估和 Prompt 管理能力。

### 4.1 链路追踪

**配置 LangSmith：**

```bash
export LANGSMITH_API_KEY="lsv2_pt_xxxxx"
export LANGSMITH_PROJECT="my-ai-app"
export LANGSMITH_TRACING=true
```

配置环境变量后，所有 LangChain/LangGraph 调用自动上报 Trace，无需修改代码。

**Trace 可视化：**

LangSmith 的 Trace 视图展示每次调用的完整执行链路：

```
Trace: "用户问天气"
├── ChatPromptTemplate  (0.2ms)
│   ├── Input: {"city": "北京"}
│   └── Output: [SystemMessage, HumanMessage]
├── ChatOpenAI  (850ms)  ← 延迟瓶颈
│   ├── Input: 2 messages
│   ├── Output: AIMessage with tool_calls
│   └── Token usage: 150 input, 30 output
├── ToolNode: get_weather  (120ms)
│   ├── Input: {"city": "北京"}
│   └── Output: "晴天 25°C"
└── ChatOpenAI  (620ms)
    ├── Input: 4 messages
    └── Output: "北京今天晴天，气温25°C..."
```

**手动添加 Trace（非 LangChain 代码）：**

```python
from langsmith import traceable

@traceable(name="my_custom_function")
def process_data(data: str) -> str:
    # 自定义处理逻辑
    result = data.upper()
    return result

# 调用时自动记录到 LangSmith
process_data("hello world")
```

### 4.2 评估

LangSmith 提供系统化的评估框架，用于衡量 LLM 应用的质量。关于更完整的评估方法论，请参考 [LLMOps 章节](../05-production/01-llmops.md)。

**创建评估数据集并运行评估：**

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 创建数据集
dataset = client.create_dataset("qa-eval-set")
client.create_examples(
    inputs=[
        {"question": "什么是 RAG？"},
        {"question": "LangGraph 和 LangChain 的关系？"},
    ],
    outputs=[
        {"answer": "RAG 是检索增强生成，通过检索外部知识来提升生成质量"},
        {"answer": "LangGraph 是 LangChain 生态中的工作流编排引擎"},
    ],
    dataset_id=dataset.id,
)

# 定义目标函数（被评估的应用）
def my_app(inputs: dict) -> dict:
    result = rag_chain.invoke(inputs["question"])
    return {"answer": result}

# 定义评估器
def correctness(outputs: dict, reference_outputs: dict) -> dict:
    """简单的关键词匹配评估器"""
    pred = outputs["answer"].lower()
    ref = reference_outputs["answer"].lower()
    # 检查参考答案中的关键词是否出现在预测中
    keywords = ref.split("，")
    matches = sum(1 for kw in keywords if kw in pred)
    score = matches / len(keywords) if keywords else 0
    return {"key": "correctness", "score": score}

# 运行评估
results = evaluate(
    my_app,
    data="qa-eval-set",
    evaluators=[correctness],
    experiment_prefix="rag-v1",
)
```

**使用 LLM 作为评估器（LLM-as-Judge）：**

```python
from langsmith.evaluation import LangChainStringEvaluator

# 内置的 LLM 评估器
qa_evaluator = LangChainStringEvaluator(
    "qa",
    config={"llm": ChatOpenAI(model="gpt-4o", temperature=0)},
)

results = evaluate(
    my_app,
    data="qa-eval-set",
    evaluators=[qa_evaluator],
    experiment_prefix="rag-v1-llm-judge",
)
```

### 4.3 Prompt Hub

LangSmith Prompt Hub 提供 Prompt 的版本管理和团队协作：

```python
from langsmith import Client

client = Client()

# 推送 Prompt 到 Hub
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}助手，请简洁准确地回答问题。"),
    ("human", "{question}"),
])

client.push_prompt("my-org/qa-prompt", object=prompt)

# 从 Hub 拉取 Prompt
pulled_prompt = client.pull_prompt("my-org/qa-prompt")

# 使用特定版本
pulled_prompt_v2 = client.pull_prompt("my-org/qa-prompt:v2")
```

**Prompt Hub 的核心价值：**

- **版本控制**：每次修改自动生成版本，支持回滚
- **A/B 测试**：不同版本的 Prompt 可以在评估中对比效果
- **团队协作**：产品经理可以直接在 Web UI 中编辑 Prompt，无需修改代码
- **与代码解耦**：Prompt 变更不需要重新部署应用

### 4.4 Time-Travel Debugging

LangGraph 与 LangSmith 的深度集成带来了一项强大的调试能力：**Time-Travel Debugging（时间旅行调试）**。

**核心原理：**

LangGraph 的 Checkpointer 在每个节点执行后自动保存完整的状态快照。LangSmith 将这些快照可视化，开发者可以像"回放录像"一样逐步查看图的执行过程。

```
执行时间线：
  t0          t1          t2          t3          t4
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
START → [research] → [write] → [review] → [research] → ...
  📸          📸          📸          📸          📸
 快照0       快照1       快照2       快照3       快照4

Time-Travel：可以跳回任意快照，从该点重新执行
```

**实际应用场景：**

1. **调试失败的执行**：Agent 在第 5 步产生了错误结果？跳回第 4 步的状态快照，修改输入后重新执行，无需从头开始。

2. **从崩溃中恢复**：使用 Postgres Checkpointer 时，即使服务器崩溃，所有状态都已持久化。重启后从最后一个检查点继续执行。

3. **Human-in-the-Loop 长等待**：Agent 在审批节点中断后，人工审批可能需要数小时甚至数天。Checkpointer 保存完整状态，审批完成后随时恢复执行。

**在 LangSmith 中使用 Time-Travel：**

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "debug-session-1"}}

    # 正常执行
    result = await app.ainvoke(input_data, config=config)

    # 获取执行历史 — 每个节点的状态快照
    history = [state async for state in app.aget_state_history(config)]
    for state in history:
        print(f"节点: {state.next}, 状态: {state.values.keys()}")

    # 回溯到特定检查点并重新执行
    target_state = history[2]  # 跳回第 3 个检查点
    replay_config = {"configurable": {"thread_id": "debug-session-1", "checkpoint_id": target_state.config["configurable"]["checkpoint_id"]}}
    result = await app.ainvoke(None, config=replay_config)
```

在 LangSmith Web UI 中，这一切都可以通过可视化界面完成：点击 Trace 中的任意节点，查看该节点的输入/输出状态，然后选择"Replay from here"从该点重新执行。

---

## 5. 实战：多步骤 Research Agent

### 5.1 架构设计

我们将构建一个**研究助手 Agent**，它能够：
1. 接收用户的研究问题
2. 搜索相关信息
3. 撰写研究报告草稿
4. 自我审核并决定是否需要补充搜索
5. 输出最终报告

**状态图设计：**

```
                ┌──────────┐
                │  START   │
                └────┬─────┘
                     ▼
              ┌─────────────┐
              │   research  │ ← 搜索信息
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │    write    │ ← 撰写草稿
              └──────┬──────┘
                     ▼
              ┌─────────────┐
         ┌────│   review    │ ← 自我审核
         │    └──────┬──────┘
         │           │
    需要补充    ▼ 审核通过
         │    ┌─────────────┐
         │    │    END      │
         │    └─────────────┘
         │           ▲
         └───────────┘
           (回到 research)
```

### 5.2 代码实现

```python
"""Research Agent — 基于 LangGraph 的多步骤研究助手"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# ── 工具定义 ──

@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。"""
    # 生产环境中替换为真实搜索 API（如 Tavily、SerpAPI）
    return f"[搜索结果] 关于 '{query}' 的信息：这是一个重要的技术话题..."

# ── 状态定义 ──

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str                  # 研究问题
    search_results: list[str]   # 搜索结果
    draft: str                  # 报告草稿
    review_feedback: str        # 审核反馈
    revision_count: int         # 修改次数
    is_approved: bool           # 是否通过审核

# ── 节点定义 ──

model = ChatOpenAI(model="gpt-4o", temperature=0.3)

def research_node(state: ResearchState) -> dict:
    """搜索相关信息"""
    query = state["query"]
    feedback = state.get("review_feedback", "")

    search_query = query
    if feedback:
        search_query = f"{query} — 补充搜索：{feedback}"

    result = web_search.invoke({"query": search_query})
    current_results = state.get("search_results", [])
    return {"search_results": current_results + [result]}

def write_node(state: ResearchState) -> dict:
    """撰写研究报告"""
    context = "\n".join(state["search_results"])
    prompt_messages = [
        SystemMessage(content="你是一个专业的研究分析师。根据搜索结果撰写简洁的研究报告。"),
        HumanMessage(content=f"研究问题：{state['query']}\n\n搜索结果：\n{context}\n\n请撰写研究报告："),
    ]
    response = model.invoke(prompt_messages)
    return {"draft": response.content}

def review_node(state: ResearchState) -> dict:
    """审核报告质量"""
    prompt_messages = [
        SystemMessage(content=(
            "你是一个严格的审核员。评估以下研究报告的质量。\n"
            "如果报告质量足够好，回复 'APPROVED'。\n"
            "如果需要补充信息，回复 'REVISE: ' 加上需要补充的方向。\n"
            "最多允许修改 2 次。"
        )),
        HumanMessage(content=f"研究问题：{state['query']}\n\n报告：\n{state['draft']}"),
    ]
    response = model.invoke(prompt_messages)
    content = response.content

    revision_count = state.get("revision_count", 0) + 1
    is_approved = content.startswith("APPROVED") or revision_count >= 3

    return {
        "review_feedback": content,
        "revision_count": revision_count,
        "is_approved": is_approved,
    }

# ── 路由函数 ──

def should_continue(state: ResearchState) -> str:
    """决定是继续修改还是结束"""
    if state.get("is_approved", False):
        return "end"
    return "research"

# ── 构建图 ──

graph = StateGraph(ResearchState)

# 添加节点
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_node("review", review_node)

# 添加边
graph.add_edge(START, "research")
graph.add_edge("research", "write")
graph.add_edge("write", "review")
graph.add_conditional_edges(
    "review",
    should_continue,
    {"research": "research", "end": END},
)

# 编译
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# ── 运行 ──

async def main():
    config = {"configurable": {"thread_id": "research-001"}}

    result = await app.ainvoke(
        {
            "query": "2026 年 AI Agent 技术的最新发展趋势",
            "messages": [{"role": "user", "content": "请研究 AI Agent 最新趋势"}],
            "search_results": [],
            "draft": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
        },
        config=config,
    )

    print("=" * 60)
    print("最终报告：")
    print(result["draft"])
    print(f"\n修改次数：{result['revision_count']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 5.3 部署

**方式一：使用 LangServe 快速部署 API**

```python
# server.py
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="Research Agent API")

# 将 LangGraph 应用暴露为 REST API
add_routes(app, research_app, path="/research")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# 启动服务
pip install langserve[all]
python server.py

# 调用 API
curl -X POST http://localhost:8000/research/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "AI Agent 趋势", "messages": [], "search_results": [], "draft": "", "review_feedback": "", "revision_count": 0, "is_approved": false}}'
```

**方式二：LangGraph Cloud（推荐生产环境）**

LangGraph Cloud 是官方的托管部署平台，专为有状态的 LangGraph 应用设计，解决了自行部署时的核心痛点：

| 能力 | 说明 |
|------|------|
| **自动扩缩容** | 根据请求量自动调整实例数，支持缩容到零 |
| **持久化存储** | 内置 PostgreSQL Checkpointer，无需自行管理数据库 |
| **长时间运行** | 支持运行数小时甚至数天的工作流（如等待人工审批），不受 HTTP 超时限制 |
| **Cron 任务** | 定时触发工作流，适合定期数据处理、报告生成 |
| **双重部署模式** | 支持 Cloud SaaS 和 Self-Hosted（BYOC）两种模式 |
| **LangSmith 集成** | 所有执行自动上报 Trace，支持 Time-Travel Debugging |
| **按量计费** | 基于实际使用量（节点执行次数）计费，无固定月费 |

部署步骤：

```yaml
# langgraph.json — 项目配置文件
{
  "dependencies": ["."],
  "graphs": {
    "research_agent": "./agent.py:app"
  },
  "env": ".env"
}
```

```bash
# 使用 LangGraph CLI 部署
pip install langgraph-cli
langgraph up        # 本地测试
langgraph deploy    # 部署到 LangGraph Cloud
```

**LangGraph Cloud API 调用：**

```python
from langgraph_sdk import get_client

client = get_client(url="https://your-deployment.langgraph.app")

# 创建线程
thread = await client.threads.create()

# 运行 Agent
run = await client.runs.create(
    thread["thread_id"],
    assistant_id="research_agent",
    input={
        "query": "AI Agent 趋势",
        "messages": [],
        "search_results": [],
        "draft": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
    },
)

# 流式获取结果
async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="research_agent",
    input=None,
    stream_mode="updates",
):
    print(event)
```

---

## 练习

### 练习 1：LCEL 基础（⭐）

构建一个 LCEL 链，实现以下功能：
- 接收一段文本
- 并行生成：① 中文摘要 ② 英文翻译 ③ 3 个关键词
- 将三个结果合并为一个字典输出

**提示**：使用 `RunnableParallel` 实现并行执行。

### 练习 2：LangGraph 工具 Agent（⭐⭐）

使用 LangGraph 构建一个具备以下工具的 Agent：
- `calculator`：计算数学表达式
- `unit_converter`：单位转换（如公里 ↔ 英里）
- `current_time`：获取当前时间

要求：
1. 使用 `StateGraph` 构建
2. 支持多轮对话（使用 `MemorySaver`）
3. Agent 能够根据问题自动选择合适的工具

### 练习 3：Human-in-the-loop 审批流（⭐⭐⭐）

构建一个"内容发布审批"工作流：
1. 用户提交文章主题
2. Agent 自动生成文章
3. 在发布前中断，等待人工审批
4. 人工可以选择"通过"或"修改"（附带修改意见）
5. 如果需要修改，Agent 根据意见重新生成

**提示**：使用 `interrupt_before` 和检查点恢复。

### 练习 4：RAG + LangGraph（⭐⭐⭐）

将 [RAG 章节](../02-core-tech/04-rag.md) 中的检索增强生成与 LangGraph 结合：
1. 用户提问
2. 检索相关文档
3. 生成回答
4. 自动评估回答质量（是否基于检索到的文档）
5. 如果质量不达标，重新检索（使用不同的查询策略）

---

## 延伸阅读

### 官方文档

- [LangChain Python 文档](https://python.langchain.com/docs/introduction/) — 最权威的 API 参考和教程
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 状态图、检查点、部署的完整指南
- [LangSmith 文档](https://docs.smith.langchain.com/) — 追踪、评估、Prompt Hub 使用指南
- [LCEL 概念指南](https://python.langchain.com/docs/concepts/lcel/) — 深入理解 Runnable 接口和管道组合

### 推荐教程

- [LangGraph Academy](https://academy.langchain.com/courses/intro-to-langgraph) — LangChain 官方出品的 LangGraph 免费课程，从基础到高级循序渐进
- [LangChain AI Handbook](https://www.pinecone.io/learn/series/langchain/) — Pinecone 出品的 LangChain 实战手册，配合向量数据库讲解

### GitHub 仓库

- [langchain-ai/langchain](https://github.com/langchain-ai/langchain) — LangChain 主仓库，包含核心代码和示例
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — LangGraph 仓库，包含大量 Agent 示例
- [langchain-ai/langchain-academy](https://github.com/langchain-ai/langchain-academy) — 官方教学仓库，配套 Jupyter Notebook

### 相关章节

- [Function Calling](../02-core-tech/03-function-calling.md) — 工具调用的底层原理
- [RAG](../02-core-tech/04-rag.md) — 检索增强生成的完整技术栈
- [AI Agent 开发](./04-ai-agent-development.md) — Agent 架构设计、记忆与安全
- [LLMOps](../05-production/01-llmops.md) — 评估流水线与 CI/CD
- [Agentic 系统设计](../06-advanced/01-agentic-system-design.md) — 多 Agent 编排与容错
