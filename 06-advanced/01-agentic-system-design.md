# Agentic 系统设计

> 构建复杂、可靠的自主 AI 系统

## 学习目标

- 掌握复杂工作流编排模式
- 理解持久化执行与容错设计
- 设计企业级多 Agent 架构

---

## 1. 工作流编排

Agentic 系统的核心挑战之一是如何将复杂任务分解为可管理的步骤，并以可靠的方式编排执行。根据任务特性的不同，主要有三种编排模式：DAG、状态机和事件驱动。

### 1.1 DAG（有向无环图）

DAG 是最常见的工作流编排模式。每个节点代表一个任务，边代表依赖关系。没有依赖的任务可以并行执行，所有上游任务完成后才执行下游任务。

**核心概念：**

| 概念 | 说明 |
|------|------|
| 节点（Node） | 一个独立的执行单元，如 LLM 调用、工具调用、数据处理 |
| 边（Edge） | 节点间的依赖关系，定义执行顺序 |
| 扇出（Fan-out） | 一个节点的输出分发给多个下游节点并行处理 |
| 扇入（Fan-in） | 多个节点的输出汇聚到一个节点进行合并 |

```
        ┌──→ [研究竞品A] ──┐
[分析需求] ──→ [研究竞品B] ──→ [汇总报告] → [生成建议]
        └──→ [研究竞品C] ──┘
```

用 LangGraph 实现一个并行研究再汇总的 DAG：

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    report: str

def analyze_requirements(state: ResearchState) -> dict:
    return {"topic": state["topic"]}

def research_competitor(name: str):
    """工厂函数：为每个竞品创建研究节点"""
    def _research(state: ResearchState) -> dict:
        # 实际项目中调用 LLM 进行研究
        return {"findings": [f"{name} 的分析结果..."]}
    return _research

def summarize(state: ResearchState) -> dict:
    combined = "\n".join(state["findings"])
    return {"report": f"汇总报告：\n{combined}"}

graph = StateGraph(ResearchState)
graph.add_node("analyze", analyze_requirements)
graph.add_node("research_a", research_competitor("竞品A"))
graph.add_node("research_b", research_competitor("竞品B"))
graph.add_node("research_c", research_competitor("竞品C"))
graph.add_node("summarize", summarize)

graph.add_edge(START, "analyze")
# 扇出：并行研究
graph.add_edge("analyze", "research_a")
graph.add_edge("analyze", "research_b")
graph.add_edge("analyze", "research_c")
# 扇入：汇总
graph.add_edge("research_a", "summarize")
graph.add_edge("research_b", "summarize")
graph.add_edge("research_c", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()
result = app.invoke({"topic": "CRM 市场分析", "findings": []})
```

DAG 适合任务依赖关系明确、不需要循环的场景，如数据处理流水线、多源信息汇总、并行工具调用等。

### 1.2 状态机

当工作流包含条件分支、循环或需要根据中间结果动态决定下一步时，状态机是更合适的选择。LangGraph 的核心设计就是基于状态机模型。

**与 DAG 的关键区别：**

| 特性 | DAG | 状态机 |
|------|-----|--------|
| 循环 | ❌ 不支持 | ✅ 支持 |
| 动态路由 | ❌ 静态边 | ✅ 条件边 |
| 适用场景 | 固定流水线 | 交互式、迭代式任务 |

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class AgentState(TypedDict):
    query: str
    draft: str
    feedback: str
    revision_count: int

def write_draft(state: AgentState) -> dict:
    # LLM 生成初稿
    return {"draft": "初稿内容...", "revision_count": state.get("revision_count", 0)}

def review_draft(state: AgentState) -> dict:
    # LLM 审查并给出反馈
    return {"feedback": "需要补充数据支撑..."}

def revise_draft(state: AgentState) -> dict:
    count = state["revision_count"] + 1
    return {"draft": f"修订版 v{count}...", "revision_count": count}

def should_revise(state: AgentState) -> Literal["revise", "done"]:
    """条件路由：决定是否继续修改"""
    if state["revision_count"] >= 3:
        return "done"
    if "需要" in state.get("feedback", ""):
        return "revise"
    return "done"

graph = StateGraph(AgentState)
graph.add_node("write", write_draft)
graph.add_node("review", review_draft)
graph.add_node("revise", revise_draft)

graph.add_edge(START, "write")
graph.add_edge("write", "review")
graph.add_conditional_edges("review", should_revise, {
    "revise": "revise",
    "done": END,
})
graph.add_edge("revise", "review")  # 循环：修改后再审查

app = graph.compile()
```

状态机模式非常适合需要迭代优化的场景，如写作助手（写→审→改循环）、代码生成（生成→测试→修复循环）、客服对话（多轮状态流转）等。

### 1.3 事件驱动

事件驱动架构将工作流解耦为独立的事件生产者和消费者，通过消息队列实现异步通信。这种模式适合高吞吐、松耦合的系统。

**核心组件：**

```
[用户请求] → [消息队列] → [Agent Worker 1]
                        → [Agent Worker 2] → [结果队列] → [响应聚合]
                        → [Agent Worker 3]
```

事件驱动的典型实现使用 Redis Streams 或 Kafka 作为消息中间件：

```python
import asyncio
import redis.asyncio as redis

class AgentEventBus:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url)

    async def publish(self, stream: str, event: dict):
        """发布事件到指定流"""
        await self.client.xadd(stream, event)

    async def subscribe(self, stream: str, group: str, consumer: str):
        """消费者组订阅事件"""
        try:
            await self.client.xgroup_create(stream, group, mkstream=True)
        except redis.ResponseError:
            pass  # 组已存在

        while True:
            messages = await self.client.xreadgroup(
                group, consumer, {stream: ">"}, count=1, block=5000
            )
            for _, msgs in messages:
                for msg_id, data in msgs:
                    yield msg_id, data
                    await self.client.xack(stream, group, msg_id)
```

**三种编排模式的选型指南：**

| 场景 | 推荐模式 | 理由 |
|------|----------|------|
| 数据处理流水线 | DAG | 步骤固定，可并行 |
| 对话式 Agent | 状态机 | 需要循环和条件分支 |
| 微服务架构 | 事件驱动 | 松耦合，高可扩展 |
| 多步骤审批流 | 状态机 | 需要人工介入和状态流转 |
| 实时数据流处理 | 事件驱动 | 高吞吐，异步处理 |

---

## 2. 持久化执行

Agentic 系统经常需要执行长时间运行的任务——可能持续数分钟甚至数小时。网络中断、服务重启、LLM API 超时等故障随时可能发生。持久化执行确保系统能从故障中恢复，而不是从头开始。

### 2.1 检查点机制

检查点（Checkpoint）是持久化执行的基础。在工作流的关键节点保存完整状态快照，故障发生后可以从最近的检查点恢复执行。

LangGraph 内置了检查点支持，使用 `MemorySaver`（开发）或 `PostgresSaver`（生产）：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from typing import TypedDict

class TaskState(TypedDict):
    task: str
    steps_completed: list[str]
    result: str

def step_a(state: TaskState) -> dict:
    return {"steps_completed": state.get("steps_completed", []) + ["step_a"]}

def step_b(state: TaskState) -> dict:
    return {"steps_completed": state["steps_completed"] + ["step_b"]}

def step_c(state: TaskState) -> dict:
    return {"result": f"完成，共 {len(state['steps_completed']) + 1} 步"}

graph = StateGraph(TaskState)
graph.add_node("a", step_a)
graph.add_node("b", step_b)
graph.add_node("c", step_c)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", "c")
graph.add_edge("c", END)

# 生产环境使用 PostgresSaver 持久化检查点
async def run():
    async with AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost/checkpoints"
    ) as saver:
        await saver.setup()
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "task-001"}}
        result = await app.ainvoke(
            {"task": "数据分析", "steps_completed": []},
            config=config,
        )

        # 查看检查点历史
        async for checkpoint in saver.alist(config):
            print(f"检查点: {checkpoint}")
```

**检查点策略：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 每步保存 | 每个节点执行后保存 | 步骤耗时长、失败代价高 |
| 关键节点保存 | 仅在重要节点保存 | 步骤多但执行快 |
| 定时保存 | 按时间间隔保存 | 长时间连续处理 |

### 2.2 长时间运行任务

对于可能运行数分钟到数小时的任务，需要超时控制和心跳机制来保证系统的可靠性。

```python
import asyncio
from datetime import datetime, timedelta

class LongRunningTask:
    def __init__(self, task_id: str, timeout: int = 3600):
        self.task_id = task_id
        self.timeout = timeout
        self.last_heartbeat = datetime.now()
        self._cancelled = False

    async def heartbeat(self):
        """定期发送心跳，表明任务仍在运行"""
        self.last_heartbeat = datetime.now()

    def is_alive(self) -> bool:
        """检查任务是否仍然活跃"""
        return (datetime.now() - self.last_heartbeat) < timedelta(seconds=60)

    async def execute_with_timeout(self, coro):
        """带超时的任务执行"""
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout)
        except asyncio.TimeoutError:
            print(f"任务 {self.task_id} 超时，执行清理...")
            await self.cleanup()
            raise

    async def cleanup(self):
        """超时或取消后的清理逻辑"""
        self._cancelled = True
```

### 2.3 故障恢复

生产环境中，故障恢复策略直接决定系统的可靠性。核心策略包括重试、补偿事务和幂等性设计。

**重试策略：**

```python
import asyncio
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"第 {attempt + 1} 次失败: {e}，{delay:.1f}s 后重试")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, base_delay=1.0)
async def call_llm(prompt: str) -> str:
    # LLM API 调用，失败时自动重试
    ...
```

**幂等性设计** 是故障恢复的关键——同一操作执行多次的结果与执行一次相同：

```python
class IdempotentExecutor:
    def __init__(self, store):
        self.store = store  # Redis 或数据库

    async def execute_once(self, operation_id: str, func, *args):
        """确保操作只执行一次"""
        # 检查是否已执行
        result = await self.store.get(f"op:{operation_id}")
        if result is not None:
            return result  # 直接返回缓存结果

        # 首次执行
        result = await func(*args)
        await self.store.set(f"op:{operation_id}", result)
        return result
```

**补偿事务（Saga 模式）** 用于多步骤操作的回滚：当某一步失败时，按逆序执行前面步骤的补偿操作。例如"创建订单 → 扣减库存 → 发送通知"，如果发送通知失败，需要依次恢复库存、取消订单。

---

## 3. Human-in-the-loop

完全自主的 Agent 在高风险场景中并不可靠。Human-in-the-loop（HITL）模式在关键决策点引入人工干预，在自动化效率和人工可控性之间取得平衡。

### 3.1 审批节点

审批节点是最常见的 HITL 模式。当 Agent 执行到需要人工确认的步骤时，暂停执行并等待审批。

LangGraph 通过 `interrupt` 机制原生支持审批节点：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class OrderState(TypedDict):
    order: dict
    approved: bool
    result: str

def prepare_order(state: OrderState) -> dict:
    return {"order": {"item": "GPU 服务器", "amount": 50000}}

def human_approval(state: OrderState) -> dict:
    """审批节点：暂停等待人工审批"""
    order = state["order"]
    decision = interrupt({
        "message": f"请审批订单：{order['item']}，金额 ¥{order['amount']}",
        "options": ["approve", "reject"],
    })
    return {"approved": decision == "approve"}

def process_order(state: OrderState) -> dict:
    if state["approved"]:
        return {"result": "订单已处理"}
    return {"result": "订单已拒绝"}

graph = StateGraph(OrderState)
graph.add_node("prepare", prepare_order)
graph.add_node("approve", human_approval)
graph.add_node("process", process_order)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "approve")
graph.add_edge("approve", "process")
graph.add_edge("process", END)

app = graph.compile(checkpointer=MemorySaver())

# 第一次调用：执行到审批节点暂停
config = {"configurable": {"thread_id": "order-001"}}
result = app.invoke({"order": {}, "approved": False, "result": ""}, config=config)

# 人工审批后恢复执行
result = app.invoke(Command(resume="approve"), config=config)
print(result["result"])  # "订单已处理"
```

**审批触发条件设计：**

| 条件 | 示例 | 说明 |
|------|------|------|
| 金额阈值 | 订单 > ¥10,000 | 大额操作需审批 |
| 风险评分 | 风险分 > 0.8 | 高风险操作需审批 |
| 操作类型 | 删除、修改权限 | 不可逆操作需审批 |
| 置信度 | LLM 置信度 < 0.7 | 模型不确定时需审批 |

### 3.2 交互式修正

交互式修正允许人工在工作流执行过程中查看中间结果并进行调整，而不仅仅是简单的批准/拒绝。

典型场景包括：

- **文档生成**：Agent 生成大纲后，用户调整结构再继续生成正文
- **数据分析**：Agent 提出分析方案后，用户修改分析维度
- **代码重构**：Agent 提出重构计划后，用户选择性采纳

```python
def interactive_review(state: dict) -> dict:
    """交互式修正节点"""
    draft = state["draft"]
    feedback = interrupt({
        "type": "review",
        "content": draft,
        "message": "请审查并修改以下内容，或输入 'ok' 接受：",
    })

    if feedback == "ok":
        return {"draft": draft, "human_edited": False}
    return {"draft": feedback, "human_edited": True}
```

### 3.3 升级机制

升级机制（Escalation）在 Agent 遇到超出能力范围的问题时，自动将任务转交给人工处理。好的升级机制需要准确判断何时升级，避免过度升级（降低效率）或升级不足（产生错误）。

```python
class EscalationPolicy:
    def __init__(self):
        self.rules = [
            {"condition": "confidence_low", "threshold": 0.5},
            {"condition": "sensitive_topic", "keywords": ["投诉", "法律", "赔偿"]},
            {"condition": "max_turns", "limit": 10},
            {"condition": "user_request", "trigger": "转人工"},
        ]

    def should_escalate(self, context: dict) -> tuple[bool, str]:
        # 用户主动请求
        if context.get("user_message", "").find("转人工") >= 0:
            return True, "用户主动请求转人工"

        # 置信度过低
        if context.get("confidence", 1.0) < 0.5:
            return True, "模型置信度过低"

        # 敏感话题
        msg = context.get("user_message", "")
        for kw in ["投诉", "法律", "赔偿"]:
            if kw in msg:
                return True, f"检测到敏感话题: {kw}"

        # 对话轮次过多
        if context.get("turn_count", 0) > 10:
            return True, "对话轮次超限"

        return False, ""
```

升级时应传递完整的对话上下文和 Agent 的分析摘要，让人工客服能快速接手而无需用户重复描述问题。

---

## 4. 多 Agent 架构

当单个 Agent 无法胜任复杂任务时，需要多个专业化 Agent 协作完成。多 Agent 架构的核心问题是：如何组织 Agent 之间的协作关系？

### 4.1 Supervisor 模式

Supervisor 模式是最直观的多 Agent 架构。一个 Supervisor Agent 负责理解任务、分配子任务给 Worker Agent，并汇总结果。

```
                    ┌──→ [研究 Agent] ──┐
[用户请求] → [Supervisor] ──→ [写作 Agent] ──→ [Supervisor] → [最终输出]
                    └──→ [审查 Agent] ──┘
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict, Literal

class TeamState(TypedDict):
    task: str
    messages: list[dict]
    next_agent: str
    result: str

def supervisor(state: TeamState) -> Command[Literal["researcher", "writer", "reviewer"]]:
    """Supervisor 决定下一步由哪个 Agent 执行"""
    # 实际项目中用 LLM 做路由决策
    task = state["task"]
    if not state.get("messages"):
        return Command(goto="researcher", update={"next_agent": "researcher"})
    last = state["messages"][-1]
    if last.get("role") == "researcher":
        return Command(goto="writer", update={"next_agent": "writer"})
    return Command(goto="reviewer", update={"next_agent": "reviewer"})

def researcher(state: TeamState) -> dict:
    return {"messages": state["messages"] + [
        {"role": "researcher", "content": "研究结果：..."}
    ]}

def writer(state: TeamState) -> dict:
    return {"messages": state["messages"] + [
        {"role": "writer", "content": "文章草稿：..."}
    ]}

def reviewer(state: TeamState) -> dict:
    return {"result": "审查通过，最终输出：..."}

graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.add_edge(START, "supervisor")
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_edge("reviewer", END)

app = graph.compile()
```

**优点：** 控制流清晰，易于调试和监控。**缺点：** Supervisor 是单点瓶颈，所有通信都经过它。

### 4.2 Swarm 模式

Swarm 模式（也称 Handoff 模式）中没有中心调度者，Agent 之间直接传递控制权。每个 Agent 根据当前状态决定是自己处理还是交给其他 Agent。OpenAI 的 Swarm 框架就是这种模式的代表。

```
[用户请求] → [分诊 Agent] → [技术支持 Agent] → [退款 Agent] → [完成]
                                    ↓
                              [升级 Agent]
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

def triage_agent(state: dict) -> Command:
    """分诊 Agent：根据问题类型路由"""
    query = state["query"]
    if "退款" in query or "退货" in query:
        return Command(goto="refund_agent")
    if "故障" in query or "报错" in query:
        return Command(goto="tech_agent")
    return Command(goto="general_agent")

def refund_agent(state: dict) -> Command:
    """退款 Agent：处理退款或交回分诊"""
    # 处理退款逻辑...
    if state.get("needs_tech_check"):
        return Command(goto="tech_agent")  # 直接交给技术 Agent
    return Command(goto=END, update={"result": "退款已处理"})
```

**Supervisor vs Swarm 对比：**

| 特性 | Supervisor | Swarm |
|------|-----------|-------|
| 控制方式 | 中心化 | 去中心化 |
| 通信路径 | 星型（经过 Supervisor） | 网状（Agent 间直接） |
| 灵活性 | 中等 | 高 |
| 可调试性 | 高（单一决策点） | 低（分散决策） |
| 适用规模 | 3-8 个 Agent | 2-5 个 Agent |

### 4.3 层级模式

层级模式将 Agent 组织为树状结构，上层 Agent 管理下层 Agent。适合大规模、多层次的复杂任务。

```
                [总监 Agent]
               /            \
      [研究经理]           [工程经理]
      /        \           /        \
[搜索Agent] [分析Agent] [前端Agent] [后端Agent]
```

```python
from langgraph.graph import StateGraph, START, END

def build_team(manager_name: str, workers: list[str]):
    """构建一个管理者+执行者的子图"""
    team = StateGraph(dict)

    def manager(state):
        # 管理者分配任务给 workers
        return {"assignments": {w: f"子任务-{w}" for w in workers}}

    team.add_node("manager", manager)
    for worker in workers:
        team.add_node(worker, lambda s, w=worker: {"results": {w: f"{w}完成"}})
        team.add_edge("manager", worker)
        team.add_edge(worker, END)
    team.add_edge(START, "manager")
    return team.compile()

# 构建层级结构
research_team = build_team("研究经理", ["搜索Agent", "分析Agent"])
engineering_team = build_team("工程经理", ["前端Agent", "后端Agent"])
```

层级模式的优势在于可以递归分解复杂任务，每一层只需关注自己的职责范围。缺点是层级过深会导致信息传递延迟和失真。

### 4.4 辩论模式

辩论模式让多个 Agent 对同一问题给出独立观点，通过多轮讨论达成共识或由裁判 Agent 做最终决策。这种模式特别适合需要多角度分析的场景。

```python
from typing import TypedDict

class DebateState(TypedDict):
    topic: str
    rounds: list[dict]
    consensus: str

def optimist_agent(state: DebateState) -> dict:
    """乐观派 Agent"""
    history = state.get("rounds", [])
    # LLM 基于历史讨论给出乐观视角的分析
    return {"rounds": history + [{"agent": "optimist", "argument": "从积极面看..."}]}

def pessimist_agent(state: DebateState) -> dict:
    """悲观派 Agent"""
    history = state.get("rounds", [])
    return {"rounds": history + [{"agent": "pessimist", "argument": "风险在于..."}]}

def judge_agent(state: DebateState) -> dict:
    """裁判 Agent：综合各方观点做出判断"""
    all_arguments = state["rounds"]
    # LLM 综合分析所有观点
    return {"consensus": "综合考虑，建议..."}
```

**四种架构的选型指南：**

| 架构 | 最佳场景 | Agent 数量 | 复杂度 |
|------|----------|-----------|--------|
| Supervisor | 任务分配明确的团队协作 | 3-8 | 中 |
| Swarm | 客服路由、流程流转 | 2-5 | 低 |
| 层级 | 大型项目、多团队协作 | 10+ | 高 |
| 辩论 | 决策分析、风险评估 | 2-4 | 中 |

---

## 5. Agent 间通信

多 Agent 系统的效率很大程度上取决于 Agent 之间如何交换信息。通信机制的设计直接影响系统的可扩展性和可维护性。

### 5.1 消息传递

消息传递是最基本的通信方式。每个 Agent 通过结构化消息交换信息，消息中包含发送者、接收者、消息类型和负载。

```python
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class MessageType(str, Enum):
    TASK = "task"           # 任务分配
    RESULT = "result"       # 结果返回
    QUERY = "query"         # 信息查询
    FEEDBACK = "feedback"   # 反馈

class AgentMessage(BaseModel):
    sender: str
    receiver: str
    msg_type: MessageType
    payload: dict
    timestamp: datetime = datetime.now()
    correlation_id: str | None = None  # 关联请求-响应

# 使用示例
msg = AgentMessage(
    sender="researcher",
    receiver="writer",
    msg_type=MessageType.RESULT,
    payload={
        "findings": ["发现1", "发现2"],
        "confidence": 0.85,
        "sources": ["url1", "url2"],
    },
    correlation_id="task-001",
)
```

消息传递的关键设计原则：
- **消息不可变**：发送后不能修改，保证可追溯
- **携带上下文**：包含足够的上下文信息，接收方无需额外查询
- **使用 correlation_id**：关联请求和响应，支持异步通信

### 5.2 共享状态

共享状态模式中，所有 Agent 读写同一个全局状态对象。LangGraph 的 `State` 就是这种模式——所有节点共享同一个状态字典。

**黑板模式（Blackboard Pattern）** 是共享状态的经典实现：

```python
from typing import Any
import asyncio

class Blackboard:
    """黑板模式：Agent 通过共享黑板交换信息"""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._watchers: dict[str, list] = {}

    async def write(self, key: str, value: Any, author: str):
        async with self._lock:
            self._data[key] = {
                "value": value,
                "author": author,
                "version": self._data.get(key, {}).get("version", 0) + 1,
            }
        # 通知关注该 key 的 Agent
        for callback in self._watchers.get(key, []):
            await callback(key, value)

    async def read(self, key: str) -> Any:
        return self._data.get(key, {}).get("value")

    def watch(self, key: str, callback):
        """注册对某个 key 的监听"""
        self._watchers.setdefault(key, []).append(callback)
```

**消息传递 vs 共享状态：**

| 特性 | 消息传递 | 共享状态 |
|------|----------|----------|
| 耦合度 | 低 | 高 |
| 一致性 | 最终一致 | 强一致（需加锁） |
| 可扩展性 | 高（可分布式） | 中（状态同步开销） |
| 调试难度 | 中（需追踪消息流） | 低（状态可直接查看） |

### 5.3 协议设计

当 Agent 系统规模增大时，需要定义标准化的通信协议。好的协议应该包含握手、确认和错误处理机制。

```python
from pydantic import BaseModel
from enum import Enum

class ProtocolPhase(str, Enum):
    REQUEST = "request"
    ACK = "ack"           # 确认收到
    PROCESSING = "processing"
    RESULT = "result"
    ERROR = "error"

class ProtocolMessage(BaseModel):
    phase: ProtocolPhase
    task_id: str
    sender: str
    receiver: str
    payload: dict = {}
    error: str | None = None

class AgentProtocol:
    """Agent 通信协议"""

    async def request(self, target: str, task: dict) -> str:
        """发送任务请求，返回 task_id"""
        task_id = f"task-{id(task)}"
        msg = ProtocolMessage(
            phase=ProtocolPhase.REQUEST,
            task_id=task_id,
            sender=self.name,
            receiver=target,
            payload=task,
        )
        await self.send(msg)
        # 等待 ACK
        ack = await self.wait_for(task_id, ProtocolPhase.ACK, timeout=30)
        if not ack:
            raise TimeoutError(f"Agent {target} 未响应")
        return task_id

    async def acknowledge(self, task_id: str, sender: str):
        """确认收到任务"""
        msg = ProtocolMessage(
            phase=ProtocolPhase.ACK,
            task_id=task_id,
            sender=self.name,
            receiver=sender,
        )
        await self.send(msg)
```

在实际项目中，MCP（Model Context Protocol）正在成为 Agent 与外部工具通信的标准协议。对于 Agent 之间的通信，可以参考 MCP 的设计理念，定义统一的消息格式和交互流程。

---

## 6. 企业级设计

将 Agentic 系统从原型推向生产，需要解决可扩展性、可观测性和安全性三大核心问题。

### 6.1 可扩展性

企业级 Agentic 系统需要支持高并发和水平扩展。关键设计包括：

**无状态 Worker + 外部状态存储：**

```python
from fastapi import FastAPI
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

app = FastAPI()

# Worker 无状态，状态存储在 PostgreSQL
async def get_graph():
    saver = AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@db:5432/agents"
    )
    await saver.setup()
    graph = build_agent_graph()  # 构建图
    return graph.compile(checkpointer=saver)

@app.post("/run")
async def run_agent(request: dict):
    graph = await get_graph()
    config = {"configurable": {"thread_id": request["thread_id"]}}
    result = await graph.ainvoke(request["input"], config=config)
    return result
```

**扩展策略：**

| 策略 | 实现方式 | 适用场景 |
|------|----------|----------|
| 水平扩展 Worker | Kubernetes HPA | 请求量波动大 |
| 任务队列 | Celery / Redis Queue | 长时间任务 |
| Agent 池化 | 预创建 Agent 实例复用 | 初始化开销大 |
| 模型路由 | 简单任务用小模型 | 降低成本和延迟 |

### 6.2 可观测性

Agentic 系统的决策链路长且复杂，可观测性是调试和优化的基础。需要追踪每一步的输入、输出、耗时和 token 消耗。

```python
import time
import uuid
from functools import wraps

class AgentTracer:
    def __init__(self):
        self.spans: list[dict] = []

    def trace(self, node_name: str):
        """追踪装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                span = {
                    "span_id": str(uuid.uuid4()),
                    "node": node_name,
                    "start_time": time.time(),
                    "input": str(args)[:500],
                }
                try:
                    result = await func(*args, **kwargs)
                    span["output"] = str(result)[:500]
                    span["status"] = "success"
                    return result
                except Exception as e:
                    span["error"] = str(e)
                    span["status"] = "error"
                    raise
                finally:
                    span["duration_ms"] = (time.time() - span["start_time"]) * 1000
                    self.spans.append(span)
            return wrapper
        return decorator

    def get_trace_summary(self) -> dict:
        total_ms = sum(s["duration_ms"] for s in self.spans)
        errors = [s for s in self.spans if s["status"] == "error"]
        return {
            "total_spans": len(self.spans),
            "total_duration_ms": total_ms,
            "error_count": len(errors),
            "spans": self.spans,
        }
```

生产环境推荐使用 LangSmith 或 OpenTelemetry 进行全链路追踪，它们提供可视化的决策树视图、token 用量统计和延迟分析。

### 6.3 安全架构

Agentic 系统拥有执行工具的能力，安全设计至关重要。核心原则是 **最小权限** 和 **纵深防御**。

```python
from enum import Enum

class Permission(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    DATABASE_WRITE = "database_write"

class AgentSandbox:
    """Agent 沙箱：限制 Agent 的权限和资源"""

    def __init__(self, agent_id: str, permissions: set[Permission]):
        self.agent_id = agent_id
        self.permissions = permissions
        self.resource_limits = {
            "max_tokens_per_call": 4096,
            "max_calls_per_minute": 60,
            "max_file_size_bytes": 10 * 1024 * 1024,
            "allowed_domains": ["api.openai.com", "internal-api.company.com"],
        }
        self._call_count = 0

    def check_permission(self, action: Permission) -> bool:
        if action not in self.permissions:
            raise PermissionError(
                f"Agent {self.agent_id} 无权执行 {action.value}"
            )
        return True

    def check_rate_limit(self) -> bool:
        self._call_count += 1
        if self._call_count > self.resource_limits["max_calls_per_minute"]:
            raise RuntimeError("超出速率限制")
        return True

# 为不同角色的 Agent 配置不同权限
researcher_sandbox = AgentSandbox("researcher", {
    Permission.READ_FILE, Permission.NETWORK_ACCESS
})
writer_sandbox = AgentSandbox("writer", {
    Permission.READ_FILE, Permission.WRITE_FILE
})
```

**安全设计清单：**

| 层面 | 措施 |
|------|------|
| 权限控制 | 每个 Agent 只授予必要的最小权限 |
| 输入验证 | 对 Agent 的工具调用参数进行校验 |
| 输出过滤 | 检查 Agent 输出中是否包含敏感信息 |
| 资源限制 | 限制 token 用量、API 调用频率、文件大小 |
| 沙箱隔离 | 代码执行在隔离容器中运行 |
| 审计日志 | 记录所有 Agent 的决策和操作 |

---

## 练习

1. 用 LangGraph 实现一个带审批节点的工作流
2. 设计一个 Supervisor + Worker 的多 Agent 系统
3. 实现检查点持久化与故障恢复

## 延伸阅读

- [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Building Effective Agents (Anthropic)](https://docs.anthropic.com/en/docs/build-with-claude/agent)
