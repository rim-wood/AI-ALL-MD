# LangChain / LangGraph

> 最流行的 LLM 应用开发框架与工作流编排引擎

## 学习目标

- 掌握 LangChain 核心抽象与 LCEL
- 理解 LangGraph 状态机与工作流编排
- 能够使用 LangSmith 进行调试与追踪

---

## 1. LangChain 核心概念

### 1.1 架构概览

<!-- langchain-core / langchain / langchain-community / 第三方集成 -->

### 1.2 核心抽象

<!-- ChatModel、PromptTemplate、OutputParser、Tool、Memory -->

### 1.3 LCEL（LangChain Expression Language）

<!-- 管道操作符、Runnable 接口、组合与并行 -->

## 2. 常用组件

### 2.1 Chat Models

<!-- 模型封装、统一接口、流式调用 -->

### 2.2 Prompt Templates

<!-- ChatPromptTemplate、MessagesPlaceholder -->

### 2.3 Output Parsers

<!-- JSON、Pydantic、结构化输出 -->

### 2.4 Tools & Toolkits

<!-- 工具定义、@tool 装饰器、内置工具 -->

### 2.5 Retrievers

<!-- 向量检索、多查询、上下文压缩 -->

## 3. LangGraph

### 3.1 为什么需要 LangGraph

<!-- Chain 的局限、循环与条件分支需求 -->

### 3.2 核心概念

<!-- StateGraph、Node、Edge、条件边 -->

### 3.3 状态管理

<!-- TypedDict / Pydantic State、状态更新 -->

### 3.4 检查点与持久化

<!-- MemorySaver、PostgresSaver、断点恢复 -->

### 3.5 Human-in-the-loop

<!-- 中断节点、人工审批、交互式工作流 -->

## 4. LangSmith

### 4.1 链路追踪

<!-- Trace 可视化、延迟分析 -->

### 4.2 评估

<!-- 数据集、评估器、自动化测试 -->

### 4.3 Prompt Hub

<!-- 提示词管理与版本控制 -->

## 5. 实战：多步骤 Agent

### 5.1 架构设计

<!-- 需求分析、状态图设计 -->

### 5.2 代码实现

<!-- 完整 LangGraph Agent 示例 -->

### 5.3 部署

<!-- LangServe / LangGraph Cloud -->

---

## 练习

1. 用 LCEL 构建一个 RAG 链
2. 用 LangGraph 实现一个带工具调用的 ReAct Agent
3. 在 LangSmith 中追踪和评估你的 Agent

## 延伸阅读

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangSmith 文档](https://docs.smith.langchain.com/)
