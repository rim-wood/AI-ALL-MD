# AI 应用开发完整指南（2026版）

> 从基础知识到生产落地，系统掌握 AI 应用开发全栈技能

## 项目介绍

本指南面向希望系统学习 AI 应用开发的工程师，涵盖从编程基础、LLM 原理、RAG/Agent 核心技术，到框架实战、生产部署和前沿探索的完整知识体系。

## 目标读者

- 有编程基础，想转型 AI 应用开发的后端/前端工程师
- 已有 AI 基础，想系统补全生产落地知识的开发者
- 技术负责人，需要了解 AI 应用技术全景

## 前置要求

- Python 或 TypeScript 基础
- 基本的 Web 开发概念（HTTP、API、数据库）
- 命令行操作能力

## 知识体系全景图

```
基础知识 → 核心技术 → 框架工具 → 应用实践 → 生产部署 → 进阶前沿
```

## 目录

### 第一部分：基础知识

| 章节 | 内容 | 难度 |
|------|------|------|
| [Python 编程基础](01-fundamentals/01-python-basics.md) | 类型提示、异步编程、FastAPI、Pydantic | ⭐ |
| [TypeScript 编程基础](01-fundamentals/02-typescript-basics.md) | 类型系统、Zod、流式处理、Vercel AI SDK | ⭐ |
| [AI/ML 基础](01-fundamentals/03-ai-ml-basics.md) | 机器学习、Transformer、LLM 原理 | ⭐⭐ |
| [Prompt Engineering](01-fundamentals/04-prompt-engineering.md) | 提示词设计、CoT、ReAct、版本管理 | ⭐ |

### 第二部分：核心技术栈

| 章节 | 内容 | 难度 |
|------|------|------|
| [LLM API 与模型选型](02-core-tech/01-llm-api-and-models.md) | API 调用模式、选型决策框架、成本控制、企业最佳实践 | ⭐⭐ |
| [Context Engineering](02-core-tech/02-context-engineering.md) | 上下文工程、四大策略（Write/Select/Compress/Isolate）、长任务管理 | ⭐⭐ |
| [Function Calling](02-core-tech/03-function-calling.md) | 工具调用、结构化输出、错误处理 | ⭐⭐ |
| [RAG](02-core-tech/04-rag.md) | 检索增强生成、分块、混合检索、评估 | ⭐⭐⭐ |
| [Embedding 与向量搜索](02-core-tech/05-embeddings-and-vector-search.md) | 嵌入模型、向量数据库、ANN 算法 | ⭐⭐ |

### 第三部分：框架与工具

| 章节 | 内容 | 难度 |
|------|------|------|
| [LangChain / LangGraph](03-frameworks/01-langchain-langgraph.md) | 链式调用、状态机、工作流编排 | ⭐⭐ |
| [LlamaIndex](03-frameworks/02-llamaindex.md) | 数据连接、索引构建、查询引擎 | ⭐⭐ |
| [CrewAI / AutoGen](03-frameworks/03-crewai-autogen.md) | 多 Agent 协作框架 | ⭐⭐ |
| [AI Agent 开发](03-frameworks/04-ai-agent-development.md) | Agent 架构、记忆、工具、安全 | ⭐⭐⭐ |
| [MCP 协议](03-frameworks/05-mcp-protocol.md) | Model Context Protocol、Server 开发 | ⭐⭐ |

### 第四部分：应用实践

| 章节 | 内容 | 难度 |
|------|------|------|
| [智能客服与问答](04-practice/01-chatbot-and-qa.md) | 对话机器人、知识库问答、多轮对话 | ⭐⭐ |
| [前端与用户体验](04-practice/02-frontend-and-ux.md) | 流式输出、对话 UI、反馈机制 | ⭐⭐ |
| [HR 智能助手](04-practice/03-hr-assistant.md) | 简历筛选、政策问答、入职自动化 | ⭐⭐ |
| [数据分析 Agent](04-practice/04-data-analysis-agent.md) | Text-to-SQL、可视化、BI 集成 | ⭐⭐ |
| [多模态应用](04-practice/05-multimodal-apps.md) | 视觉、语音、图像生成、视频分析 | ⭐⭐⭐ |

### 第五部分：生产部署与运维

| 章节 | 内容 | 难度 |
|------|------|------|
| [LLMOps](05-production/01-llmops.md) | CI/CD、评估流水线、A/B 测试 | ⭐⭐⭐ |
| [监控与可观测性](05-production/02-monitoring.md) | 指标追踪、幻觉检测、漂移检测 | ⭐⭐⭐ |
| [安全与合规](05-production/03-security-and-compliance.md) | Prompt Injection、Guardrails、隐私 | ⭐⭐⭐ |
| [性能优化与成本控制](05-production/04-performance-and-cost.md) | 缓存、模型路由、推理优化 | ⭐⭐⭐ |

### 第六部分：进阶与前沿

| 章节 | 内容 | 难度 |
|------|------|------|
| [Agentic 系统设计](06-advanced/01-agentic-system-design.md) | 工作流编排、容错、多 Agent 架构 | ⭐⭐⭐⭐ |
| [微调与模型定制](06-advanced/02-fine-tuning.md) | LoRA/QLoRA、数据准备、评估 | ⭐⭐⭐⭐ |
| [前沿方向](06-advanced/03-frontier-topics.md) | 推理模型、超长上下文、AGI 探索 | ⭐⭐⭐⭐ |

### 第七部分：面试题

| 章节 | 内容 | 难度 |
|------|------|------|
| [AI 应用开发面试题](07-interview/01-interview-questions.md) | 25 道高频面试题，覆盖六大知识模块 | ⭐⭐ |

## 推荐学习路线

- **快速上手路线**：01 → 03（Prompt） → 06（Function Calling） → 08（LangChain） → 11（实战）
- **系统学习路线**：按章节顺序从头到尾
- **生产导向路线**：03 → 06 → 07 → 14 → 15 → 16 → 17

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Anthropic API 文档](https://docs.anthropic.com/)
- [MCP 协议规范](https://modelcontextprotocol.io/)

## License

MIT
