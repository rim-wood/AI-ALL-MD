# LlamaIndex

> 数据连接与 RAG 应用的专业框架

## 学习目标

- 理解 LlamaIndex 核心架构
- 掌握数据连接、索引构建与查询引擎
- 能够构建企业级知识库应用

---

## 1. 核心概念

### 1.1 架构概览

<!-- llama-index-core / 集成包 / LlamaHub -->

### 1.2 Document & Node

<!-- 文档抽象、节点关系 -->

### 1.3 Index

<!-- VectorStoreIndex、SummaryIndex、KnowledgeGraphIndex -->

## 2. 数据连接

### 2.1 LlamaHub 数据连接器

<!-- 文件、数据库、API、Web 爬虫 -->

### 2.2 文档解析

<!-- LlamaParse、PDF/HTML/Markdown 解析 -->

### 2.3 数据转换

<!-- 分块、元数据提取、节点后处理 -->

## 3. 索引构建

### 3.1 向量索引

<!-- VectorStoreIndex、存储后端集成 -->

### 3.2 摘要索引

<!-- SummaryIndex、层级摘要 -->

### 3.3 知识图谱索引

<!-- KnowledgeGraphIndex、图存储 -->

### 3.4 多索引策略

<!-- 路由查询、子问题分解 -->

## 4. 查询引擎

### 4.1 基础查询

<!-- QueryEngine、检索 + 生成 -->

### 4.2 Chat Engine

<!-- 多轮对话、上下文管理 -->

### 4.3 Sub-Question Query

<!-- 复杂问题分解 -->

### 4.4 路由查询

<!-- 多数据源路由 -->

## 5. 高级特性

### 5.1 Agent 集成

<!-- LlamaIndex Agent、工具封装 -->

### 5.2 Workflow

<!-- 事件驱动工作流 -->

### 5.3 Observability

<!-- 回调、追踪集成 -->

## 6. 与 LangChain 的对比与协作

### 6.1 定位差异

<!-- LlamaIndex 专注数据、LangChain 专注编排 -->

### 6.2 协作使用

<!-- LlamaIndex 作为 LangChain 的 Retriever -->

## 7. 实战：企业知识库

### 7.1 需求与架构

<!-- 多数据源、权限控制、增量更新 -->

### 7.2 代码实现

<!-- 完整示例 -->

---

## 练习

1. 用 LlamaIndex 构建一个多文档 RAG 系统
2. 实现一个带路由的多索引查询引擎
3. 对比 LlamaIndex 和 LangChain 实现同一 RAG 任务

## 延伸阅读

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaHub](https://llamahub.ai/)
- [LlamaParse](https://docs.cloud.llamaindex.ai/)
