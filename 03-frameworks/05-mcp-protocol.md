# MCP 协议（Model Context Protocol）

> AI 应用的通用集成标准——"AI 的 USB-C"

## 学习目标

- 理解 MCP 协议的架构与设计理念
- 掌握 MCP Server 的开发方法
- 了解 MCP 与 Function Calling 的区别与协作

---

## 1. MCP 概述

### 1.1 什么是 MCP

<!-- 开放协议、标准化 AI 与外部工具的连接 -->

### 1.2 为什么需要 MCP

<!-- 碎片化集成问题、N×M → N+M -->

### 1.3 核心优势

<!-- 可复用、可发现、标准化、安全 -->

## 2. 架构设计

### 2.1 三层架构

<!-- Host（应用）→ Client（协议层）→ Server（工具层） -->

### 2.2 Server 能力

<!-- Tools（工具）、Resources（资源）、Prompts（提示词模板） -->

### 2.3 传输层

<!-- stdio（本地）、HTTP + SSE（远程） -->

### 2.4 会话生命周期

<!-- 初始化、能力协商、请求/响应、关闭 -->

## 3. MCP Server 开发

### 3.1 Python SDK

<!-- mcp 包、@server.tool 装饰器 -->

### 3.2 TypeScript SDK

<!-- @modelcontextprotocol/sdk -->

### 3.3 工具定义

<!-- 名称、描述、参数 Schema、返回值 -->

### 3.4 资源暴露

<!-- 文件、数据库、API 数据 -->

### 3.5 提示词模板

<!-- 可复用的提示词片段 -->

## 4. 认证与安全

### 4.1 OAuth 2.0 集成

<!-- 认证流程、Token 管理 -->

### 4.2 权限控制

<!-- 工具级权限、数据访问范围 -->

### 4.3 安全最佳实践

<!-- 输入验证、沙箱、审计日志 -->

## 5. MCP vs Function Calling vs Plugins

### 5.1 对比分析

<!-- 标准化程度、可复用性、生态 -->

### 5.2 协作使用

<!-- MCP 作为 Function Calling 的标准化层 -->

### 5.3 选型建议

<!-- 不同场景的推荐方案 -->

## 6. 生态与工具

### 6.1 官方 Server 列表

<!-- 文件系统、GitHub、Slack、数据库 -->

### 6.2 MCP Inspector

<!-- 调试与测试工具 -->

### 6.3 部署方案

<!-- 本地、Docker、Cloudflare Workers -->

## 7. 实战：自定义 MCP Server

### 7.1 需求设计

<!-- 数据库查询 MCP Server -->

### 7.2 代码实现

<!-- Python 完整示例 -->

### 7.3 测试与调试

<!-- MCP Inspector 使用 -->

### 7.4 集成到 AI 应用

<!-- Claude Desktop / Cursor / 自定义应用 -->

---

## 练习

1. 用 Python SDK 开发一个文件管理 MCP Server
2. 实现 OAuth 认证的远程 MCP Server
3. 将 MCP Server 集成到一个 LangGraph Agent 中

## 延伸阅读

- [MCP 协议规范](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Servers 列表](https://github.com/modelcontextprotocol/servers)
