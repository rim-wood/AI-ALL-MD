# Prompt Engineering

> 与大语言模型高效沟通的核心技能

## 学习目标

- 掌握提示词设计的基本结构与原则
- 熟练运用主流提示词技巧
- 建立提示词版本管理与评估流程

---

## 1. 提示词基本结构

### 1.1 角色设定（System Prompt）

<!-- 角色定义、行为约束、输出格式 -->

### 1.2 任务描述

<!-- 清晰指令、边界条件、示例 -->

### 1.3 输出格式控制

<!-- JSON、Markdown、XML 标签 -->

## 2. 核心技巧

### 2.1 Zero-shot Prompting

<!-- 直接指令、无示例 -->

### 2.2 Few-shot Prompting

<!-- 示例驱动、示例选择策略 -->

### 2.3 Chain-of-Thought（CoT）

<!-- 逐步推理、"Let's think step by step" -->

### 2.4 ReAct 模式

<!-- 推理 + 行动交替、工具调用场景 -->

### 2.5 其他高级技巧

<!-- Self-Consistency、Tree-of-Thought、Reflection -->

## 3. 系统提示词设计模式

### 3.1 角色扮演模式

<!-- 专家角色、行为规范 -->

### 3.2 约束与防护

<!-- 输出边界、拒绝策略、安全规则 -->

### 3.3 多步骤任务编排

<!-- 分阶段指令、条件分支 -->

## 4. 提示词模板化

### 4.1 变量注入

<!-- 模板引擎、动态内容插入 -->

### 4.2 模板组合

<!-- 模块化提示词、可复用组件 -->

## 5. 调试与优化

### 5.1 常见陷阱

<!-- 指令冲突、上下文污染、幻觉诱导 -->

### 5.2 调试方法

<!-- 逐步简化、对比测试、日志分析 -->

### 5.3 版本管理

<!-- 提示词版本控制、变更追踪 -->

## 6. 评估

### 6.1 评估维度

<!-- 准确性、相关性、格式合规、安全性 -->

### 6.2 自动化评估

<!-- LLM-as-Judge、评估数据集构建 -->

---

## 练习

1. 为一个客服场景设计完整的系统提示词
2. 用 CoT 技巧解决一个多步推理问题
3. 对比同一任务在不同提示词下的输出质量

## 延伸阅读

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
