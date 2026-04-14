# 微调与模型定制

> 当 Prompt Engineering 和 RAG 不够时

## 学习目标

- 理解何时需要微调
- 掌握 LoRA/QLoRA 参数高效微调
- 建立数据准备与评估流程

---

## 1. 何时微调

### 1.1 决策框架

<!-- Prompt Engineering → RAG → 微调 的决策树 -->

### 1.2 适用场景

<!-- 特定格式输出、领域术语、风格一致性 -->

### 1.3 不适用场景

<!-- 知识注入（用 RAG）、通用能力提升 -->

## 2. 数据准备

### 2.1 数据收集

<!-- 真实数据、合成数据、数据增强 -->

### 2.2 数据格式

<!-- 对话格式、指令格式、偏好对 -->

### 2.3 数据质量

<!-- 清洗、去重、质量评估 -->

### 2.4 数据规模

<!-- 不同任务的最小数据量建议 -->

## 3. 微调方法

### 3.1 全量微调

<!-- 全参数更新、资源需求 -->

### 3.2 LoRA

<!-- 低秩适配、参数效率 -->

### 3.3 QLoRA

<!-- 量化 + LoRA、显存优化 -->

### 3.4 关键超参数

<!-- 学习率、Epoch、Rank、Alpha -->

## 4. 训练平台

### 4.1 本地训练

<!-- Unsloth、Hugging Face Transformers -->

### 4.2 云端训练

<!-- AWS SageMaker、Together AI、Modal -->

### 4.3 API 微调

<!-- OpenAI Fine-tuning API -->

## 5. 评估与迭代

### 5.1 评估指标

<!-- 任务特定指标、通用能力保持 -->

### 5.2 基准对比

<!-- 微调前后对比、与 RAG 方案对比 -->

### 5.3 迭代策略

<!-- 数据迭代、超参数调优 -->

## 6. 部署

### 6.1 模型合并

<!-- LoRA 权重合并、导出 -->

### 6.2 量化部署

<!-- GPTQ、AWQ、GGUF 格式 -->

### 6.3 推理服务

<!-- vLLM、TGI 部署 -->

---

## 练习

1. 用 QLoRA 微调一个 7B 模型完成特定任务
2. 对比微调前后的任务表现
3. 将微调模型部署为 API 服务

## 延伸阅读

- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [Unsloth](https://github.com/unslothai/unsloth)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
