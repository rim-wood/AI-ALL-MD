# AI/ML 基础

> 理解大语言模型背后的核心原理

## 学习目标

- 理解机器学习三大范式
- 掌握 Transformer 架构核心机制
- 了解 LLM 的训练与推理过程

---

## 1. 机器学习基础

### 1.1 监督学习

<!-- 分类、回归、损失函数、梯度下降 -->

### 1.2 无监督学习

<!-- 聚类、降维、异常检测 -->

### 1.3 强化学习

<!-- 奖励机制、策略优化、RLHF 基础 -->

## 2. 神经网络基础

### 2.1 前馈神经网络

<!-- 神经元、层、激活函数 -->

### 2.2 反向传播

<!-- 梯度计算、权重更新、优化器 -->

### 2.3 常见架构

<!-- CNN、RNN、Seq2Seq -->

## 3. Transformer 架构

### 3.1 Self-Attention 机制

<!-- Q/K/V、注意力分数、多头注意力 -->

### 3.2 位置编码

<!-- 正弦编码、RoPE、ALiBi -->

### 3.3 编码器-解码器结构

<!-- Encoder-only / Decoder-only / Encoder-Decoder -->

### 3.4 关键改进

<!-- Layer Norm、残差连接、KV Cache -->

## 4. 大语言模型（LLM）

### 4.1 预训练

<!-- 自回归语言建模、训练数据、Scaling Laws -->

### 4.2 指令微调（Instruction Tuning）

<!-- SFT、指令数据集构建 -->

### 4.3 RLHF / DPO

<!-- 人类反馈强化学习、直接偏好优化 -->

### 4.4 Tokenization

<!-- BPE、SentencePiece、Token 与成本的关系 -->

## 5. 推理过程

### 5.1 自回归生成

<!-- 逐 Token 生成、KV Cache -->

### 5.2 采样策略

<!-- Temperature、Top-p、Top-k、Beam Search -->

### 5.3 上下文窗口

<!-- 上下文长度限制、长文本处理策略 -->

---

## 练习

1. 手绘 Transformer 架构图并标注各组件
2. 对比不同 Temperature 值对生成结果的影响
3. 计算一段文本的 Token 数量与 API 成本

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
