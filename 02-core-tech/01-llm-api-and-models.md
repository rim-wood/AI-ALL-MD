# LLM API 与模型选型

> 掌握 API 调用模式与模型选型决策框架——不追具体模型参数，聚焦选型方法论

## 学习目标

- 掌握 LLM API 三种调用模式（同步/流式/批量）及适用场景
- 建立可复用的模型选型决策框架，而非记忆具体模型参数
- 理解企业级模型选型的成本、安全与工程化考量
- 能在面试和实际项目中清晰阐述选型思路

> **为什么不列模型清单？** LLM 领域每季度都有新模型发布，具体参数和价格很快过时。本章聚焦**选型方法论**——掌握了框架，面对任何新模型都能快速评估。最新模型信息请查阅 [LMSYS Chatbot Arena](https://chat.lmsys.org/) 和 [Artificial Analysis](https://artificialanalysis.ai/)。

---

## 1. API 调用模式

所有主流 LLM 提供商的 API 都遵循相似的调用模式。掌握这三种模式，切换任何提供商都能快速上手。

### 1.1 同步调用

发送请求，等待完整响应返回。适合后端处理、批量任务等不需要实时反馈的场景。

```python
from openai import OpenAI

client = OpenAI()  # 自动读取 OPENAI_API_KEY 环境变量

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "用一句话解释什么是 Transformer"},
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
print(f"Token 用量: {response.usage.total_tokens}")
```

**关键参数速查：**

| 参数 | 范围 | 实战建议 |
|------|------|---------|
| `temperature` | 0 - 2 | 事实/分类任务用 0-0.3，创作/对话用 0.7-1.0 |
| `max_tokens` | 1 - 模型上限 | 按需设置，避免浪费 Token 预算 |
| `top_p` | 0 - 1 | 与 temperature 二选一调节，不要同时改 |
| `stop` | 字符串/数组 | 控制输出边界，结构化场景很有用 |

### 1.2 流式调用

通过 SSE 逐 Token 返回，用户实时看到生成过程。**所有面向用户的对话界面都应使用流式调用。**

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首关于编程的诗"}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### 1.3 批量调用

Batch API 适合大规模离线处理（数据标注、批量翻译），通常可获得 **50% 价格折扣**，响应时间 24 小时内。

### 1.4 多提供商统一调用

大多数提供商兼容 OpenAI API 格式，一套代码即可切换模型——这是企业级项目的标准做法：

```python
from openai import OpenAI

PROVIDERS = {
    "openai": {"base_url": "https://api.openai.com/v1", "api_key": "sk-..."},
    "deepseek": {"base_url": "https://api.deepseek.com", "api_key": "sk-..."},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
}

def chat(provider: str, model: str, prompt: str) -> str:
    """统一调用接口，一套代码切换多个模型"""
    config = PROVIDERS[provider]
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
```

> **实战经验：** 项目初期就应该抽象出 LLM 调用层，不要把具体模型硬编码到业务逻辑中。后续切换模型、做 A/B 测试、实现 fallback 都会非常方便。

---

## 2. 模型选型决策框架

### 2.1 五维评估模型

选型不是"哪个模型最强"，而是"哪个模型最适合当前场景"。用以下五个维度系统评估：

| 维度 | 关键问题 | 评估方法 |
|------|---------|---------|
| **任务类型** | 通用对话？代码？推理？多模态？ | 用 3-5 个代表性 case 测试 |
| **质量要求** | 能容忍偶尔错误？还是必须高度准确？ | 定义验收标准，量化评估 |
| **延迟要求** | 实时交互（<2s）？还是离线处理？ | 测量 TTFT 和总响应时间 |
| **成本预算** | 日均调用量 × 单次 Token 量 × 单价 | 先算月成本，再决定模型 |
| **数据隐私** | 数据能否发送到第三方？有合规要求？ | 合规红线不可妥协 |

### 2.2 决策路径

```
开始选型
│
├─ 数据能否离开内网？
│  └─ 不能 → 开源模型自部署（Llama / Qwen / Mistral 等）
│     ├─ GPU 资源充足？ → 70B+ 大模型
│     └─ 资源有限？ → 7B-14B 小模型 或 MoE 架构
│
├─ 需要多模态能力？
│  └─ 是 → 优先 Gemini（最全面）或 GPT-4o（图片+音频）
│
├─ 需要深度推理（数学/编程/复杂分析）？
│  └─ 是 → 推理模型系列（o3 / Claude Extended Thinking / DeepSeek-R1）
│
├─ 成本敏感（高频调用）？
│  └─ 是 → 轻量模型（GPT-4o mini / Gemini Flash / DeepSeek-V3）
│
└─ 以上都不是 → 通用旗舰模型（GPT-4o / Claude Sonnet）
```

### 2.3 实战选型方法论

**第一步：建立 Baseline**

不要一开始就选最贵的模型。用最便宜的模型（如 GPT-4o mini）+ 精心设计的 Prompt 建立 baseline：

```
便宜模型 + 好 Prompt ≈ 贵模型 + 普通 Prompt
```

很多场景下，Prompt 工程的 ROI 远高于升级模型。

**第二步：量化评估**

准备 50-100 条测试用例，覆盖正常 case、边界 case 和对抗 case，用统一指标评估：

```python
# 评估框架示例
evaluation_metrics = {
    "accuracy": "回答正确率（人工标注 or 自动评估）",
    "latency_p50": "50 分位响应时间",
    "latency_p99": "99 分位响应时间",
    "cost_per_call": "单次调用平均成本",
    "monthly_cost": "按预估调用量算月成本",
}
```

**第三步：分级路由**

生产环境中，单一模型很少能满足所有需求。按任务复杂度路由到不同模型：

```python
def route_model(task_complexity: str) -> str:
    """根据任务复杂度选择模型——企业级标准做法"""
    return {
        "simple": "gpt-4o-mini",      # 简单问答、分类、提取
        "medium": "gpt-4o",           # 内容生成、摘要、翻译
        "complex": "o3",              # 复杂推理、代码审查、数据分析
    }[task_complexity]
```

> **面试考点：** 面试官问"你们项目用什么模型"时，不要只说模型名字。要说清楚：为什么选它、评估了哪些替代方案、怎么做的 benchmark、成本是多少、有没有 fallback 方案。

---

## 3. 成本控制实战

### 3.1 成本计算公式

```
月成本 = 日均调用量 × (平均输入Token × 输入单价 + 平均输出Token × 输出单价) × 30
```

```python
def estimate_monthly_cost(
    calls_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """估算月成本（美元）"""
    daily = (avg_input_tokens * calls_per_day / 1_000_000) * input_price_per_m \
          + (avg_output_tokens * calls_per_day / 1_000_000) * output_price_per_m
    return round(daily * 30, 2)

# 示例：智能客服，日均 10000 次，每次 1500 输入 + 500 输出
# GPT-4o mini: $0.15/$0.60 per M tokens
print(estimate_monthly_cost(10000, 1500, 500, 0.15, 0.60))  # ~$15.75/月
```

> **注意：** 模型价格变动频繁，计算时请查阅提供商官网最新定价。推理模型的"思考 Token"也计入输出费用，实际成本可能是通用模型的 3-10 倍。

### 3.2 四大优化策略

| 策略 | 效果 | 实现难度 | 说明 |
|------|------|---------|------|
| **模型分级路由** | 节省 60-80% | 中 | 简单任务用便宜模型，复杂任务用强模型 |
| **Prompt Caching** | 节省 50-90% 输入成本 | 低 | 保持消息前缀一致即可自动触发 |
| **Batch API** | 节省 50% | 低 | 非实时任务使用批量接口 |
| **响应缓存** | 节省 90%+ | 中 | 相同/相似问题缓存结果，避免重复调用 |

---

## 4. 企业最佳实践

### 4.1 架构设计原则

**抽象 LLM 调用层**

永远不要在业务代码中直接调用某个模型的 SDK。抽象出统一接口，方便：
- 切换模型提供商（避免供应商锁定）
- A/B 测试不同模型
- 实现 fallback（主模型不可用时自动切换备用模型）
- 统一监控和计费

**多模型 Fallback 策略**

生产环境必须有 fallback。任何单一提供商都可能出现服务中断：

```python
MODEL_CHAIN = ["gpt-4o", "claude-4-sonnet", "deepseek-chat"]

async def call_with_fallback(messages: list[dict]) -> str:
    """依次尝试多个模型，直到成功"""
    for model in MODEL_CHAIN:
        try:
            return await call_llm(model, messages)
        except Exception as e:
            logger.warning(f"{model} failed: {e}")
    raise RuntimeError("All models failed")
```

### 4.2 生产环境 Checklist

| 检查项 | 说明 |
|--------|------|
| ✅ 模型调用层抽象 | 不硬编码模型名，通过配置切换 |
| ✅ Fallback 机制 | 至少准备一个备用模型/提供商 |
| ✅ 速率限制处理 | 实现指数退避重试（exponential backoff） |
| ✅ Token 用量监控 | 按用户/功能维度追踪，设置预算告警 |
| ✅ 响应质量监控 | 抽样人工评估 + 自动化指标 |
| ✅ 成本预算告警 | 日/月成本超阈值时自动通知 |
| ✅ 数据合规审查 | 确认数据传输符合隐私法规（GDPR/个保法） |
| ✅ API Key 安全 | 密钥不入代码库，使用环境变量或密钥管理服务 |

### 4.3 常见踩坑经验

**1. "最强模型"陷阱**

> 项目一上来就用最贵的模型，后来发现 80% 的请求用 mini 模型就够了，白白多花了 10 倍成本。

**教训：** 先用便宜模型建立 baseline，用数据证明需要升级再升级。

**2. 忽视延迟的代价**

> 用推理模型处理简单客服问答，用户等 8 秒才出结果，体验极差。

**教训：** 推理模型适合离线分析，不适合实时交互。实时场景优先考虑 TTFT（Time To First Token）。

**3. 供应商锁定**

> 项目深度绑定某个提供商的私有 API 特性，该提供商涨价 50% 后迁移成本巨大。

**教训：** 尽量使用 OpenAI 兼容格式，保持切换灵活性。私有特性（如特定模型的 system prompt 格式）要做适配层隔离。

**4. 忽略 Token 限制**

> 上下文窗口 128K 不代表塞满 128K 效果好。超过 32K 后，模型对中间位置信息的关注度明显下降。

**教训：** 关键信息放在输入的开头或结尾（"Lost in the Middle" 问题）。需要处理长文档时，用 RAG 而非暴力塞入上下文。

---

## 练习

1. **模型选型评估表：** 为一个假设项目（如"电商智能客服"或"代码审查助手"）设计选型评估表，包含：任务描述、五维评估、候选模型对比、最终推荐及理由。

2. **成本估算：** 假设你的项目日均 50,000 次调用，每次平均 2,000 输入 Token + 800 输出 Token，分别计算使用旗舰模型和轻量模型的月成本差异。

3. **Fallback 实现：** 基于本章的统一调用接口，实现一个带 fallback、重试和超时控制的 LLM 调用函数。

## 延伸阅读

- [LMSYS Chatbot Arena](https://chat.lmsys.org/) — 模型能力众包排名，选型必看
- [Artificial Analysis](https://artificialanalysis.ai/) — 模型性能、价格、速度的独立评测
- [OpenAI API 文档](https://platform.openai.com/docs) — API 参考和最新模型列表
- [Anthropic API 文档](https://docs.anthropic.com/) — Claude 系列文档
- [Google AI for Developers](https://ai.google.dev/) — Gemini API 文档
