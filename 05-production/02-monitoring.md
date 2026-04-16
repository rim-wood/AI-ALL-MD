# 监控与可观测性

> 让 AI 应用在生产环境中可靠运行

## 学习目标

- 建立 AI 应用的核心监控指标
- 实现幻觉检测与质量监控
- 掌握漂移检测与告警机制

---

## 1. 核心指标

AI 应用的监控指标分为三大类：性能、成本、质量。与传统 Web 服务不同，AI 应用需要额外关注输出质量和 Token 消耗。

### 1.1 性能指标

| 指标 | 定义 | 目标值（参考） |
|------|------|--------------|
| P50 延迟 | 50% 请求的响应时间 | < 2s |
| P95 延迟 | 95% 请求的响应时间 | < 5s |
| P99 延迟 | 99% 请求的响应时间 | < 10s |
| TTFT | Time To First Token，首 Token 延迟 | < 500ms |
| 吞吐量 | 每秒处理的请求数（RPS） | 视业务而定 |
| 错误率 | 失败请求占比 | < 1% |

用 Prometheus 定义指标并在 FastAPI 中采集：

```python
from prometheus_client import Histogram, Counter, Gauge
import time

# 定义指标
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM 请求延迟",
    labelnames=["model", "endpoint"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

LLM_TTFT = Histogram(
    "llm_ttft_seconds",
    "首 Token 延迟",
    labelnames=["model"],
    buckets=[0.1, 0.2, 0.5, 1, 2, 5],
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "LLM 请求错误数",
    labelnames=["model", "error_type"],
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Token 消耗总量",
    labelnames=["model", "direction"],  # direction: input/output
)

LLM_ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "当前进行中的 LLM 请求数",
    labelnames=["model"],
)


async def call_llm_with_metrics(model: str, messages: list[dict]) -> str:
    """带指标采集的 LLM 调用"""
    LLM_ACTIVE_REQUESTS.labels(model=model).inc()
    start = time.time()

    try:
        response = await client.chat.completions.create(
            model=model, messages=messages
        )
        duration = time.time() - start

        # 记录延迟
        LLM_LATENCY.labels(model=model, endpoint="chat").observe(duration)

        # 记录 Token 用量
        usage = response.usage
        LLM_TOKENS.labels(model=model, direction="input").inc(usage.prompt_tokens)
        LLM_TOKENS.labels(model=model, direction="output").inc(usage.completion_tokens)

        return response.choices[0].message.content

    except Exception as e:
        LLM_ERRORS.labels(model=model, error_type=type(e).__name__).inc()
        raise
    finally:
        LLM_ACTIVE_REQUESTS.labels(model=model).dec()
```

### 1.2 成本指标

Token 是 LLM 应用的核心成本单元。需要追踪每次调用的 Token 消耗并计算费用：

```python
from dataclasses import dataclass

# 模型定价（每百万 Token，美元）
MODEL_PRICING = {
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3-mini":     {"input": 1.10, "output": 4.40},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}

@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    user_id: str
    feature: str  # 功能模块名

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 0, "output": 0})
        return (
            self.input_tokens * pricing["input"] / 1_000_000
            + self.output_tokens * pricing["output"] / 1_000_000
        )


class CostTracker:
    """成本追踪器"""

    def __init__(self):
        self.records: list[UsageRecord] = []

    def record(self, usage: UsageRecord):
        self.records.append(usage)

        # 同时更新 Prometheus 指标
        from prometheus_client import Counter
        cost_counter = Counter(
            "llm_cost_usd_total", "累计费用（美元）",
            labelnames=["model", "feature"],
        )
        cost_counter.labels(
            model=usage.model, feature=usage.feature
        ).inc(usage.cost_usd)

    def daily_summary(self) -> dict:
        """按模型和功能汇总日成本"""
        from collections import defaultdict
        summary = defaultdict(float)
        for r in self.records:
            summary[f"{r.model}/{r.feature}"] += r.cost_usd
        return dict(summary)
```

### 1.3 质量指标

质量指标是 AI 应用特有的监控维度：

| 指标 | 度量方式 | 采集方法 |
|------|---------|---------|
| 回答准确率 | 人工标注 / LLM-as-Judge | 抽样评估 |
| 用户满意度 | 👍/👎 反馈率 | 用户交互 |
| 幻觉率 | 回答与来源不一致的比例 | 自动检测 |
| 拒答率 | 拒绝回答的比例 | 日志统计 |
| 引用准确率 | RAG 引用来源的正确率 | 自动验证 |

```python
from prometheus_client import Counter, Gauge

# 用户反馈指标
USER_FEEDBACK = Counter(
    "user_feedback_total",
    "用户反馈计数",
    labelnames=["type"],  # thumbs_up, thumbs_down, report
)

# 质量评分（滑动窗口平均）
QUALITY_SCORE = Gauge(
    "llm_quality_score",
    "输出质量评分（0-1）",
    labelnames=["dimension"],  # faithfulness, relevance, safety
)


def record_feedback(feedback_type: str, request_id: str, comment: str = ""):
    """记录用户反馈"""
    USER_FEEDBACK.labels(type=feedback_type).inc()

    # 存储详细反馈用于后续分析
    store_feedback({
        "request_id": request_id,
        "type": feedback_type,
        "comment": comment,
    })
```

---

## 2. 幻觉检测

幻觉（Hallucination）是 LLM 生成与事实不符或无中生有内容的现象。在 RAG 场景中，幻觉表现为回答与检索到的文档内容不一致。

### 2.1 检测方法

**方法一：来源验证（Source Verification）**

检查回答中的每个声明是否能在检索到的文档中找到依据：

```python
from openai import OpenAI

client = OpenAI()

def check_faithfulness(answer: str, context: str) -> dict:
    """检查回答对上下文的忠实度"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""分析以下回答中的每个事实声明，判断是否有上下文支持。

## 上下文
{context}

## 回答
{answer}

请按 JSON 格式输出：
{{
  "claims": [
    {{"claim": "具体声明", "supported": true/false, "evidence": "支持的上下文片段或null"}}
  ],
  "faithfulness_score": 0.0到1.0,
  "hallucinated_claims": ["不被支持的声明列表"]
}}"""
        }],
        response_format={"type": "json_object"},
    )

    import json
    return json.loads(response.choices[0].message.content)
```

**方法二：自我一致性检测（Self-Consistency）**

对同一问题多次采样，检查回答之间的一致性。不一致的部分更可能是幻觉：

```python
async def self_consistency_check(
    question: str, context: str, n_samples: int = 3
) -> dict:
    """通过多次采样检测幻觉"""
    responses = []
    for _ in range(n_samples):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,  # 需要一定随机性
            messages=[
                {"role": "system", "content": f"基于以下内容回答问题：\n{context}"},
                {"role": "user", "content": question},
            ],
        )
        responses.append(response.choices[0].message.content)

    # 用 LLM 分析一致性
    consistency_check = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""以下是对同一问题的 {n_samples} 个回答。
请分析它们之间的一致性，找出矛盾之处。

问题：{question}

{"".join(f"回答{i+1}：{r}" for i, r in enumerate(responses))}

输出 JSON：
{{"consistency_score": 0.0到1.0, "contradictions": ["矛盾点列表"], "reliable_facts": ["一致的事实列表"]}}"""
        }],
        response_format={"type": "json_object"},
    )

    import json
    return json.loads(consistency_check.choices[0].message.content)
```

### 2.2 实时检测

在生产环境中，对每个回答进行实时幻觉检测：

```python
import asyncio
from dataclasses import dataclass

@dataclass
class QualityCheckResult:
    faithfulness: float
    passed: bool
    hallucinated_claims: list[str]

class RealtimeHallucinationDetector:
    """实时幻觉检测管道"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    async def check(self, answer: str, context: str) -> QualityCheckResult:
        """检测回答中的幻觉"""
        result = check_faithfulness(answer, context)

        score = result["faithfulness_score"]
        passed = score >= self.threshold

        if not passed:
            # 记录告警
            LLM_ERRORS.labels(
                model="quality_check", error_type="hallucination"
            ).inc()

        QUALITY_SCORE.labels(dimension="faithfulness").set(score)

        return QualityCheckResult(
            faithfulness=score,
            passed=passed,
            hallucinated_claims=result.get("hallucinated_claims", []),
        )


# 集成到请求处理流程
async def handle_request(question: str, context: str) -> dict:
    """带幻觉检测的请求处理"""
    answer = await call_llm(question, context)

    detector = RealtimeHallucinationDetector(threshold=0.8)
    quality = await detector.check(answer, context)

    if not quality.passed:
        # 策略：添加免责声明 / 降级到保守回答 / 拒绝回答
        answer = f"{answer}\n\n⚠️ 此回答的置信度较低，建议核实相关信息。"

    return {
        "answer": answer,
        "quality": {"faithfulness": quality.faithfulness, "passed": quality.passed},
    }
```

### 2.3 离线分析

定期对历史数据进行批量评估，发现系统性问题：

```python
import json
from pathlib import Path

async def batch_evaluate(log_dir: str, sample_size: int = 200):
    """批量评估历史回答的质量"""
    log_path = Path(log_dir)
    logs = list(log_path.glob("*.json"))

    # 随机抽样
    import random
    sampled = random.sample(logs, min(sample_size, len(logs)))

    results = {"total": len(sampled), "passed": 0, "failed": 0, "scores": []}

    for log_file in sampled:
        record = json.loads(log_file.read_text())
        result = check_faithfulness(record["answer"], record["context"])
        score = result["faithfulness_score"]
        results["scores"].append(score)

        if score >= 0.8:
            results["passed"] += 1
        else:
            results["failed"] += 1

    import numpy as np
    scores = np.array(results["scores"])
    results["summary"] = {
        "mean": round(float(scores.mean()), 3),
        "p50": round(float(np.percentile(scores, 50)), 3),
        "p10": round(float(np.percentile(scores, 10)), 3),
        "pass_rate": round(results["passed"] / results["total"], 3),
    }

    return results
```

---

## 3. 漂移检测

漂移（Drift）是指系统行为随时间逐渐偏离预期的现象。LLM 应用面临三种漂移。

### 3.1 Prompt Drift

提示词效果随时间变化，通常由模型更新或用户行为变化引起。

检测方法：定期在固定数据集上运行评估，追踪分数趋势。

```python
import json
from datetime import datetime
from pathlib import Path

class PromptDriftDetector:
    """提示词漂移检测"""

    def __init__(self, baseline_file: str = "baseline_scores.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> dict:
        if self.baseline_file.exists():
            return json.loads(self.baseline_file.read_text())
        return {}

    def run_benchmark(self, eval_fn, test_cases: list[dict]) -> dict:
        """在固定数据集上运行评估"""
        scores = []
        for case in test_cases:
            result = eval_fn(case["input"])
            score = case["scorer"](result, case["expected"])
            scores.append(score)

        import numpy as np
        current = {
            "timestamp": datetime.now().isoformat(),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_cases": len(scores),
        }
        return current

    def check_drift(self, current: dict, threshold: float = 0.05) -> dict:
        """对比当前分数与基线，检测漂移"""
        if not self.baseline:
            return {"drifted": False, "message": "无基线数据，当前结果将作为基线"}

        baseline_mean = self.baseline["mean_score"]
        current_mean = current["mean_score"]
        delta = current_mean - baseline_mean

        drifted = abs(delta) > threshold
        return {
            "drifted": drifted,
            "baseline_score": baseline_mean,
            "current_score": current_mean,
            "delta": round(delta, 4),
            "message": f"分数{'下降' if delta < 0 else '上升'} {abs(delta):.4f}"
            if drifted else "未检测到显著漂移",
        }
```

### 3.2 数据漂移

用户输入的分布随时间变化，可能导致系统在新类型的输入上表现不佳。

```python
from collections import Counter
import numpy as np

class InputDriftDetector:
    """用户输入分布漂移检测"""

    def __init__(self):
        self.baseline_distribution: dict | None = None

    def compute_distribution(self, texts: list[str]) -> dict:
        """计算输入文本的特征分布"""
        lengths = [len(t) for t in texts]

        # 简单的主题分类（生产中可用嵌入聚类）
        topic_keywords = {
            "退货": ["退货", "退款", "退回"],
            "物流": ["快递", "物流", "配送", "发货"],
            "支付": ["支付", "付款", "价格", "优惠"],
            "账户": ["账号", "密码", "登录", "注册"],
        }
        topic_counts = Counter()
        for text in texts:
            for topic, keywords in topic_keywords.items():
                if any(kw in text for kw in keywords):
                    topic_counts[topic] += 1
                    break
            else:
                topic_counts["其他"] += 1

        total = len(texts)
        return {
            "avg_length": np.mean(lengths),
            "topic_distribution": {k: v / total for k, v in topic_counts.items()},
            "sample_size": total,
        }

    def detect_drift(self, current_texts: list[str], threshold: float = 0.1) -> dict:
        """检测输入分布是否发生漂移"""
        current = self.compute_distribution(current_texts)

        if self.baseline_distribution is None:
            self.baseline_distribution = current
            return {"drifted": False, "message": "已设置基线"}

        # 对比主题分布（简化版 KL 散度）
        baseline_topics = self.baseline_distribution["topic_distribution"]
        current_topics = current["topic_distribution"]

        all_topics = set(baseline_topics) | set(current_topics)
        max_shift = 0
        shifts = {}
        for topic in all_topics:
            b = baseline_topics.get(topic, 0)
            c = current_topics.get(topic, 0)
            shift = abs(c - b)
            shifts[topic] = round(shift, 3)
            max_shift = max(max_shift, shift)

        return {
            "drifted": max_shift > threshold,
            "max_shift": round(max_shift, 3),
            "topic_shifts": shifts,
        }
```

### 3.3 模型漂移

当 LLM 提供商更新模型时（如 OpenAI 的模型升级），即使提示词不变，输出行为也可能改变。

检测策略：

```python
class ModelDriftDetector:
    """模型行为漂移检测"""

    def __init__(self, golden_dataset: list[dict]):
        """
        golden_dataset: [{"input": ..., "expected_output": ..., "scorer": ...}]
        """
        self.golden_dataset = golden_dataset

    async def snapshot(self, model: str) -> dict:
        """对模型行为做快照"""
        results = []
        for case in self.golden_dataset:
            output = await call_llm(case["input"], model=model)
            score = case["scorer"](output, case["expected_output"])
            results.append({
                "input": case["input"],
                "output": output,
                "score": score,
            })

        scores = [r["score"] for r in results]
        return {
            "model": model,
            "mean_score": np.mean(scores),
            "results": results,
        }

    def compare_snapshots(self, before: dict, after: dict) -> dict:
        """对比两次快照"""
        delta = after["mean_score"] - before["mean_score"]

        # 找出分数变化最大的用例
        regressions = []
        for b, a in zip(before["results"], after["results"]):
            score_diff = a["score"] - b["score"]
            if score_diff < -0.2:  # 分数下降超过 0.2
                regressions.append({
                    "input": b["input"],
                    "before_score": b["score"],
                    "after_score": a["score"],
                })

        return {
            "score_delta": round(delta, 4),
            "regressions": regressions,
            "regression_count": len(regressions),
        }
```

> **最佳实践**：在模型提供商宣布更新后，立即在 golden dataset 上运行快照对比，确认行为变化在可接受范围内。

---

## 4. 可观测性工具

### 4.1 LangFuse

LangFuse 是一个开源的 LLM 可观测性平台，支持追踪、评估和分析。

```python
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

langfuse = Langfuse()

@observe()
def rag_pipeline(question: str) -> str:
    """被追踪的 RAG 管道"""
    # 检索阶段 — 自动记录为子 Span
    docs = retrieve_documents(question)

    # 生成阶段
    answer = generate_answer(question, docs)

    # 记录自定义指标
    langfuse_context.update_current_observation(
        metadata={"doc_count": len(docs)},
    )
    langfuse_context.score_current_trace(
        name="user_feedback",
        value=1,  # 后续由用户反馈更新
    )

    return answer

@observe()
def retrieve_documents(question: str) -> list[str]:
    """检索文档 — 自动作为子 Span 追踪"""
    # 向量搜索逻辑
    results = vector_store.similarity_search(question, k=5)
    return [doc.page_content for doc in results]

@observe()
def generate_answer(question: str, docs: list[str]) -> str:
    """生成回答 — 自动追踪 LLM 调用"""
    context = "\n".join(docs)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"基于以下内容回答：\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
```

LangFuse 会自动记录每个被 `@observe()` 装饰的函数的输入、输出、延迟和嵌套关系，在 Web UI 中展示完整的调用链。

### 4.2 Helicone

Helicone 采用 API 代理模式，只需修改 base URL 即可接入，无需改动业务代码：

```python
from openai import OpenAI

# 只需修改 base_url，所有请求自动经过 Helicone 代理
client = OpenAI(
    base_url="https://oai.helicone.ai/v1",
    default_headers={
        "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
        # 自定义属性，用于筛选和分析
        "Helicone-Property-Feature": "customer-support",
        "Helicone-Property-Environment": "production",
        "Helicone-User-Id": "user_123",
        # 启用缓存
        "Helicone-Cache-Enabled": "true",
    },
)

# 正常调用，Helicone 自动记录请求/响应/延迟/Token
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}],
)
```

Helicone 的优势是**零侵入**：不需要修改任何业务逻辑，只改一行 base URL。它自动记录所有 LLM 调用的延迟、Token 用量、成本，并提供 Web 仪表盘。

### 4.3 Phoenix (Arize)

Phoenix 是 Arize AI 开源的可观测性工具，特别擅长嵌入向量的可视化分析：

```python
import phoenix as px
from phoenix.trace.openai import OpenAIInstrumentor

# 启动 Phoenix 服务
session = px.launch_app()

# 自动追踪 OpenAI 调用
OpenAIInstrumentor().instrument()

# 之后所有 OpenAI 调用自动被追踪
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}],
)

# 在 Phoenix UI 中可以：
# - 查看每次调用的输入/输出/延迟/Token
# - 可视化嵌入向量的分布（UMAP 降维）
# - 检测嵌入漂移
# - 运行在线评估
print(f"Phoenix UI: {session.url}")
```

### 4.4 自建方案

如果需要完全控制数据，可以基于 OpenTelemetry 构建自己的追踪系统：

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化 OpenTelemetry
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-app")


async def traced_llm_call(model: str, messages: list[dict]) -> str:
    """带 OpenTelemetry 追踪的 LLM 调用"""
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.message_count", len(messages))

        try:
            response = await client.chat.completions.create(
                model=model, messages=messages
            )

            # 记录 Token 用量
            usage = response.usage
            span.set_attribute("llm.input_tokens", usage.prompt_tokens)
            span.set_attribute("llm.output_tokens", usage.completion_tokens)
            span.set_attribute("llm.total_tokens", usage.total_tokens)

            content = response.choices[0].message.content
            span.set_attribute("llm.response_length", len(content))

            return content

        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
```

工具选型对比：

| 工具 | 部署方式 | 核心优势 | 适用场景 |
|------|---------|---------|---------|
| LangFuse | 自托管/云 | 开源、功能全面 | 需要完整可观测性 |
| Helicone | 云服务 | 零侵入接入 | 快速接入、不想改代码 |
| Phoenix | 本地/自托管 | 嵌入可视化 | RAG 调试、嵌入分析 |
| OpenTelemetry | 自建 | 完全可控 | 已有 OTEL 基础设施 |

---

## 5. 日志与追踪

### 5.1 结构化日志

LLM 应用的日志需要包含比传统应用更多的信息：

```python
import json
import logging
import uuid
from datetime import datetime

class LLMLogger:
    """LLM 应用结构化日志"""

    def __init__(self):
        self.logger = logging.getLogger("llm_app")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_request(
        self,
        request_id: str,
        model: str,
        messages: list[dict],
        response: str,
        usage: dict,
        latency_ms: float,
        metadata: dict | None = None,
    ):
        """记录一次 LLM 请求的完整信息"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "event": "llm_request",
            "model": model,
            "input": {
                "messages": messages,
                "message_count": len(messages),
                "total_chars": sum(len(m["content"]) for m in messages),
            },
            "output": {
                "response": response[:500],  # 截断避免日志过大
                "response_length": len(response),
            },
            "usage": usage,
            "latency_ms": round(latency_ms, 2),
            "metadata": metadata or {},
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

llm_logger = LLMLogger()
```

### 5.2 链路追踪

一个完整的 RAG 请求涉及多个阶段，需要端到端的链路追踪：

```
用户请求 (trace_id: abc123)
├── [Span] 输入预处理        12ms
├── [Span] 查询改写          450ms
│   └── [Span] LLM 调用      420ms
├── [Span] 向量检索          85ms
│   ├── [Span] Embedding     35ms
│   └── [Span] ANN 搜索      50ms
├── [Span] 重排序            120ms
├── [Span] 回答生成          1800ms
│   └── [Span] LLM 调用      1750ms
└── [Span] 质量检测          300ms
    └── [Span] LLM 调用      280ms
总耗时: 2767ms
```

```python
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None
    start_time: float = 0
    end_time: float = 0
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

class SimpleTracer:
    """轻量级链路追踪"""

    def __init__(self):
        self.spans: list[Span] = []
        self._current_trace_id: str | None = None
        self._current_span: Span | None = None

    @contextmanager
    def trace(self, name: str):
        """开始一个新的 Trace"""
        self._current_trace_id = uuid.uuid4().hex[:16]
        span = Span(name=name, trace_id=self._current_trace_id)
        span.start_time = time.time()
        self._current_span = span
        try:
            yield span
        finally:
            span.end_time = time.time()
            self.spans.append(span)

    @contextmanager
    def span(self, name: str):
        """在当前 Trace 中创建子 Span"""
        parent = self._current_span
        child = Span(
            name=name,
            trace_id=self._current_trace_id,
            parent_id=parent.span_id if parent else None,
        )
        child.start_time = time.time()
        self._current_span = child
        try:
            yield child
        finally:
            child.end_time = time.time()
            self.spans.append(child)
            self._current_span = parent

# 使用示例
tracer = SimpleTracer()

with tracer.trace("rag_request") as root:
    with tracer.span("retrieve") as s:
        docs = retrieve(question)
        s.attributes["doc_count"] = len(docs)

    with tracer.span("generate") as s:
        answer = generate(question, docs)
        s.attributes["model"] = "gpt-4o"
```

### 5.3 日志存储与查询

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| ELK (Elasticsearch) | 全文搜索强、生态成熟 | 资源消耗大、运维复杂 | 已有 ELK 基础设施 |
| ClickHouse | 列式存储、聚合查询快 | 全文搜索弱 | 大量指标分析 |
| Loki + Grafana | 轻量、与 Grafana 集成好 | 查询能力有限 | 中小规模、已用 Grafana |

推荐架构：

```
应用日志 → Fluent Bit → ClickHouse（指标/聚合）
                     → Elasticsearch（全文搜索/调试）
                     → S3（长期归档）
```

---

## 6. 告警与响应

### 6.1 告警规则

为 AI 应用配置的关键告警：

| 告警名称 | 条件 | 严重级别 | 响应动作 |
|---------|------|---------|---------|
| 高延迟 | P95 > 10s 持续 5 分钟 | Warning | 检查模型服务状态 |
| 高错误率 | 错误率 > 5% 持续 3 分钟 | Critical | 启动降级方案 |
| 幻觉率飙升 | 幻觉率 > 15% 持续 10 分钟 | Critical | 切换到保守提示词 |
| 成本异常 | 小时成本 > 日均 3 倍 | Warning | 检查是否有异常流量 |
| Token 用量激增 | 平均 Token > 基线 2 倍 | Warning | 检查输入是否异常 |

Prometheus 告警规则示例：

```yaml
# prometheus/alerts.yaml
groups:
  - name: llm_alerts
    rules:
      - alert: LLMHighLatency
        expr: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM P95 延迟超过 10 秒"

      - alert: LLMHighErrorRate
        expr: rate(llm_errors_total[5m]) / rate(llm_request_duration_seconds_count[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "LLM 错误率超过 5%"

      - alert: LLMCostSpike
        expr: increase(llm_cost_usd_total[1h]) > 3 * avg_over_time(increase(llm_cost_usd_total[1h])[24h:1h])
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "LLM 小时成本异常，超过日均 3 倍"
```

### 6.2 事件响应流程

```
告警触发
  │
  ├─ P0 (Critical): 服务不可用 / 大面积幻觉
  │   → 立即启动降级方案
  │   → 通知 On-call 工程师（电话）
  │   → 15 分钟内响应
  │
  ├─ P1 (High): 性能严重下降 / 成本异常
  │   → 切换到备用模型
  │   → 通知团队（即时消息）
  │   → 1 小时内响应
  │
  └─ P2 (Warning): 指标轻微异常
      → 记录工单
      → 下个工作日处理
```

Runbook 模板（以幻觉率飙升为例）：

```markdown
## Runbook: 幻觉率飙升

### 触发条件
幻觉率 > 15% 持续 10 分钟

### 排查步骤
1. 检查模型提供商状态页，确认是否有模型更新或故障
2. 查看最近的提示词变更记录
3. 抽样检查最近的请求日志，分析幻觉模式
4. 检查 RAG 检索质量，确认知识库是否正常

### 缓解措施
1. 切换到上一个稳定版本的提示词
2. 如果是模型问题，切换到备用模型
3. 临时提高幻觉检测阈值，对低置信度回答添加免责声明
4. 必要时启用保守模式（只返回检索到的原文，不做生成）

### 恢复确认
- 幻觉率回到 < 10%
- 用户反馈无异常
- 持续观察 30 分钟
```

---

## 练习

1. 用 LangFuse 为一个 RAG 应用添加追踪
2. 实现一个幻觉检测管道
3. 设计告警规则并配置通知

## 延伸阅读

- [LangFuse 文档](https://langfuse.com/docs)
- [Helicone 文档](https://docs.helicone.ai/)
- [Phoenix 文档](https://docs.arize.com/phoenix)
