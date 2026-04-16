# 性能优化与成本控制

> 让 AI 应用更快、更省、更稳

## 学习目标

- 掌握语义缓存与模型路由策略
- 了解推理优化与自托管方案
- 建立成本监控与预算管理

---

## 1. 缓存策略

LLM 调用是 AI 应用中最昂贵的操作。缓存可以显著降低成本和延迟。

### 1.1 精确缓存

对完全相同的输入直接返回缓存结果：

```python
import hashlib
import json
import redis

class ExactCache:
    """精确匹配缓存"""

    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl

    def _make_key(self, model: str, messages: list[dict], **kwargs) -> str:
        """生成缓存键"""
        payload = json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True)
        return f"llm:exact:{hashlib.sha256(payload.encode()).hexdigest()}"

    def get(self, model: str, messages: list[dict], **kwargs) -> str | None:
        """查询缓存"""
        key = self._make_key(model, messages, **kwargs)
        cached = self.redis.get(key)
        return cached.decode() if cached else None

    def set(self, model: str, messages: list[dict], response: str, ttl: int | None = None, **kwargs):
        """写入缓存"""
        key = self._make_key(model, messages, **kwargs)
        self.redis.setex(key, ttl or self.default_ttl, response)

    async def cached_call(self, model: str, messages: list[dict], **kwargs) -> tuple[str, bool]:
        """带缓存的 LLM 调用，返回 (response, from_cache)"""
        cached = self.get(model, messages, **kwargs)
        if cached:
            return cached, True

        response = await client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        content = response.choices[0].message.content
        self.set(model, messages, content, **kwargs)
        return content, False
```

### 1.2 语义缓存

精确缓存要求输入完全一致，命中率有限。语义缓存通过向量相似度匹配"意思相近"的查询：

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

class SemanticCache:
    """基于向量相似度的语义缓存"""

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.cache: list[dict] = []  # 生产中用向量数据库

    def _get_embedding(self, text: str) -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> str | None:
        """查找语义相似的缓存"""
        query_emb = self._get_embedding(query)

        best_match = None
        best_score = 0

        for entry in self.cache:
            score = self._cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= self.threshold:
            return best_match["response"]
        return None

    def set(self, query: str, response: str):
        """添加到缓存"""
        embedding = self._get_embedding(query)
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "response": response,
        })
```

生产环境中推荐使用 Redis 的向量搜索功能：

```python
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class RedisSemanticCache:
    """基于 Redis Vector Search 的语义缓存"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.index_name = "llm_cache"
        self._create_index()

    def _create_index(self):
        """创建向量索引"""
        try:
            self.redis.ft(self.index_name).info()
        except redis.ResponseError:
            schema = (
                TextField("query"),
                TextField("response"),
                VectorField(
                    "embedding", "HNSW",
                    {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"},
                ),
            )
            self.redis.ft(self.index_name).create_index(
                schema, definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            )

    def search(self, query_embedding: list[float], threshold: float = 0.92) -> str | None:
        """向量搜索最相似的缓存"""
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        q = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .return_fields("response", "score")
            .dialect(2)
        )
        results = self.redis.ft(self.index_name).search(q, query_params={"vec": query_vector})

        if results.docs:
            doc = results.docs[0]
            similarity = 1 - float(doc.score)  # Redis 返回距离，转换为相似度
            if similarity >= threshold:
                return doc.response
        return None
```

### 1.3 缓存失效

缓存失效策略决定了数据新鲜度和命中率的平衡：

| 策略 | 适用场景 | 实现方式 |
|------|---------|---------|
| TTL 过期 | 通用场景 | 设置固定过期时间 |
| 版本化缓存 | 知识库更新 | 缓存键包含知识库版本号 |
| 主动失效 | 数据变更时 | 更新数据时清除相关缓存 |
| LRU 淘汰 | 内存有限 | 淘汰最久未使用的缓存 |

```python
class VersionedCache:
    """版本化缓存 — 知识库更新时自动失效"""

    def __init__(self, redis_client):
        self.redis = redis_client

    def _make_key(self, query: str, kb_version: str) -> str:
        """缓存键包含知识库版本"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"llm:v{kb_version}:{query_hash}"

    def get(self, query: str, kb_version: str) -> str | None:
        cached = self.redis.get(self._make_key(query, kb_version))
        return cached.decode() if cached else None

    def set(self, query: str, kb_version: str, response: str, ttl: int = 3600):
        self.redis.setex(self._make_key(query, kb_version), ttl, response)

    def invalidate_version(self, kb_version: str):
        """清除指定版本的所有缓存"""
        pattern = f"llm:v{kb_version}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

---

## 2. 模型路由

### 2.1 大小模型分流

不是所有请求都需要最强的模型。简单问题用小模型，复杂问题用大模型，可以大幅降低成本：

```python
from dataclasses import dataclass

@dataclass
class RouteResult:
    model: str
    reason: str

class ModelRouter:
    """大小模型智能路由"""

    # 模型配置
    MODELS = {
        "small": {"name": "gpt-4o-mini", "cost_per_1k": 0.00015},
        "large": {"name": "gpt-4o", "cost_per_1k": 0.0025},
    }

    def route(self, query: str, context: dict | None = None) -> RouteResult:
        """根据查询复杂度选择模型"""
        complexity = self._assess_complexity(query)

        if complexity == "simple":
            return RouteResult(model="gpt-4o-mini", reason="简单查询，使用小模型")
        else:
            return RouteResult(model="gpt-4o", reason="复杂查询，使用大模型")

    def _assess_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        # 简单规则（生产中可用分类模型）
        simple_indicators = ["是什么", "怎么样", "多少钱", "在哪里", "营业时间"]
        complex_indicators = ["为什么", "对比", "分析", "建议", "如何优化", "区别"]

        query_lower = query.lower()

        if any(ind in query_lower for ind in complex_indicators):
            return "complex"
        if any(ind in query_lower for ind in simple_indicators):
            return "simple"
        if len(query) > 200:
            return "complex"

        return "simple"
```

### 2.2 路由策略

更精确的路由可以使用 LLM 分类器：

```python
class LLMBasedRouter:
    """基于 LLM 分类的模型路由"""

    async def route(self, query: str) -> RouteResult:
        """用小模型判断查询复杂度，决定用哪个模型回答"""
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"""将以下查询分类为 simple 或 complex。

simple：事实查询、简单问答、格式转换
complex：需要推理、分析、创作、多步骤任务

查询：{query}

只输出一个词：simple 或 complex""",
            }],
        )

        complexity = response.choices[0].message.content.strip().lower()
        model = "gpt-4o-mini" if complexity == "simple" else "gpt-4o"
        return RouteResult(model=model, reason=f"分类结果: {complexity}")
```

### 2.3 降级方案

当主模型不可用时，自动切换到备用模型：

```python
import asyncio
from openai import OpenAI, APIError, RateLimitError

class ResilientLLMClient:
    """带降级能力的 LLM 客户端"""

    def __init__(self):
        self.providers = [
            {"name": "openai", "client": OpenAI(), "model": "gpt-4o"},
            {"name": "openai_mini", "client": OpenAI(), "model": "gpt-4o-mini"},
        ]

    async def call(self, messages: list[dict], timeout: float = 30) -> dict:
        """依次尝试各提供商，直到成功"""
        last_error = None

        for provider in self.providers:
            try:
                response = await asyncio.wait_for(
                    self._call_provider(provider, messages),
                    timeout=timeout,
                )
                return {
                    "content": response,
                    "provider": provider["name"],
                    "model": provider["model"],
                    "degraded": provider != self.providers[0],
                }
            except (APIError, RateLimitError, asyncio.TimeoutError) as e:
                last_error = e
                continue

        raise RuntimeError(f"所有模型均不可用，最后错误: {last_error}")

    async def _call_provider(self, provider: dict, messages: list[dict]) -> str:
        response = provider["client"].chat.completions.create(
            model=provider["model"], messages=messages
        )
        return response.choices[0].message.content
```

---

## 3. Token 优化

### 3.1 上下文压缩

RAG 检索到的文档往往包含大量冗余信息。压缩上下文可以减少 Token 消耗：

```python
async def compress_context(documents: list[str], question: str, max_tokens: int = 1000) -> str:
    """压缩检索到的文档，只保留与问题相关的信息"""
    full_context = "\n\n---\n\n".join(documents)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""从以下文档中提取与问题相关的关键信息，去除无关内容。
保持事实准确，不要添加文档中没有的信息。
输出控制在 {max_tokens} 个 Token 以内。

问题：{question}

文档：
{full_context}

提取的关键信息：""",
        }],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

### 3.2 Prompt 优化

精简提示词可以在不影响质量的前提下减少 Token 消耗：

```
# ❌ 冗余的提示词（约 150 tokens）
你是一个非常专业的、经验丰富的客户服务代表。你的主要职责是帮助客户
解决他们遇到的各种各样的问题。在回答问题的时候，你需要确保你的回答
是准确的、有帮助的、并且是友好的。你应该基于提供给你的知识库内容来
回答问题。如果知识库中没有相关的信息，你应该诚实地告诉客户你不知道
答案，而不是编造一个答案。

# ✅ 精简的提示词（约 60 tokens）
你是客服助手。基于知识库回答问题。
规则：1）只用知识库内容回答 2）不知道就说不知道 3）简洁友好
```

Token 节省技巧：

| 技巧 | 节省比例 | 说明 |
|------|---------|------|
| 精简系统提示词 | 30-60% | 去除冗余描述，保留核心指令 |
| 压缩检索上下文 | 40-70% | 只保留与问题相关的片段 |
| 限制对话历史 | 20-50% | 只保留最近 N 轮或做摘要 |
| 结构化输出 | 10-30% | JSON 比自然语言更紧凑 |

### 3.3 输出控制

```python
# 限制输出长度
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=256,          # 限制最大输出 Token
    stop=["\n\n---", "参考资料："],  # 遇到停止序列时截断
)

# 使用结构化输出减少冗余
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "用 JSON 格式回答：这个产品能退货吗？",
    }],
    response_format={"type": "json_object"},
    max_tokens=100,
)
# 输出: {"can_return": true, "period": "7天", "condition": "商品完好"}
# 比自然语言回答节省约 50% Token
```

---

## 4. 批量处理

### 4.1 Batch API

OpenAI Batch API 提供 50% 的价格折扣，适合非实时场景：

```python
import json
from openai import OpenAI
from pathlib import Path

client = OpenAI()

def create_batch_file(requests: list[dict], output_path: str) -> str:
    """创建 Batch API 输入文件（JSONL 格式）"""
    with open(output_path, "w") as f:
        for i, req in enumerate(requests):
            line = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req.get("model", "gpt-4o-mini"),
                    "messages": req["messages"],
                    "max_tokens": req.get("max_tokens", 1024),
                },
            }
            f.write(json.dumps(line) + "\n")
    return output_path

def submit_batch(input_file: str) -> str:
    """提交批量任务"""
    # 上传文件
    with open(input_file, "rb") as f:
        file = client.files.create(file=f, purpose="batch")

    # 创建批量任务
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch ID: {batch.id}, Status: {batch.status}")
    return batch.id

def check_and_download(batch_id: str) -> list[dict] | None:
    """检查批量任务状态并下载结果"""
    batch = client.batches.retrieve(batch_id)
    print(f"Status: {batch.status}, Progress: {batch.request_counts}")

    if batch.status == "completed":
        content = client.files.content(batch.output_file_id)
        results = []
        for line in content.text.strip().split("\n"):
            results.append(json.loads(line))
        return results
    return None

# 使用示例
requests = [
    {"messages": [{"role": "user", "content": f"总结第{i}章"}]}
    for i in range(100)
]
create_batch_file(requests, "batch_input.jsonl")
batch_id = submit_batch("batch_input.jsonl")
# 等待完成后下载结果
# results = check_and_download(batch_id)
```

成本对比：

| 方式 | GPT-4o 输入价格 | GPT-4o 输出价格 | 延迟 |
|------|---------------|---------------|------|
| 实时 API | $2.50/M tokens | $10.00/M tokens | 秒级 |
| Batch API | $1.25/M tokens | $5.00/M tokens | 24h 内 |
| **节省** | **50%** | **50%** | — |

### 4.2 队列设计

对于高并发场景，使用任务队列管理 LLM 请求：

```python
import asyncio
from dataclasses import dataclass, field
from enum import Enum

class Priority(Enum):
    HIGH = 0    # VIP 用户、实时对话
    NORMAL = 1  # 普通请求
    LOW = 2     # 后台任务、批量处理

@dataclass(order=True)
class LLMTask:
    priority: int
    task_id: str = field(compare=False)
    messages: list[dict] = field(compare=False)
    model: str = field(compare=False, default="gpt-4o-mini")

class LLMTaskQueue:
    """带优先级和限流的 LLM 任务队列"""

    def __init__(self, max_concurrent: int = 10, rpm_limit: int = 500):
        self.queue = asyncio.PriorityQueue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rpm_limit = rpm_limit
        self._request_count = 0

    async def submit(self, task: LLMTask) -> str:
        """提交任务"""
        await self.queue.put(task)
        return task.task_id

    async def worker(self):
        """工作协程，从队列取任务执行"""
        while True:
            task = await self.queue.get()
            async with self.semaphore:
                try:
                    result = await self._execute(task)
                    await self._store_result(task.task_id, result)
                except Exception as e:
                    await self._store_result(task.task_id, {"error": str(e)})
                finally:
                    self.queue.task_done()

    async def _execute(self, task: LLMTask) -> str:
        """执行 LLM 调用（带限流）"""
        # 简单的令牌桶限流
        while self._request_count >= self.rpm_limit:
            await asyncio.sleep(0.1)

        self._request_count += 1
        try:
            response = await client.chat.completions.create(
                model=task.model, messages=task.messages
            )
            return response.choices[0].message.content
        finally:
            # 1 分钟后释放计数
            asyncio.get_event_loop().call_later(60, self._release_count)

    def _release_count(self):
        self._request_count = max(0, self._request_count - 1)

    async def _store_result(self, task_id: str, result):
        """存储结果（实际中用 Redis/DB）"""
        pass
```

---

## 5. 自托管推理

### 5.1 推理引擎

| 引擎 | 核心优势 | 适用场景 | 吞吐量 |
|------|---------|---------|--------|
| vLLM | PagedAttention、高吞吐 | 生产服务、高并发 | 最高 |
| TGI | HuggingFace 生态、易部署 | HF 模型快速部署 | 高 |
| Ollama | 极简部署、本地开发 | 开发测试、个人使用 | 中 |
| llama.cpp | CPU 推理、GGUF 格式 | 无 GPU 环境 | 低 |

**vLLM 部署示例：**

```bash
# 安装
pip install vllm

# 启动 OpenAI 兼容的 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8000
```

```python
# 客户端调用（与 OpenAI API 完全兼容）
from openai import OpenAI

local_client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = local_client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    messages=[{"role": "user", "content": "你好"}],
)
```

**Ollama 部署示例：**

```bash
# 安装后一行命令启动
ollama run qwen2.5:32b

# 或以服务模式运行
ollama serve
```

```python
# Ollama 也兼容 OpenAI API
local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = local_client.chat.completions.create(
    model="qwen2.5:32b",
    messages=[{"role": "user", "content": "你好"}],
)
```

### 5.2 量化

量化通过降低模型权重的精度来减少显存占用和提升推理速度：

| 量化方法 | 精度 | 显存节省 | 质量损失 | 推理速度 |
|---------|------|---------|---------|---------|
| FP16 | 16-bit | 基准 | 无 | 基准 |
| GPTQ | 4-bit | ~75% | 极小 | 快 |
| AWQ | 4-bit | ~75% | 极小 | 最快 |
| GGUF | 2-8 bit | 50-87% | 小-中 | 中（支持 CPU） |

以 72B 参数模型为例：

| 精度 | 显存需求 | 所需 GPU |
|------|---------|---------|
| FP16 | ~144 GB | 2× A100 80GB |
| 4-bit (AWQ) | ~36 GB | 1× A100 40GB |
| GGUF Q4 | ~36 GB | CPU 可运行（慢） |

### 5.3 部署方案

**GPU 选型指南：**

| GPU | 显存 | 适合模型 | 参考价格（云） |
|-----|------|---------|-------------|
| T4 | 16 GB | 7B 量化模型 | ~$0.5/h |
| A10G | 24 GB | 7-14B 模型 | ~$1.0/h |
| L4 | 24 GB | 7-14B 模型 | ~$0.8/h |
| A100 40GB | 40 GB | 70B 量化模型 | ~$3.5/h |
| A100 80GB | 80 GB | 70B FP16 | ~$5.0/h |
| H100 | 80 GB | 70B+ 高吞吐 | ~$8.0/h |

**Docker 容器化部署：**

```dockerfile
# Dockerfile.vllm
FROM vllm/vllm-openai:latest

ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV MAX_MODEL_LEN=8192

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_NAME}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--port", "8000"]
```

```yaml
# docker-compose.yaml
services:
  vllm:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
      - MAX_MODEL_LEN=8192
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Kubernetes 自动扩缩：**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Pods
      pods:
        metric:
          name: llm_active_requests
        target:
          type: AverageValue
          averageValue: "8"  # 每个 Pod 平均 8 个并发请求时扩容
```

---

## 6. 成本管理

### 6.1 成本监控

实时追踪 LLM 使用成本，按用户、功能、团队归因：

```python
from collections import defaultdict
from datetime import datetime, timedelta

class CostDashboard:
    """成本监控仪表盘"""

    def __init__(self):
        self.records: list[dict] = []

    def record_usage(
        self, model: str, input_tokens: int, output_tokens: int,
        user_id: str, feature: str, team: str,
    ):
        """记录一次使用"""
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        self.records.append({
            "timestamp": datetime.now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "user_id": user_id,
            "feature": feature,
            "team": team,
        })

    def summary(self, hours: int = 24) -> dict:
        """生成成本摘要"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.records if r["timestamp"] > cutoff]

        by_model = defaultdict(float)
        by_feature = defaultdict(float)
        by_team = defaultdict(float)
        total_tokens = {"input": 0, "output": 0}

        for r in recent:
            by_model[r["model"]] += r["cost_usd"]
            by_feature[r["feature"]] += r["cost_usd"]
            by_team[r["team"]] += r["cost_usd"]
            total_tokens["input"] += r["input_tokens"]
            total_tokens["output"] += r["output_tokens"]

        return {
            "period_hours": hours,
            "total_cost_usd": round(sum(r["cost_usd"] for r in recent), 4),
            "total_requests": len(recent),
            "total_tokens": total_tokens,
            "by_model": dict(by_model),
            "by_feature": dict(by_feature),
            "by_team": dict(by_team),
        }
```

### 6.2 预算告警

```python
class BudgetGuard:
    """预算守卫 — 超预算时自动限流"""

    def __init__(self, daily_budget_usd: float = 100.0, warning_threshold: float = 0.8):
        self.daily_budget = daily_budget_usd
        self.warning_threshold = warning_threshold
        self.today_cost = 0.0
        self.limited = False

    def add_cost(self, cost_usd: float):
        self.today_cost += cost_usd

        usage_ratio = self.today_cost / self.daily_budget

        if usage_ratio >= 1.0 and not self.limited:
            self.limited = True
            self._trigger_alert("critical", f"日预算已用完: ${self.today_cost:.2f}/${self.daily_budget}")
            self._enable_rate_limit()

        elif usage_ratio >= self.warning_threshold:
            self._trigger_alert("warning", f"日预算已用 {usage_ratio:.0%}: ${self.today_cost:.2f}/${self.daily_budget}")

    def can_proceed(self) -> bool:
        """检查是否允许继续调用"""
        if self.limited:
            return False
        return True

    def _trigger_alert(self, level: str, message: str):
        """发送告警"""
        print(f"[{level.upper()}] {message}")
        # 实际中发送到 Slack/PagerDuty

    def _enable_rate_limit(self):
        """启用限流：降级到小模型 + 限制 QPS"""
        print("已启用限流模式：降级到 gpt-4o-mini，QPS 限制为 10")
```

### 6.3 成本优化清单

| 优化项 | 预期节省 | 实施难度 | 优先级 |
|-------|---------|---------|-------|
| 精确缓存 | 20-40% | 低 | ⭐⭐⭐ |
| 语义缓存 | 30-50% | 中 | ⭐⭐⭐ |
| 大小模型路由 | 40-60% | 中 | ⭐⭐⭐ |
| 精简提示词 | 10-30% | 低 | ⭐⭐ |
| 上下文压缩 | 20-40% | 中 | ⭐⭐ |
| Batch API | 50% | 低 | ⭐⭐ |
| 限制输出长度 | 10-20% | 低 | ⭐⭐ |
| 自托管模型 | 60-80% | 高 | ⭐ |

推荐的优化顺序：

```
第一步（快速见效）：
  ├── 精简提示词
  ├── 限制输出长度
  └── 精确缓存

第二步（中等投入）：
  ├── 大小模型路由
  ├── 语义缓存
  └── Batch API（非实时场景）

第三步（长期投资）：
  ├── 上下文压缩
  └── 自托管推理（高流量场景）
```

---

## 练习

1. 实现语义缓存并测量缓存命中率
2. 构建一个大小模型路由系统
3. 对比 vLLM 和 Ollama 的推理性能

## 延伸阅读

- [vLLM 文档](https://docs.vllm.ai/)
- [Ollama 文档](https://ollama.com/)
- [GPTCache](https://github.com/zilliztech/GPTCache)
