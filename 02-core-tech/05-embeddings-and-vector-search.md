# Embedding 与向量搜索

> 文本语义化表示与高效相似度检索

## 学习目标

- 理解 Embedding 的原理与应用
- 掌握向量数据库选型与使用
- 了解 ANN 算法与性能优化

---

## 1. Embedding 基础

### 1.1 什么是 Embedding

Embedding（嵌入）是将离散的非结构化数据（文本、图像、音频等）映射为连续的高维向量的过程。生成的向量在语义空间中具有如下性质：

- **语义相近的内容，向量距离近**：「如何退款」和「怎么申请退货」的向量高度相似
- **语义无关的内容，向量距离远**：「如何退款」和「今天天气不错」的向量相距甚远
- **支持向量运算**：经典示例 `king - man + woman ≈ queen`

与传统的关键词匹配（TF-IDF、BM25）不同，Embedding 捕获的是**语义层面**的相似性，而非字面重合。这使得它成为 RAG、语义搜索、推荐系统等 AI 应用的基石。

```
文本输入                    Embedding 模型                 向量输出
"机器学习入门"  ──────►  [Transformer Encoder]  ──────►  [0.12, -0.34, 0.56, ..., 0.78]
                                                          (1536 维浮点数组)
```

**快速体验：生成 Embedding**

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="机器学习是人工智能的一个分支"
)

vector = response.data[0].embedding
print(f"维度: {len(vector)}")        # 1536
print(f"前5个分量: {vector[:5]}")     # [0.012, -0.034, ...]
```

### 1.2 嵌入模型原理

现代文本嵌入模型的核心架构是 **Bi-Encoder（双编码器）**，训练方式以**对比学习（Contrastive Learning）** 为主。

#### Bi-Encoder 架构

Bi-Encoder 将 Query 和 Document 分别独立编码为向量，再通过相似度函数计算匹配分数：

```
Query: "什么是RAG"          Document: "RAG是检索增强生成技术..."
       │                              │
  [Transformer Encoder]          [Transformer Encoder]
       │                              │
   q = [0.1, 0.3, ...]          d = [0.2, 0.4, ...]
       │                              │
       └──────── similarity(q, d) ────┘
                     ↓
                   0.92 (高相似度)
```

与 Cross-Encoder（将 Query 和 Document 拼接后联合编码）相比，Bi-Encoder 的优势在于 Document 向量可以**离线预计算并索引**，查询时只需编码 Query 并做向量检索，延迟从秒级降到毫秒级。

#### 对比学习训练

模型通过对比学习优化：拉近正样本对（语义相关）的向量距离，推远负样本对（语义无关）的向量距离。常用的损失函数是 **InfoNCE Loss**：

$$L = -\log \frac{e^{sim(q, d^+)/\tau}}{e^{sim(q, d^+)/\tau} + \sum_{i=1}^{N} e^{sim(q, d_i^-)/\tau}}$$

其中 $\tau$ 是温度参数，$d^+$ 是正样本，$d_i^-$ 是负样本。

**关键训练技巧**：

| 技巧 | 说明 |
|------|------|
| Hard Negative Mining | 挖掘与 Query 表面相似但语义不同的负样本，提升模型区分能力 |
| In-batch Negatives | 同一 batch 内其他样本的 Document 作为负样本，提高训练效率 |
| Knowledge Distillation | 用 Cross-Encoder（精度高但慢）的分数指导 Bi-Encoder 训练 |
| Matryoshka Representation | 训练时同时优化多个维度切片，支持灵活降维而不重新训练 |

## 2. 主流嵌入模型

### 2.1 闭源模型

**OpenAI text-embedding-3 系列**

OpenAI 提供两个版本：`text-embedding-3-small`（1536 维，性价比高）和 `text-embedding-3-large`（3072 维，精度更高）。两者均支持 Matryoshka 降维——通过 `dimensions` 参数截断向量而保持较好性能。

```python
from openai import OpenAI

client = OpenAI()

# 使用 large 模型并降维到 1024
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=["检索增强生成技术", "RAG 的工作原理"],
    dimensions=1024  # 从 3072 降维到 1024
)

vectors = [item.embedding for item in response.data]
print(f"向量数: {len(vectors)}, 维度: {len(vectors[0])}")
```

**Cohere embed-v4**

Cohere 的最新嵌入模型，原生支持多模态（文本 + 图像），并区分 `search_document` 和 `search_query` 两种 input type，针对检索场景优化。支持 Matryoshka 降维和二进制量化。

**Google text-embedding-005**

Google Vertex AI 提供的嵌入模型，768 维，支持 `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` 等 task type，与 Google Cloud 生态深度集成。

### 2.2 开源模型

**BGE 系列（BAAI）**

北京智源研究院推出的开源嵌入模型，中英文表现优异。`bge-m3` 支持多语言、多粒度、多功能（Dense + Sparse + ColBERT），是开源模型中的全能选手。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

sentences = ["什么是向量数据库", "向量数据库的作用和原理"]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"维度: {embeddings.shape}")  # (2, 1024)
```

**Jina Embeddings v3**

Jina AI 推出的多语言嵌入模型，支持 8192 token 长上下文，通过 LoRA adapter 针对不同任务（检索、分类、聚类）切换，Matryoshka 降维支持。

**E5 系列（Microsoft）**

微软推出的 E5（EmbEddings from bidirEctional Encoder rEpresentations）系列，`multilingual-e5-large-instruct` 支持通过自然语言指令控制嵌入行为，多语言表现出色。

### 2.3 模型对比与基准

以下基于 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 的综合评测数据（2026 年初）：

| 模型 | 类型 | 维度 | 最大 Token | MTEB 均分 | 中文支持 | 价格（每 M tokens） |
|------|------|------|-----------|-----------|---------|-------------------|
| text-embedding-3-large | 闭源 | 3072 | 8191 | ~65.0 | ✅ 良好 | $0.13 |
| text-embedding-3-small | 闭源 | 1536 | 8191 | ~62.3 | ✅ 良好 | $0.02 |
| Cohere embed-v4 | 闭源 | 1024 | 128000 | ~66.2 | ✅ 良好 | $0.10 |
| bge-m3 | 开源 | 1024 | 8192 | ~64.5 | ✅ 优秀 | 免费 |
| bge-large-zh-v1.5 | 开源 | 1024 | 512 | ~63.5 | ✅ 优秀 | 免费 |
| jina-embeddings-v3 | 开源 | 1024 | 8192 | ~65.5 | ✅ 良好 | 免费 |
| multilingual-e5-large-instruct | 开源 | 1024 | 514 | ~64.1 | ✅ 良好 | 免费 |

**选型建议**：

- **快速原型 / 小规模应用**：`text-embedding-3-small`，成本极低，接入简单
- **生产级中文检索**：`bge-m3` 或 `bge-large-zh-v1.5`，中文效果最佳且免费
- **多语言 + 长文本**：`jina-embeddings-v3` 或 `Cohere embed-v4`
- **追求最高精度**：`text-embedding-3-large` 或 `Cohere embed-v4`

## 3. 相似度度量

向量检索的核心是度量两个向量之间的"距离"或"相似度"。三种主流度量方式各有适用场景。

### 3.1 余弦相似度（Cosine Similarity）

衡量两个向量方向的一致性，忽略向量长度（模），值域为 $[-1, 1]$：

$$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])
c = np.array([-1.0, -2.0, -3.0])

print(cosine_similarity(a, b))  # 1.0  (方向完全一致)
print(cosine_similarity(a, c))  # -1.0 (方向完全相反)
```

**适用场景**：绝大多数文本嵌入检索场景的默认选择。对向量长度不敏感，适合不同长度文本的比较。

### 3.2 点积（Dot Product）

直接计算两个向量的内积：

$$\text{dot}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} a_i \cdot b_i$$

当向量已经 **L2 归一化**（模为 1）时，点积等价于余弦相似度。此时点积计算更快（省去除法），因此许多向量数据库推荐对归一化向量使用点积。

```python
# 归一化后点积 = 余弦相似度
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
print(np.dot(a_norm, b_norm))  # 1.0
```

**适用场景**：向量已归一化时的首选（如 BGE 模型默认输出归一化向量）。计算效率最高。

### 3.3 欧氏距离（Euclidean / L2 Distance）

衡量两个向量在空间中的直线距离，值域为 $[0, +\infty)$，**值越小越相似**：

$$L_2(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

当向量已归一化时，欧氏距离与余弦相似度存在单调关系：$L_2^2 = 2(1 - \text{cosine})$。

**适用场景**：需要考虑向量"绝对位置"的场景（如聚类）。在文本检索中较少使用。

**三种度量对比**：

| 度量 | 值域 | 相似 = | 归一化后关系 | 计算速度 | 推荐场景 |
|------|------|--------|-------------|---------|---------|
| 余弦相似度 | [-1, 1] | 值越大 | — | 中 | 通用文本检索 |
| 点积 | (-∞, +∞) | 值越大 | = 余弦相似度 | 快 | 归一化向量检索 |
| 欧氏距离 | [0, +∞) | 值越小 | 与余弦单调 | 中 | 聚类、异常检测 |

## 4. ANN 算法

精确的最近邻搜索（Exact KNN）需要遍历所有向量，时间复杂度 $O(n)$，在百万级以上数据中不可接受。**近似最近邻（Approximate Nearest Neighbor, ANN）** 算法通过构建索引结构，以微小的精度损失换取数量级的速度提升。

### 4.1 HNSW（Hierarchical Navigable Small World）

HNSW 是目前最主流的 ANN 算法，构建多层跳表式的图结构。上层稀疏、用于快速定位大致区域；下层稠密、用于精确搜索。

```
Layer 2 (稀疏):    A ─────────────── D
                   │                 │
Layer 1 (中等):    A ──── B ──── C ── D
                   │      │      │    │
Layer 0 (稠密):    A ─ B ─ C ─ E ─ F ─ D ─ G ─ H
```

**核心参数**：

| 参数 | 含义 | 典型值 | 调优建议 |
|------|------|--------|---------|
| `M` | 每个节点的最大连接数 | 16 | 增大提升召回率，但增加内存和构建时间 |
| `ef_construction` | 构建索引时的搜索宽度 | 200 | 越大索引质量越高，构建越慢 |
| `ef_search` | 查询时的搜索宽度 | 50-200 | 越大召回率越高，延迟越大 |

**特点**：查询速度极快（亚毫秒级），召回率高（>95%），但内存占用较大（需存储图结构）。适合对延迟敏感、数据规模在千万级以内的场景。

### 4.2 IVF（Inverted File Index）

IVF 先用 K-Means 将向量空间划分为 `nlist` 个聚类（Voronoi cell），查询时只搜索最近的 `nprobe` 个聚类内的向量。

```
构建阶段：K-Means 聚类
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Cluster 0│  │ Cluster 1│  │ Cluster 2│  ...  nlist 个聚类
│ v1,v5,v9 │  │ v2,v3,v7 │  │ v4,v6,v8 │
└──────────┘  └──────────┘  └──────────┘

查询阶段：只搜索最近的 nprobe 个聚类
Query → 找到最近聚类 [1, 2] → 在 Cluster 1, 2 内暴力搜索
```

**核心参数**：

| 参数 | 含义 | 典型值 | 调优建议 |
|------|------|--------|---------|
| `nlist` | 聚类数量 | $\sqrt{n}$ 到 $4\sqrt{n}$ | 数据量越大，nlist 越大 |
| `nprobe` | 查询时搜索的聚类数 | 10-50 | 增大提升召回率，降低速度 |

**特点**：内存占用低于 HNSW，构建速度快，适合数据频繁更新的场景。但查询延迟略高于 HNSW。

### 4.3 PQ（Product Quantization，乘积量化）

PQ 将高维向量切分为多个子空间，每个子空间独立做聚类量化，用聚类中心的 ID（通常 1 字节）代替原始子向量。这将向量存储从浮点数组压缩为字节数组。

```
原始向量 (1024维, 4096字节):
[0.12, -0.34, ..., 0.56, 0.78]

PQ 压缩 (128个子空间, 128字节):
切分为 128 个 8 维子向量 → 每个子向量量化为 1 字节 ID
[23, 156, 42, ..., 89, 201]  ← 32x 压缩
```

**核心参数**：

| 参数 | 含义 | 典型值 | 调优建议 |
|------|------|--------|---------|
| `m` | 子空间数量 | 维度/4 到 维度/8 | 越大精度越高，压缩率越低 |
| `nbits` | 每个子空间的量化位数 | 8 | 通常固定为 8（256 个聚类中心） |

**特点**：极致的内存压缩（可达 32-64x），适合超大规模数据集（亿级）。通常与 IVF 组合使用（IVF-PQ），先粗筛再精排。

**三种算法对比**：

| 算法 | 查询速度 | 召回率 | 内存占用 | 构建速度 | 适用规模 |
|------|---------|--------|---------|---------|---------|
| HNSW | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 百万~千万 |
| IVF | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 百万~亿 |
| PQ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 亿级+ |
| IVF-PQ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 亿级+ |

## 5. 向量数据库选型

### 5.1 Pinecone

全托管的向量数据库服务，开发者无需管理基础设施。

- **Serverless 架构**：按查询量计费，自动扩缩容，冷启动快
- **Namespace 隔离**：同一索引内通过 namespace 实现多租户隔离
- **稀疏-稠密混合检索**：原生支持 Hybrid Search
- **适用场景**：快速上线、不想运维、中小规模（百万~千万级）

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-index")

# Upsert 向量
index.upsert(vectors=[
    {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"source": "faq", "lang": "zh"}},
    {"id": "doc2", "values": [0.3, 0.4, ...], "metadata": {"source": "manual", "lang": "zh"}},
])

# 查询（带元数据过滤）
results = index.query(
    vector=[0.15, 0.25, ...],
    top_k=5,
    filter={"source": {"$eq": "faq"}},
    include_metadata=True
)
```

### 5.2 Weaviate

开源的 AI-native 向量数据库，以模块化和混合搜索著称。

- **模块化架构**：通过模块集成 OpenAI、Cohere、HuggingFace 等嵌入服务，支持自动向量化
- **混合搜索**：BM25 关键词搜索 + 向量语义搜索的融合排序
- **GraphQL API**：灵活的查询接口，支持复杂过滤和聚合
- **多租户**：原生支持租户级数据隔离
- **适用场景**：需要混合搜索、多模态、GraphQL 生态的项目

### 5.3 Qdrant

Rust 编写的高性能向量数据库，以过滤性能和易用性见长。

- **高效过滤**：自研的 payload 索引，过滤查询性能业界领先
- **丰富的数据类型**：支持 keyword、integer、float、geo、datetime 等过滤条件
- **灵活部署**：单机内存/磁盘模式、分布式集群、Qdrant Cloud 全托管
- **量化支持**：Scalar Quantization 和 Binary Quantization，降低内存占用
- **适用场景**：需要复杂过滤条件、自托管、对性能要求高的场景

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(url="http://localhost:6333")

# 创建集合
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# 插入向量
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, ...], payload={"source": "faq", "lang": "zh"}),
        PointStruct(id=2, vector=[0.3, 0.4, ...], payload={"source": "manual", "lang": "en"}),
    ]
)

# 带过滤的查询
results = client.query_points(
    collection_name="documents",
    query=[0.15, 0.25, ...],
    query_filter=Filter(must=[FieldCondition(key="lang", match=MatchValue(value="zh"))]),
    limit=5,
)
```

### 5.4 Milvus

开源的分布式向量数据库，专为大规模场景设计。

- **分布式架构**：存算分离，支持水平扩展到百亿级向量
- **多索引支持**：HNSW、IVF_FLAT、IVF_PQ、DiskANN 等多种索引
- **GPU 加速**：支持 GPU 构建索引和查询，大幅提升吞吐
- **Zilliz Cloud**：全托管的 Milvus 云服务
- **适用场景**：超大规模数据、高吞吐需求、企业级部署

### 5.5 pgvector

PostgreSQL 的向量搜索扩展，将向量能力嵌入关系型数据库。

- **零额外运维**：复用现有 PostgreSQL 基础设施
- **SQL 原生**：用标准 SQL 做向量搜索，与业务数据 JOIN 查询
- **索引支持**：ivfflat 和 hnsw 两种索引类型
- **适用场景**：数据规模不大（百万级以内）、已有 PostgreSQL、希望简化架构

```python
import psycopg2

conn = psycopg2.connect("postgresql://localhost/mydb")
cur = conn.cursor()

# 启用扩展并创建表
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        source VARCHAR(50),
        embedding vector(1536)
    )
""")

# 创建 HNSW 索引
cur.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
""")

# 语义搜索（余弦距离，<=> 操作符）
query_vector = "[0.1, 0.2, ...]"
cur.execute("""
    SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE source = 'faq'
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_vector, query_vector))

for row in cur.fetchall():
    print(f"ID: {row[0]}, 相似度: {row[2]:.4f}, 内容: {row[1][:50]}")
```

### 5.6 选型对比

| 特性 | Pinecone | Weaviate | Qdrant | Milvus | pgvector |
|------|----------|----------|--------|--------|----------|
| **部署方式** | 全托管 | 自托管/云 | 自托管/云 | 自托管/云 | PostgreSQL 扩展 |
| **开源** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **实现语言** | — | Go | Rust | Go/C++ | C |
| **最大规模** | 数十亿 | 数亿 | 数亿 | 百亿+ | 数百万 |
| **混合搜索** | ✅ | ✅ 原生 | ✅ 稀疏向量 | ✅ | ❌ 需配合 tsvector |
| **过滤性能** | 良好 | 良好 | 优秀 | 良好 | 优秀（SQL） |
| **多租户** | Namespace | 原生 | Collection/Payload | Partition | Schema/RLS |
| **GPU 加速** | — | ❌ | ❌ | ✅ | ❌ |
| **运维复杂度** | 低（托管） | 中 | 低~中 | 高 | 低 |
| **学习曲线** | 低 | 中 | 低 | 中~高 | 低（会 SQL 即可） |
| **适合阶段** | MVP/中期 | 中期/成熟 | 中期/成熟 | 大规模生产 | MVP/小规模 |

**选型决策树**：

```
数据规模？
├── < 100 万 → 已有 PostgreSQL？
│   ├── 是 → pgvector（最简方案）
│   └── 否 → Qdrant 单机 或 Pinecone Free
├── 100 万 ~ 1 亿 → 需要自托管？
│   ├── 是 → 需要复杂过滤？
│   │   ├── 是 → Qdrant
│   │   └── 否 → Weaviate（混合搜索强）
│   └── 否 → Pinecone Serverless
└── > 1 亿 → Milvus 分布式集群
```

## 6. 实践优化

### 6.1 索引策略

**分区与命名空间**

将数据按业务维度分区，缩小每次查询的搜索范围：

- **按租户分区**：每个客户一个 namespace/collection，天然隔离
- **按数据类型分区**：FAQ、文档、工单分别建索引，查询时指定范围
- **按时间分区**：历史数据归档到冷索引，热数据保持高性能

**元数据索引**

为高频过滤字段建立索引，避免全量扫描后过滤：

```python
# Qdrant: 为 payload 字段创建索引
client.create_payload_index(
    collection_name="documents",
    field_name="source",
    field_schema="keyword",  # keyword / integer / float / geo / datetime
)
```

**向量降维**

利用 Matryoshka 模型的特性，根据精度需求选择合适维度：

| 维度 | 存储（每向量） | 召回率损失 | 适用场景 |
|------|--------------|-----------|---------|
| 3072（原始） | 12 KB | 0% | 最高精度需求 |
| 1024 | 4 KB | ~1-2% | 生产推荐 |
| 512 | 2 KB | ~3-5% | 大规模 + 成本敏感 |
| 256 | 1 KB | ~5-10% | 粗筛 / 候选召回 |

### 6.2 批量写入

**批量 Upsert**

避免逐条写入，使用批量操作提升吞吐：

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def batch_embed_and_upsert(texts: list[str], batch_size: int = 100):
    """分批生成 embedding 并写入向量数据库"""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # 批量生成 embedding
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        vectors = [item.embedding for item in response.data]

        # 批量写入（以 Qdrant 为例）
        points = [
            PointStruct(id=i + j, vector=vec, payload={"text": text})
            for j, (vec, text) in enumerate(zip(vectors, batch))
        ]
        qdrant_client.upsert(collection_name="documents", points=points)

        print(f"已写入 {min(i + batch_size, len(texts))}/{len(texts)}")

# asyncio.run(batch_embed_and_upsert(all_texts))
```

**写入优化要点**：

- **批量大小**：Embedding API 通常限制每次 2048 条，向量数据库建议 100-500 条/批
- **并发控制**：使用 `asyncio.Semaphore` 限制并发数，避免触发 API 限流
- **幂等写入**：使用确定性 ID（如内容哈希），支持重试而不产生重复数据
- **进度追踪**：大规模写入时记录 checkpoint，支持断点续传

### 6.3 查询优化

**预过滤 vs 后过滤**

| 策略 | 流程 | 优点 | 缺点 |
|------|------|------|------|
| 预过滤 | 先按元数据过滤，再在子集中做 ANN | 结果精确，不会少于 top_k | 过滤后数据太少时 ANN 效果下降 |
| 后过滤 | 先做 ANN 取 top_k，再按元数据过滤 | ANN 在全量数据上效果好 | 过滤后结果可能不足 top_k |

大多数向量数据库（Qdrant、Weaviate、Milvus）默认使用**预过滤**或混合策略。Pinecone 也在查询时同时应用过滤。

**Top-K 选择**

- RAG 场景通常 `top_k = 3~10`，取决于 LLM 上下文窗口和 chunk 大小
- 可以先取较大的 `top_k`（如 20），再用 Cross-Encoder 重排序取 top 5
- 设置相似度阈值（如 cosine > 0.7），过滤掉低质量结果

**查询端 Embedding 缓存**

对高频查询缓存 Embedding 结果，减少 API 调用：

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_cached_embedding(text: str) -> tuple[float, ...]:
    """缓存高频查询的 embedding（生产环境建议用 Redis）"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return tuple(response.data[0].embedding)
```

**端到端语义搜索示例（OpenAI + pgvector）**：

```python
from openai import OpenAI
import psycopg2
import json

client = OpenAI()
conn = psycopg2.connect("postgresql://localhost/mydb")

def semantic_search(query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
    """端到端语义搜索：Query → Embedding → pgvector 检索"""
    # 1. 生成查询向量
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_vec = response.data[0].embedding

    # 2. 构建 SQL（带可选过滤）
    sql = """
        SELECT id, content, source, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
    """
    params = [json.dumps(query_vec)]

    if source_filter:
        sql += " WHERE source = %s"
        params.append(source_filter)

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([json.dumps(query_vec), top_k])

    # 3. 执行查询
    cur = conn.cursor()
    cur.execute(sql, params)
    return [
        {"id": r[0], "content": r[1], "source": r[2], "similarity": round(r[3], 4)}
        for r in cur.fetchall()
    ]

# 使用示例
results = semantic_search("如何配置 RAG 的检索参数", top_k=5, source_filter="docs")
for r in results:
    print(f"[{r['similarity']}] {r['content'][:80]}")
```

---

## 练习

1. 用 OpenAI Embedding 和 pgvector 构建一个语义搜索服务
2. 对比 HNSW 和 IVF 在不同数据规模下的性能
3. 实现一个带元数据过滤的向量检索

## 延伸阅读

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [pgvector 文档](https://github.com/pgvector/pgvector)
- [Qdrant 官方文档](https://qdrant.tech/documentation/)
- [Milvus 官方文档](https://milvus.io/docs)
- [Weaviate 官方文档](https://weaviate.io/developers/weaviate)
- [Sentence Transformers 文档](https://www.sbert.net/)
