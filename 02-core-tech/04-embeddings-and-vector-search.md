# Embedding 与向量搜索

> 文本语义化表示与高效相似度检索

## 学习目标

- 理解 Embedding 的原理与应用
- 掌握向量数据库选型与使用
- 了解 ANN 算法与性能优化

---

## 1. Embedding 基础

### 1.1 什么是 Embedding

<!-- 文本→高维向量、语义空间 -->

### 1.2 嵌入模型原理

<!-- 双编码器、对比学习 -->

## 2. 主流嵌入模型

### 2.1 闭源模型

<!-- OpenAI text-embedding-3、Cohere embed-v4、Google -->

### 2.2 开源模型

<!-- BGE、Jina、E5、GTE -->

### 2.3 模型对比与基准

<!-- MTEB 排行榜、维度、性能 -->

## 3. 相似度度量

### 3.1 余弦相似度

<!-- 计算方法、适用场景 -->

### 3.2 点积

<!-- 与余弦的关系、归一化 -->

### 3.3 欧氏距离

<!-- L2 距离、适用场景 -->

## 4. ANN 算法

### 4.1 HNSW

<!-- 分层可导航小世界图、参数调优 -->

### 4.2 IVF

<!-- 倒排文件索引、聚类 -->

### 4.3 PQ（乘积量化）

<!-- 压缩存储、精度权衡 -->

## 5. 向量数据库选型

### 5.1 Pinecone

<!-- 全托管、Serverless、适用场景 -->

### 5.2 Weaviate

<!-- 混合搜索、模块化、GraphQL -->

### 5.3 Qdrant

<!-- Rust 实现、过滤性能、自托管 -->

### 5.4 Milvus

<!-- 分布式、大规模、GPU 加速 -->

### 5.5 pgvector

<!-- PostgreSQL 扩展、简单集成 -->

### 5.6 选型对比

<!-- 性能、成本、运维复杂度对比表 -->

## 6. 实践优化

### 6.1 索引策略

<!-- 分区、命名空间、元数据过滤 -->

### 6.2 批量写入

<!-- 批量 upsert、并发控制 -->

### 6.3 查询优化

<!-- 预过滤 vs 后过滤、Top-K 选择 -->

---

## 练习

1. 用 OpenAI Embedding 和 pgvector 构建一个语义搜索服务
2. 对比 HNSW 和 IVF 在不同数据规模下的性能
3. 实现一个带元数据过滤的向量检索

## 延伸阅读

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [pgvector 文档](https://github.com/pgvector/pgvector)
