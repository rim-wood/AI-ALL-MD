# LlamaIndex

> 数据连接与 RAG 应用的专业框架

## 学习目标

- 理解 LlamaIndex 的模块化架构与核心抽象（Document、Node、Index）
- 掌握 LlamaHub 数据连接器与 LlamaParse 文档解析的使用方法
- 能够构建向量索引、摘要索引、知识图谱索引，并灵活组合多索引策略
- 熟练使用 QueryEngine、ChatEngine、SubQuestionQueryEngine、RouterQueryEngine
- 了解 Workflow 事件驱动框架与 AgentWorkflow 多 Agent 编排
- 理解 LlamaIndex 与 LangChain 的定位差异与协作模式
- 能够独立构建企业级知识库 RAG 应用

---

## 1. 核心概念

### 1.1 架构概览

LlamaIndex 最初名为 **GPT Index**（2022 年底由 Jerry Liu 创建），起初只是一个简单的 RAG 框架，帮助开发者将自有数据连接到 LLM。经过三年多的持续演进，到 2026 年，LlamaIndex 已完成从 "RAG 框架" 到 **"数据 + 工作流的多 Agent 框架"** 的战略转型——其官方文档已全面围绕 **AgentWorkflow** 重新组织，RAG 被定位为 Agent 能力的一个子集而非全部。

这一转型的背后是企业级落地的驱动。LlamaIndex 目前为多家大型企业提供 **Long-Term Memory**（长期记忆）解决方案，包括 Boeing 旗下的航空信息服务商 Jeppesen（用于飞行手册和航空法规的智能检索）以及 KPMG（用于审计文档和合规知识库）。这些案例验证了 LlamaIndex 在处理海量、高复杂度企业文档方面的核心竞争力。

其架构采用 **核心包 + 集成包 + 社区生态** 的模块化设计：

```
┌─────────────────────────────────────────────────────────┐
│                    你的 AI 应用                          │
├─────────────────────────────────────────────────────────┤
│              Query Engines / Chat Engines                │
│              Workflow / AgentWorkflow                     │
├─────────────────────────────────────────────────────────┤
│                   Index 层                               │
│   VectorStoreIndex │ SummaryIndex │ KnowledgeGraphIndex  │
├─────────────────────────────────────────────────────────┤
│              Node & Transformation 层                    │
│        SentenceSplitter │ MetadataExtractor              │
├─────────────────────────────────────────────────────────┤
│                Document & Reader 层                      │
│          LlamaHub (160+ Data Connectors)                 │
│          LlamaParse (企业级文档解析)                       │
├─────────────────────────────────────────────────────────┤
│                 llama-index-core                         │
│          核心抽象、接口定义、基础工具                       │
├─────────────────────────────────────────────────────────┤
│              Integration Packages                        │
│  llama-index-llms-openai  │  llama-index-vector-stores-  │
│  llama-index-llms-anthropic│  chroma / pinecone / pg...  │
│  llama-index-embeddings-*  │  llama-index-readers-*      │
└─────────────────────────────────────────────────────────┘
```

**安装方式**：

```bash
# 核心包（必装）
pip install llama-index-core

# 按需安装集成包
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-vector-stores-chroma

# 或者一键安装常用组合（包含 OpenAI 集成）
pip install llama-index
```

这种模块化设计的优势在于：
- **按需安装**：只安装项目需要的组件，避免依赖膨胀
- **独立版本**：各集成包独立发版，不会因某个集成的更新影响整体稳定性
- **社区扩展**：第三方可以轻松开发和发布自己的集成包

### 1.2 Document & Node

LlamaIndex 使用两层数据抽象来表示和处理非结构化数据：

**Document** 是数据源的容器，代表一个完整的文档（如一个 PDF 文件、一篇网页、一条数据库记录）：

```python
from llama_index.core import Document

# 手动创建 Document
doc = Document(
    text="LlamaIndex 是一个用于构建 LLM 应用的数据框架...",
    metadata={
        "source": "official_docs",
        "author": "LlamaIndex Team",
        "date": "2026-01-15",
    },
    excluded_llm_metadata_keys=["source"],      # LLM 生成时不包含此元数据
    excluded_embed_metadata_keys=["date"],       # Embedding 时不包含此元数据
)

print(doc.doc_id)       # 自动生成的唯一 ID
print(doc.metadata)     # 元数据字典
print(doc.text)         # 文本内容
```

**Node**（也称 TextNode）是 Document 经过分块后的最小检索单元。Node 之间可以维护关系（前后文、父子层级），这是 LlamaIndex 区别于简单分块方案的关键特性：

```python
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

# 创建节点并建立关系
node1 = TextNode(text="第一章：LlamaIndex 简介...", id_="node-1")
node2 = TextNode(text="第二章：核心架构...", id_="node-2")

# 建立前后关系
node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id="node-2")
node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id="node-1")

# 建立父子关系（用于层级索引）
parent_node = TextNode(text="完整文档摘要...", id_="parent-1")
node1.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id="parent-1")
```

Document 与 Node 的关系：

```
Document (完整 PDF)
  ├── Node 1 (第1段) ──NEXT──► Node 2 (第2段) ──NEXT──► Node 3 (第3段)
  │     ▲                        ▲                        ▲
  │   PARENT                   PARENT                   PARENT
  │     │                        │                        │
  └── Parent Node (文档摘要)  ◄──┘                        │
                                                          │
Document (另一个 PDF)                                      │
  ├── Node 4 ──NEXT──► Node 5 ──────────────────SOURCE───┘
```

> 💡 Node 的关系机制使得检索时可以"向上追溯"获取更多上下文，或"向前后扩展"获取相邻内容，这对提升 RAG 回答质量至关重要。详见 [RAG 章节](../02-core-tech/04-rag.md) 中关于检索优化的讨论。

### 1.3 Index

Index 是 LlamaIndex 的核心组件，负责将 Node 组织成可高效查询的数据结构。不同的 Index 类型适用于不同的查询场景：

| Index 类型 | 数据结构 | 适用场景 | 查询方式 |
|---|---|---|---|
| **VectorStoreIndex** | 向量嵌入 + ANN 索引 | 语义相似度搜索 | Embedding 相似度匹配 |
| **SummaryIndex** | 线性列表 | 全文摘要、总结类问题 | 遍历所有节点生成摘要 |
| **KnowledgeGraphIndex** | 知识图谱三元组 | 实体关系查询 | 图遍历 + 关键词匹配 |
| **TreeIndex** | 树形层级摘要 | 多层次摘要 | 自顶向下遍历 |
| **KeywordTableIndex** | 关键词倒排索引 | 关键词精确匹配 | 关键词提取 + 查表 |

最常用的是 **VectorStoreIndex**，快速上手示例：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 2. 构建索引（自动完成分块 → Embedding → 存储）
index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎并查询
query_engine = index.as_query_engine()
response = query_engine.query("LlamaIndex 的核心架构是什么？")
print(response)
```

这三行核心代码背后，LlamaIndex 自动完成了：
1. 将 Document 分块为 Node（默认使用 SentenceSplitter，chunk_size=1024）
2. 调用 Embedding 模型将每个 Node 转为向量
3. 将向量存入内存向量存储
4. 查询时将问题转为向量，检索 Top-K 相似节点，拼接为 Prompt 发送给 LLM

---

## 2. 数据连接

### 2.1 LlamaHub 数据连接器

[LlamaHub](https://llamahub.ai/) 是 LlamaIndex 的数据连接器生态，提供 **160+ 开箱即用的 Reader**，覆盖主流数据源：

| 类别 | 连接器示例 | 安装包 |
|---|---|---|
| **文件** | PDF、Word、Excel、Markdown、CSV | `llama-index-readers-file` |
| **数据库** | PostgreSQL、MySQL、MongoDB | `llama-index-readers-database` |
| **Web** | 网页爬取、Sitemap、RSS | `llama-index-readers-web` |
| **SaaS** | Notion、Slack、Google Docs、Confluence | `llama-index-readers-notion` 等 |
| **代码** | GitHub Repo、代码文件 | `llama-index-readers-github` |
| **知识库** | Wikipedia、Arxiv | `llama-index-readers-wikipedia` |

```python
# 示例：从多种数据源加载文档

# 1. 本地文件目录
from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".md", ".txt"],  # 只加载指定格式
    recursive=True,                          # 递归子目录
).load_data()

# 2. 从 Notion 加载
from llama_index.readers.notion import NotionPageReader

notion_reader = NotionPageReader(integration_token="your-token")
notion_docs = notion_reader.load_data(page_ids=["page-id-1", "page-id-2"])

# 3. 从数据库加载
from llama_index.readers.database import DatabaseReader

db_reader = DatabaseReader(uri="postgresql://user:pass@localhost/mydb")
db_docs = db_reader.load_data(query="SELECT title, content FROM articles WHERE active = true")
```

### 2.2 文档解析：LlamaParse

对于企业场景中常见的复杂文档（含表格的 PDF、多栏排版、扫描件等），简单的文本提取往往丢失结构信息。**LlamaParse** 是 LlamaIndex 推出的**企业级文档解析云服务**（非开源），专门解决这一痛点。

与 PyPDF、Unstructured 等开源解析方案相比，LlamaParse 在处理复杂文档时有显著优势——特别是包含嵌套表格、图表、数学公式和多栏排版的 PDF。在 LlamaIndex 官方基准测试中，LlamaParse 在复杂文档上的解析准确率显著优于开源替代方案，尤其在表格结构保留和跨页内容拼接方面表现突出。

> ⚠️ LlamaParse 是一项**云服务**，文档会上传到 LlamaIndex Cloud 进行解析。免费额度为每天 1000 页，企业版提供更高配额和私有部署选项。对于数据敏感的场景，需要评估合规要求。

```python
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

# 配置 LlamaParse 解析器
parser = LlamaParse(
    api_key="your-llamaparse-api-key",
    result_type="markdown",           # 输出 Markdown 格式，保留结构
    language="zh",                     # 支持中文
    parsing_instruction="这是一份技术文档，请保留所有代码块和表格结构",
)

# 指定文件类型使用 LlamaParse
file_extractor = {".pdf": parser, ".html": parser}

documents = SimpleDirectoryReader(
    input_dir="./enterprise_docs",
    file_extractor=file_extractor,
).load_data()
```

LlamaParse 的核心优势：

| 特性 | 说明 |
|---|---|
| **表格保留** | 将 PDF 表格转为 Markdown 表格，保留行列结构 |
| **多栏识别** | 正确处理双栏、三栏排版 |
| **图片描述** | 对文档中的图片生成文字描述 |
| **多语言** | 支持中文、日文、韩文等多语言文档 |
| **多格式** | PDF、Word、PowerPoint、HTML、Excel 等 |

### 2.3 数据转换

文档加载后，需要经过转换管道（Transformation Pipeline）处理为适合索引的 Node：

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
)
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline

# 构建转换管道
pipeline = IngestionPipeline(
    transformations=[
        # 1. 分块：按句子边界切分
        SentenceSplitter(chunk_size=512, chunk_overlap=64),
        # 2. 元数据提取：自动生成标题
        TitleExtractor(nodes=3),
        # 3. 元数据提取：生成该块能回答的问题
        QuestionsAnsweredExtractor(questions=3),
    ]
)

nodes = pipeline.run(documents=documents)
print(f"生成 {len(nodes)} 个节点")
print(f"示例元数据: {nodes[0].metadata}")
```

**层级分块**（Hierarchical Chunking）是 LlamaIndex 的特色功能，支持构建多粒度的节点层级：

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

# 创建三级层级：2048 → 512 → 128
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

all_nodes = hierarchical_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(all_nodes)

print(f"总节点数: {len(all_nodes)}, 叶子节点数: {len(leaf_nodes)}")
```

> 💡 层级分块配合 `AutoMergingRetriever` 使用效果最佳：检索时先匹配叶子节点，如果同一父节点下的多个叶子都被命中，则自动合并为父节点返回，提供更完整的上下文。这与 [RAG 章节](../02-core-tech/04-rag.md) 中讨论的检索优化策略一脉相承。

---

## 3. 索引构建

### 3.1 向量索引

**VectorStoreIndex** 是最常用的索引类型，将文本转为向量嵌入后存储，查询时通过语义相似度检索。

**使用内存存储（开发/测试）**：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 全局配置
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载并构建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 持久化到磁盘
index.storage_context.persist(persist_dir="./storage")

# 后续从磁盘加载（无需重新 Embedding）
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

**使用外部向量数据库（生产环境）**：

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# 连接 ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_docs")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
```

> 💡 关于向量数据库的选型（Chroma、Pinecone、Weaviate、pgvector 等），详见 [Embedding 与向量搜索](../02-core-tech/05-embeddings-and-vector-search.md) 章节。

### 3.2 摘要索引

**SummaryIndex**（原 ListIndex）将所有节点存储为线性列表，查询时遍历所有节点生成摘要。适合"总结全文"类问题：

```python
from llama_index.core import SummaryIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
summary_index = SummaryIndex.from_documents(documents)

# 摘要查询 — 会遍历所有节点
query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"  # 层级摘要，适合长文档
)
response = query_engine.query("请总结这份文档的核心要点")
print(response)
```

`response_mode` 选项：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `refine` | 逐块迭代精炼答案 | 需要综合多块信息 |
| `compact` | 尽量将多块合并后再精炼 | 减少 LLM 调用次数 |
| `tree_summarize` | 递归构建摘要树 | 长文档总结 |
| `simple_summarize` | 截断后一次性总结 | 短文档快速总结 |

### 3.3 知识图谱索引

**KnowledgeGraphIndex** 从文本中提取实体和关系，构建知识图谱。适合需要理解实体关系的场景：

```python
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o", temperature=0)

documents = SimpleDirectoryReader("./data/company").load_data()

# 构建知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,       # 每个块最多提取 10 个三元组
    include_embeddings=True,          # 同时生成嵌入，支持混合查询
)

# 查询
query_engine = kg_index.as_query_engine(
    include_text=True,                # 返回原始文本（不仅是三元组）
    response_mode="tree_summarize",
)
response = query_engine.query("张三在公司中负责哪些项目？")
print(response)
```

提取的三元组示例：
```
(张三, 担任, 技术总监)
(张三, 负责, 推荐系统项目)
(推荐系统项目, 使用, LlamaIndex)
(推荐系统项目, 部署于, AWS)
```

### 3.4 多索引策略

实际项目中，不同类型的数据适合不同的索引。LlamaIndex 支持灵活的多索引组合：

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool

# 为不同数据源构建不同索引
tech_docs = SimpleDirectoryReader("./data/tech").load_data()
finance_docs = SimpleDirectoryReader("./data/finance").load_data()

tech_index = VectorStoreIndex.from_documents(tech_docs)
finance_index = VectorStoreIndex.from_documents(finance_docs)
summary_index = SummaryIndex.from_documents(tech_docs + finance_docs)

# 封装为查询工具
tech_tool = QueryEngineTool.from_defaults(
    query_engine=tech_index.as_query_engine(),
    name="tech_docs",
    description="技术文档知识库，包含 API 文档、架构设计、技术规范",
)

finance_tool = QueryEngineTool.from_defaults(
    query_engine=finance_index.as_query_engine(),
    name="finance_docs",
    description="财务文档知识库，包含财报、预算、审计报告",
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
    name="summary",
    description="全文档摘要工具，用于总结和概览类问题",
)
```

这些工具将在下一节的 RouterQueryEngine 和 SubQuestionQueryEngine 中使用。

---

## 4. 查询引擎

查询引擎是 LlamaIndex 中连接用户问题与索引数据的桥梁。不同的查询引擎适用于不同的交互模式和问题复杂度。

### 4.1 基础查询引擎

**QueryEngine** 是最基础的查询接口，执行"检索 → 生成"的单轮问答：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 基础查询引擎，可配置检索参数
query_engine = index.as_query_engine(
    similarity_top_k=5,              # 检索 Top-5 相似节点
    response_mode="compact",          # 合并后生成
    streaming=True,                   # 流式输出
)

# 流式查询
response = query_engine.query("LlamaIndex 如何处理大规模文档？")
response.print_response_stream()     # 逐 token 打印

# 查看检索到的源节点
for node in response.source_nodes:
    print(f"[Score: {node.score:.3f}] {node.text[:100]}...")
```

**自定义检索器**可以精细控制检索行为：

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# 自定义检索器
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

# 后处理：过滤低相似度结果
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# 响应合成器
synthesizer = get_response_synthesizer(response_mode="compact")

# 组装查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    node_postprocessors=[postprocessor],
)

response = query_engine.query("什么是向量索引？")
```

### 4.2 Chat Engine

**ChatEngine** 在 QueryEngine 基础上增加了对话历史管理，支持多轮对话：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建 Chat Engine
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",  # 推荐模式
    verbose=True,
)

# 多轮对话
response1 = chat_engine.chat("LlamaIndex 支持哪些索引类型？")
print(response1)

response2 = chat_engine.chat("其中哪个最适合语义搜索？")  # 自动关联上文
print(response2)

response3 = chat_engine.chat("能给我一个代码示例吗？")    # 继续追问
print(response3)

# 重置对话历史
chat_engine.reset()
```

Chat Engine 模式对比：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `best` | 自动选择最佳策略（默认使用 Agent） | 通用场景 |
| `condense_question` | 将多轮对话压缩为单个独立问题再检索 | 简单追问 |
| `condense_plus_context` | 压缩问题 + 注入检索上下文 | 知识库问答（推荐） |
| `context` | 每轮都检索相关上下文 | 每轮问题独立 |
| `simple` | 不检索，纯 LLM 对话 | 闲聊/不需要知识库 |

### 4.3 Sub-Question Query Engine

面对复杂问题，**SubQuestionQueryEngine** 会自动将其分解为多个子问题，分别查询不同数据源后综合回答：

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# 假设已有多个索引（来自 3.4 节）
tools = [tech_tool, finance_tool, summary_tool]

# 创建子问题查询引擎
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    verbose=True,  # 打印分解过程
)

# 复杂问题 → 自动分解
response = sub_question_engine.query(
    "对比技术部门和财务部门的 Q3 预算执行情况，并总结主要差异"
)
print(response)
```

执行过程（verbose 输出）：
```
Generated 3 sub questions:
[tech_docs] 技术部门 Q3 预算执行情况如何？
[finance_docs] 财务部门 Q3 预算执行情况如何？
[summary] 两个部门预算执行的主要差异是什么？
```

### 4.4 路由查询引擎

**RouterQueryEngine** 根据问题语义自动路由到最合适的查询引擎：

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# 创建路由查询引擎
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[tech_tool, finance_tool, summary_tool],
    verbose=True,
)

# 自动路由到 tech_docs
response1 = router_engine.query("API 的认证方式是什么？")

# 自动路由到 finance_docs
response2 = router_engine.query("Q3 营收同比增长多少？")

# 自动路由到 summary
response3 = router_engine.query("请给我一个全公司的业务概览")
```

路由选择器类型：

| 选择器 | 行为 | 特点 |
|---|---|---|
| `LLMSingleSelector` | 用 LLM 选择一个最佳引擎 | 准确但有 LLM 调用开销 |
| `LLMMultiSelector` | 用 LLM 选择多个引擎并合并结果 | 适合跨领域问题 |
| `PydanticSingleSelector` | 用结构化输出选择 | 更可靠的解析 |

> 💡 RouterQueryEngine 的路由决策本质上是一次 Function Calling，与 [Function Calling 章节](../02-core-tech/03-function-calling.md) 中讨论的工具选择机制相同。

---

## 5. 高级特性

### 5.1 Agent 集成

LlamaIndex 可以将查询引擎和索引封装为 Agent 工具，让 LLM 自主决定何时检索、使用哪个数据源：

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.openai import OpenAI

# 将索引封装为工具
query_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="knowledge_base",
    description="公司内部知识库，包含产品文档和技术规范",
)

# 自定义函数工具
def get_current_date() -> str:
    """获取当前日期"""
    from datetime import date
    return date.today().isoformat()

date_tool = FunctionTool.from_defaults(fn=get_current_date)

# 创建 Agent
agent = FunctionAgent(
    tools=[query_tool, date_tool],
    llm=OpenAI(model="gpt-4o"),
    system_prompt="你是一个企业知识助手，基于知识库回答问题。",
)

# 运行 Agent
from llama_index.core.agent.workflow import AgentWorkflow

workflow = AgentWorkflow(agents=[agent])
response = await workflow.run(user_msg="最新的 API 版本是什么？")
print(response)
```

### 5.2 Workflow：事件驱动框架

**Workflow** 是 LlamaIndex 推出的事件驱动编排框架，用于构建复杂的多步骤流程。它取代了早期的 Agent 实现，提供更灵活、可控的流程编排能力。

**为什么需要 Workflow？** 标准 RAG 的 "检索 → 生成" 单次调用模式在面对以下场景时会失效：

- **多跳推理**（Multi-hop Reasoning）：问题的答案需要先检索 A，再根据 A 的结果检索 B，最后综合回答
- **检索间工具调用**：检索到的内容需要调用计算器、数据库查询等工具进一步处理后才能回答
- **动态路由**：根据检索到的内容决定下一步走哪条分支（如发现信息不足时自动改写查询重试）

Workflow 提供了一等公民级别的 **事件驱动原语**（Event-Driven Primitive），让你可以构建超越单次 retrieve-then-generate 的 **Agentic RAG** 系统。每个处理步骤通过类型安全的 Event 进行通信，支持条件分支、循环重试和并行执行，同时保持代码的可读性和可调试性。

```
┌──────────────────────────────────────────────────┐
│                   Workflow                        │
│                                                  │
│  StartEvent ──► StepA ──► StepB ──► StopEvent    │
│                   │                   ▲          │
│                   └──► StepC ─────────┘          │
│                                                  │
│  • 每个 Step 是一个异步函数                        │
│  • Step 之间通过 Event 通信                       │
│  • 支持条件分支、循环、并行                        │
└──────────────────────────────────────────────────┘
```

核心概念：
- **Event**：步骤间传递的消息，是类型安全的 Pydantic 模型
- **Step**：用 `@step` 装饰的异步函数，接收 Event 并返回新的 Event
- **Workflow**：包含多个 Step 的容器，管理事件流转

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step

# 定义自定义事件
class QueryEvent(Event):
    query: str

class RetrievalEvent(Event):
    query: str
    context: str

# 定义 Workflow
class RAGWorkflow(Workflow):
    @step
    async def rewrite_query(self, ev: StartEvent) -> QueryEvent:
        """改写用户查询"""
        llm = Settings.llm
        response = await llm.acomplete(
            f"将以下查询改写为更适合检索的形式：{ev.query}"
        )
        return QueryEvent(query=str(response))

    @step
    async def retrieve(self, ev: QueryEvent) -> RetrievalEvent:
        """检索相关文档"""
        retriever = self.index.as_retriever(similarity_top_k=5)
        nodes = await retriever.aretrieve(ev.query)
        context = "\n\n".join([n.text for n in nodes])
        return RetrievalEvent(query=ev.query, context=context)

    @step
    async def generate(self, ev: RetrievalEvent) -> StopEvent:
        """生成回答"""
        llm = Settings.llm
        prompt = f"基于以下上下文回答问题。\n\n上下文：{ev.context}\n\n问题：{ev.query}"
        response = await llm.acomplete(prompt)
        return StopEvent(result=str(response))

# 运行 Workflow
workflow = RAGWorkflow()
workflow.index = index  # 注入索引
result = await workflow.run(query="LlamaIndex 的 Workflow 是什么？")
print(result)
```

Workflow 相比传统 Chain 的优势：

| 特性 | 传统 Chain | Workflow |
|---|---|---|
| 流程控制 | 线性管道 | 条件分支、循环、并行 |
| 类型安全 | 弱类型 dict 传递 | 强类型 Event 模型 |
| 错误处理 | 全链中断 | 步骤级重试和降级 |
| 可观测性 | 需额外集成 | 内置事件追踪 |
| 异步支持 | 部分支持 | 原生 async |

### 5.3 AgentWorkflow：多 Agent 编排

**AgentWorkflow** 是 Workflow 之上的高级抽象，专门用于多 Agent 协作场景。多个 Agent 各司其职，通过 handoff（移交）机制协作完成复杂任务：

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, AgentOutput
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")

# 定义工具函数
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库"""
    engine = index.as_query_engine()
    return str(engine.query(query))

def generate_report(topic: str, content: str) -> str:
    """生成格式化报告"""
    return f"# {topic}\n\n{content}\n\n---\n生成时间: 2026-04-16"

# 创建专职 Agent
researcher = FunctionAgent(
    name="researcher",
    description="负责搜索和收集信息的研究员",
    tools=[FunctionTool.from_defaults(fn=search_knowledge_base)],
    llm=llm,
    system_prompt="你是一个研究员，负责从知识库中搜索和整理信息。完成后将结果移交给 writer。",
    can_handoff_to=["writer"],
)

writer = FunctionAgent(
    name="writer",
    description="负责撰写报告的写作者",
    tools=[FunctionTool.from_defaults(fn=generate_report)],
    llm=llm,
    system_prompt="你是一个技术写作者，基于研究员提供的信息撰写结构化报告。",
    can_handoff_to=["researcher"],  # 如需更多信息可以回传
)

# 创建多 Agent 工作流
workflow = AgentWorkflow(
    agents=[researcher, writer],
    root_agent="researcher",  # 入口 Agent
)

# 运行
response = await workflow.run(
    user_msg="请调研 LlamaIndex 的 Workflow 机制并撰写一份技术报告"
)
print(response)
```

多 Agent 协作流程：

```
用户请求 ──► researcher (搜索知识库)
                  │
                  ▼ handoff
              writer (撰写报告)
                  │
                  ▼ (如需补充信息)
              researcher (再次搜索)
                  │
                  ▼ handoff
              writer (完善报告) ──► 最终输出
```

> 💡 关于多 Agent 系统的设计模式和最佳实践，详见 [AI Agent 开发](./04-ai-agent-development.md) 和 [Agentic 系统设计](../06-advanced/01-agentic-system-design.md) 章节。

### 5.4 Observability

LlamaIndex 内置了可观测性支持，通过回调机制追踪每一步的执行细节：

```python
# 方式一：使用内置调试处理器
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# 创建调试处理器
debug_handler = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([debug_handler])

# 注入到 Settings
from llama_index.core import Settings
Settings.callback_manager = callback_manager

# 之后的所有操作都会被追踪
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("什么是 RAG？")

# 查看追踪信息
print(debug_handler.get_llm_inputs_outputs())  # LLM 输入输出
print(debug_handler.get_event_pairs())          # 事件对
```

```python
# 方式二：集成 Arize Phoenix（推荐生产环境使用）
# pip install llama-index-callbacks-arize-phoenix

import phoenix as px
from llama_index.core import set_global_handler

# 启动 Phoenix 追踪服务
px.launch_app()

# 一行代码开启全局追踪
set_global_handler("arize_phoenix")

# 之后所有 LlamaIndex 操作自动上报到 Phoenix UI
response = index.as_query_engine().query("什么是 RAG？")
# 打开 http://localhost:6006 查看追踪详情
```

可观测性追踪的关键指标：

| 指标 | 说明 | 优化方向 |
|---|---|---|
| **Retrieval Latency** | 检索耗时 | 优化索引结构、减少 Top-K |
| **LLM Latency** | LLM 调用耗时 | 使用更快的模型、减少 Prompt 长度 |
| **Token Usage** | Token 消耗量 | 优化分块大小、压缩上下文 |
| **Retrieval Relevance** | 检索结果相关性 | 改进 Embedding 模型、优化分块 |
| **Faithfulness** | 回答是否忠于检索内容 | 调整 Prompt、使用更强模型 |

> 💡 关于 LLM 应用的完整监控体系，详见 [监控与可观测性](../05-production/02-monitoring.md) 章节。

---

## 6. 与 LangChain 的对比与协作

### 6.1 定位差异

LlamaIndex 和 LangChain 是 LLM 应用开发中最主流的两个框架，但定位有明显差异：

| 维度 | LlamaIndex | LangChain |
|---|---|---|
| **核心定位** | 数据框架：连接数据 → 索引 → 查询 | 编排框架：模型 → 链 → Agent |
| **核心优势** | 数据连接器丰富、索引类型多样、RAG 开箱即用 | 模型集成广泛、链式编排灵活、Agent 生态成熟 |
| **数据处理** | 内置 160+ Reader、LlamaParse、层级分块 | 依赖社区 Loader，分块能力基础 |
| **索引能力** | 多种索引类型、自动路由、子问题分解 | 主要依赖外部向量数据库 |
| **Agent** | AgentWorkflow（较新，持续演进中） | LangGraph（成熟，功能完善） |
| **工作流** | Workflow（事件驱动） | LangGraph（状态图） |
| **可观测性** | 内置回调 + Phoenix 集成 | LangSmith（官方 SaaS） |
| **学习曲线** | RAG 场景上手快 | 概念多，灵活但复杂 |

**选型建议**：

一句话总结核心差异：**LlamaIndex 是数据密集型应用的最佳选择**（知识库、文档问答、企业搜索），**LangChain/LangGraph 是编排密集型应用的最佳选择**（多步骤 Agent、复杂工作流、多模型协调）。在实际生产系统中，两者经常组合使用——用 LlamaIndex 构建高质量的数据索引和检索层，用 LangChain/LangGraph 编排上层的 Agent 逻辑和业务流程。

```
你的项目需要什么？
│
├── 主要是 RAG / 知识库问答
│   └── ✅ 优先选 LlamaIndex（数据处理和索引是强项）
│
├── 主要是 Agent / 复杂工作流
│   └── ✅ 优先选 LangChain + LangGraph（编排能力更成熟）
│
├── 两者都需要
│   └── ✅ 组合使用（LlamaIndex 做数据层，LangChain 做编排层）
│
└── 简单原型 / 快速验证
    └── ✅ 哪个熟用哪个，两者都能胜任
```

### 6.2 协作使用

LlamaIndex 和 LangChain 并非互斥，可以优势互补。最常见的模式是将 LlamaIndex 的查询引擎作为 LangChain 的 Retriever 或 Tool 使用：

**模式一：LlamaIndex 作为 LangChain Retriever**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 用 LlamaIndex 构建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 转换为 LangChain Retriever
langchain_retriever = index.as_retriever(similarity_top_k=5)

# 在 LangChain 中使用
llm = ChatOpenAI(model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=langchain_retriever,
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "什么是向量索引？"})
print(result["result"])
```

**模式二：LlamaIndex QueryEngine 作为 LangGraph Tool**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_core.tools import tool

# 用 LlamaIndex 构建查询引擎
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

# 封装为 LangChain Tool
@tool
def search_docs(query: str) -> str:
    """搜索内部文档知识库，回答关于产品和技术的问题"""
    response = query_engine.query(query)
    return str(response)

# 在 LangGraph Agent 中使用
# （详见 LangChain / LangGraph 章节的 Agent 实现）
```

> 💡 关于 LangChain 和 LangGraph 的详细用法，参见 [LangChain / LangGraph](./01-langchain-langgraph.md) 章节。

---

## 7. 实战：企业知识库

### 7.1 需求与架构

构建一个支持多数据源、增量更新的企业知识库问答系统：

**需求**：
- 支持 PDF、Markdown、Notion 等多种数据源
- 支持增量更新，新文档自动入库
- 多轮对话，支持追问和上下文关联
- 路由查询，自动选择最相关的知识域
- 可观测性，追踪每次查询的检索和生成过程

**架构**：

```
┌─────────────────────────────────────────────────────────┐
│                      用户界面                            │
│                  (Web / API / CLI)                       │
├─────────────────────────────────────────────────────────┤
│                   Chat Engine                            │
│            (condense_plus_context 模式)                  │
├─────────────────────────────────────────────────────────┤
│               RouterQueryEngine                          │
│         ┌──────────┼──────────┐                         │
│         ▼          ▼          ▼                         │
│    技术文档索引  产品文档索引  FAQ 索引                    │
│   (VectorStore) (VectorStore) (VectorStore)             │
├─────────────────────────────────────────────────────────┤
│              Ingestion Pipeline                          │
│   LlamaParse → SentenceSplitter → MetadataExtractor     │
├─────────────────────────────────────────────────────────┤
│                   数据源                                 │
│        PDF 文件 │ Markdown │ Notion │ 数据库             │
└─────────────────────────────────────────────────────────┘
```

### 7.2 代码实现

**Step 1：配置与初始化**

```python
import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_parse import LlamaParse

# 全局配置
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ChromaDB 持久化存储
chroma_client = chromadb.PersistentClient(path="./enterprise_kb")
```

**Step 2：数据摄入管道**

```python
def create_ingestion_pipeline():
    """创建文档处理管道"""
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=64),
            TitleExtractor(nodes=3),
            Settings.embed_model,
        ]
    )

def ingest_documents(collection_name: str, input_dir: str, use_llamaparse: bool = False):
    """摄入文档到指定集合"""
    # 配置解析器
    file_extractor = {}
    if use_llamaparse:
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_PARSE_API_KEY"),
            result_type="markdown",
            language="zh",
        )
        file_extractor = {".pdf": parser, ".html": parser}

    # 加载文档
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        file_extractor=file_extractor,
        recursive=True,
    ).load_data()

    # 处理节点
    pipeline = create_ingestion_pipeline()
    nodes = pipeline.run(documents=documents)

    # 存入向量数据库
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )
    print(f"[{collection_name}] 摄入 {len(documents)} 个文档, {len(nodes)} 个节点")
    return index

# 摄入各数据源
tech_index = ingest_documents("tech_docs", "./data/tech", use_llamaparse=True)
product_index = ingest_documents("product_docs", "./data/product")
faq_index = ingest_documents("faq", "./data/faq")
```

**Step 3：构建路由查询引擎**

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# 封装查询工具
tools = [
    QueryEngineTool.from_defaults(
        query_engine=tech_index.as_query_engine(similarity_top_k=5),
        name="tech_docs",
        description="技术文档：API 参考、架构设计、部署指南、技术规范",
    ),
    QueryEngineTool.from_defaults(
        query_engine=product_index.as_query_engine(similarity_top_k=5),
        name="product_docs",
        description="产品文档：功能说明、用户手册、产品路线图、更新日志",
    ),
    QueryEngineTool.from_defaults(
        query_engine=faq_index.as_query_engine(similarity_top_k=3),
        name="faq",
        description="常见问题：账号问题、计费说明、故障排查、使用技巧",
    ),
]

# 路由查询引擎
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
)
```

**Step 4：多轮对话接口**

```python
from llama_index.core.memory import ChatMemoryBuffer

# 创建带记忆的 Chat Engine
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

chat_engine = router_engine.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "你是企业知识库助手。基于检索到的文档内容回答用户问题。"
        "如果文档中没有相关信息，请明确告知用户，不要编造答案。"
        "回答时引用信息来源。"
    ),
)

# 交互式对话
async def chat_loop():
    print("企业知识库助手已启动，输入 'quit' 退出\n")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            break
        response = await chat_engine.achat(user_input)
        print(f"\n助手: {response}\n")
        # 打印来源
        if hasattr(response, "source_nodes") and response.source_nodes:
            print("📚 参考来源:")
            for node in response.source_nodes[:3]:
                source = node.metadata.get("file_name", "未知")
                print(f"  - {source} (相关度: {node.score:.2f})")
            print()

# 运行
import asyncio
asyncio.run(chat_loop())
```

**Step 5：增量更新**

```python
def incremental_update(collection_name: str, new_files: list[str]):
    """增量更新：只处理新文件"""
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 加载新文件
    documents = SimpleDirectoryReader(input_files=new_files).load_data()

    # 处理并追加到现有索引
    pipeline = create_ingestion_pipeline()
    nodes = pipeline.run(documents=documents)

    # 追加到索引
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )
    print(f"增量更新: 新增 {len(nodes)} 个节点到 [{collection_name}]")
    return index

# 使用示例
incremental_update("tech_docs", ["./data/tech/new_api_v3.pdf"])
```

---

## 练习

### 练习 1：多文档 RAG 系统

构建一个支持 PDF 和 Markdown 的 RAG 系统：
- 使用 `SimpleDirectoryReader` 加载 `./data` 目录下的所有文档
- 使用 `SentenceSplitter` 分块（chunk_size=512）
- 构建 `VectorStoreIndex` 并持久化到 ChromaDB
- 实现查询引擎，返回答案和来源引用

**进阶**：添加 `SimilarityPostprocessor` 过滤低相关度结果（cutoff=0.7）。

### 练习 2：路由多索引查询

基于练习 1，扩展为多索引路由系统：
- 创建至少 3 个不同主题的索引（如技术、产品、FAQ）
- 使用 `RouterQueryEngine` 实现自动路由
- 使用 `SubQuestionQueryEngine` 处理跨领域复杂问题
- 对比两种引擎在不同问题类型上的表现

### 练习 3：Workflow 实践

使用 LlamaIndex Workflow 构建一个带查询改写的 RAG 流程：
- Step 1：查询改写（将口语化问题转为检索友好的形式）
- Step 2：检索（从向量索引中检索 Top-5）
- Step 3：相关性判断（判断检索结果是否相关，不相关则改写后重试）
- Step 4：生成回答

**进阶**：添加最大重试次数限制，避免无限循环。

### 练习 4：LlamaIndex + LangChain 协作

实现 LlamaIndex 与 LangChain 的协作：
- 用 LlamaIndex 构建索引和查询引擎
- 将其转换为 LangChain Retriever
- 在 LangChain 的 RetrievalQA 链中使用
- 对比纯 LlamaIndex 方案和协作方案的效果差异

---

## 延伸阅读

### 官方资源

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/) — 最权威的 API 参考和教程
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index) — 源码和示例（⭐ 38k+）
- [LlamaHub](https://llamahub.ai/) — 数据连接器市场，160+ Reader
- [LlamaParse 文档](https://docs.cloud.llamaindex.ai/) — 企业级文档解析服务
- [LlamaIndex Workflow 指南](https://docs.llamaindex.ai/en/stable/understanding/workflows/) — Workflow 框架详解

### 推荐教程

- [Building Production RAG with LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) — 官方生产级 RAG 最佳实践
- [LlamaIndex Bootcamp](https://github.com/run-llama/llama_index/tree/main/docs/docs/examples) — 官方示例集合，覆盖所有核心功能
- [Jerry Liu - Building Agentic RAG with LlamaIndex](https://www.youtube.com/watch?v=aVNgq7CMQY0) — LlamaIndex 创始人讲解 Agentic RAG（DeepLearning.AI）

### 相关章节

- [RAG（检索增强生成）](../02-core-tech/04-rag.md) — RAG 的理论基础和优化策略
- [Embedding 与向量搜索](../02-core-tech/05-embeddings-and-vector-search.md) — 向量数据库选型和 ANN 算法
- [Function Calling](../02-core-tech/03-function-calling.md) — 工具调用机制（路由查询的底层原理）
- [LangChain / LangGraph](./01-langchain-langgraph.md) — 对比框架，协作使用
- [AI Agent 开发](./04-ai-agent-development.md) — Agent 架构设计
- [监控与可观测性](../05-production/02-monitoring.md) — 生产环境监控体系