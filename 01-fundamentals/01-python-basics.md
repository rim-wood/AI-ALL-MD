# Python 编程基础

> AI 应用开发的主力语言——掌握 Python 在 AI 场景下的核心技能

## 学习目标

- 掌握 Python 类型提示、装饰器、生成器等 AI 开发常用特性
- 理解 asyncio 异步编程模型
- 能够使用 FastAPI + Pydantic 构建 API 服务
- 熟悉 AI 开发常用的包管理与数据处理工具

## 为什么是 Python

Python 是 AI/ML 生态的绝对主力：LangChain、LlamaIndex、CrewAI 等主流框架都以 Python 为第一语言，几乎所有 LLM 提供商的 SDK 都优先支持 Python。掌握 Python 是 AI 应用开发的必备基础。

---

## 1. 类型提示（Type Hints）

Python 3.10+ 引入了更简洁的类型提示语法，在 AI 开发中广泛用于 Pydantic 模型定义、函数签名和 IDE 智能提示。

### 1.1 基础类型

```python
# 基本类型注解
name: str = "GPT-4o"
temperature: float = 0.7
max_tokens: int = 4096
stream: bool = True

# 函数签名
def chat(message: str, temperature: float = 0.7) -> str:
    ...
```

### 1.2 复合类型

```python
# Python 3.10+ 联合类型语法
def get_model(name: str) -> str | None:
    ...

# 容器类型
messages: list[dict[str, str]] = [
    {"role": "user", "content": "Hello"}
]

# 字典类型
config: dict[str, str | int | float] = {
    "model": "gpt-4o",
    "max_tokens": 4096,
    "temperature": 0.7,
}
```

### 1.3 TypedDict 与泛型

```python
from typing import TypedDict, Generic, TypeVar

# TypedDict —— LangGraph 状态定义常用
class AgentState(TypedDict):
    messages: list[dict[str, str]]
    current_step: str
    context: str | None

# 泛型
T = TypeVar("T")

class Response(Generic[T]):
    def __init__(self, data: T, status: int):
        self.data = data
        self.status = status
```

## 2. 装饰器

装饰器在 AI 框架中无处不在：FastAPI 路由、LangChain 工具定义、MCP Server 等都依赖装饰器。

### 2.1 函数装饰器

```python
import time
from functools import wraps

def retry(max_attempts: int = 3, delay: float = 1.0):
    """LLM API 调用重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1.0)
async def call_llm(prompt: str) -> str:
    ...
```

### 2.2 AI 框架中的装饰器

```python
# FastAPI 路由
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    ...

# LangChain 工具定义
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索网页获取最新信息"""
    ...

# MCP Server 工具
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("my-server")

@mcp.tool()
def query_database(sql: str) -> str:
    """执行数据库查询"""
    ...
```

## 3. 生成器与迭代器

生成器是实现流式输出的基础，在 LLM 流式响应中至关重要。

### 3.1 基础生成器

```python
def count_tokens(text: str, chunk_size: int = 100):
    """分块生成文本"""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

for chunk in count_tokens(long_text):
    print(chunk, end="", flush=True)
```

### 3.2 异步生成器

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_chat(prompt: str):
    """流式调用 LLM"""
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

## 4. 上下文管理器

用于资源管理，如数据库连接、临时文件、API 客户端等。

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_connection():
    conn = await create_connection()
    try:
        yield conn
    finally:
        await conn.close()

# 使用
async with get_db_connection() as conn:
    results = await conn.execute("SELECT * FROM documents")
```

## 5. 异步编程（asyncio）

AI 应用大量涉及网络 I/O（调用 LLM API、向量数据库查询等），异步编程是提升吞吐量的关键。

### 5.1 async/await 基础

```python
import asyncio

async def call_llm(prompt: str) -> str:
    """模拟 LLM API 调用"""
    await asyncio.sleep(1)  # 模拟网络延迟
    return f"Response to: {prompt}"

async def main():
    result = await call_llm("Hello")
    print(result)

asyncio.run(main())
```

### 5.2 并发调用

```python
async def batch_call(prompts: list[str]) -> list[str]:
    """并发调用多个 LLM 请求"""
    tasks = [call_llm(p) for p in prompts]
    return await asyncio.gather(*tasks)

# 带并发限制
async def batch_call_limited(prompts: list[str], max_concurrent: int = 5):
    """限制并发数的批量调用"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_call(prompt: str) -> str:
        async with semaphore:
            return await call_llm(prompt)

    tasks = [limited_call(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

### 5.3 异步 HTTP 客户端

```python
import httpx

async def fetch_embeddings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            json={"input": texts, "model": "text-embedding-3-small"},
            headers={"Authorization": "Bearer <api-key>"},
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]
```

## 6. FastAPI

AI 应用后端的首选框架——异步原生、自动生成 API 文档、与 Pydantic 深度集成。

### 6.1 基础路由

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Chat API")

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o"
    temperature: float = 0.7

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    reply = await call_llm(request.message)
    return ChatResponse(reply=reply, tokens_used=150)
```

### 6.2 流式响应（SSE）

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in stream_chat(request.message):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 6.3 中间件与错误处理

```python
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": str(exc), "status": 500}
```

## 7. Pydantic

AI 开发中的数据验证利器——LLM 结构化输出、API 请求验证、配置管理都离不开它。

### 7.1 模型定义

```python
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str = Field(description="消息角色: system/user/assistant")
    content: str = Field(description="消息内容")

class LLMConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
```

### 7.2 嵌套模型与验证

```python
from pydantic import field_validator

class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)

    @field_validator("sources")
    @classmethod
    def sources_not_empty(cls, v):
        if not v:
            raise ValueError("必须提供至少一个来源")
        return v
```

### 7.3 用于 LLM 结构化输出

```python
from openai import OpenAI

client = OpenAI()

class ExtractedInfo(BaseModel):
    name: str
    age: int
    skills: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "提取信息：张三，25岁，会Python和机器学习"}],
    response_format=ExtractedInfo,
)

info = completion.choices[0].message.parsed
# ExtractedInfo(name='张三', age=25, skills=['Python', '机器学习'])
```

## 8. 包管理与开发环境

### 8.1 uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建项目
uv init my-ai-app
cd my-ai-app

# 添加依赖
uv add fastapi openai langchain pydantic

# 运行
uv run python main.py
```

### 8.2 常用 AI 开发依赖

```toml
# pyproject.toml
[project]
dependencies = [
    "openai",          # OpenAI SDK
    "anthropic",       # Anthropic SDK
    "langchain",       # LangChain 框架
    "fastapi",         # Web 框架
    "uvicorn",         # ASGI 服务器
    "pydantic",        # 数据验证
    "httpx",           # 异步 HTTP 客户端
]
```

---

## 练习

1. 用 FastAPI 构建一个支持流式响应的聊天 API
2. 实现一个带重试和并发限制的异步批量 LLM 调用器
3. 用 Pydantic 定义一个 RAG 系统的完整请求/响应模型

## 延伸阅读

- [Python 官方文档](https://docs.python.org/3/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Pydantic 官方文档](https://docs.pydantic.dev/)
- [uv 文档](https://docs.astral.sh/uv/)
- [Real Python - Async IO](https://realpython.com/async-io-python/)
