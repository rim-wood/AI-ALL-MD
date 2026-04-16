# 智能客服与问答

> 最常见的 AI 应用场景——对话机器人与知识库问答

## 学习目标

- 构建基础对话机器人，掌握对话历史管理与系统提示词设计
- 实现基于 RAG 的知识库问答系统
- 掌握多轮对话的上下文理解与状态管理
- 实现意图识别与多技能路由

---

## 1. 基础对话机器人

### 1.1 架构设计

一个生产级对话机器人的核心流程：

```
用户输入 → 输入预处理 → 构建消息列表 → LLM 调用 → 输出后处理 → 返回响应
```

最小可用实现：

```python
from openai import OpenAI

client = OpenAI()

def chat(user_message: str, history: list[dict]) -> str:
    """基础对话函数"""
    messages = [
        {"role": "system", "content": "你是一个友好的客服助手。"},
        *history,
        {"role": "user", "content": user_message},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content
```

用 FastAPI 包装成服务：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """流式对话接口"""
    async def generate():
        messages = [
            {"role": "system", "content": "你是一个友好的客服助手。"},
            *req.history,
            {"role": "user", "content": req.message},
        ]
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")
```

### 1.2 对话历史管理

对话历史是影响回答质量的关键因素。随着对话轮次增加，消息列表会超出上下文窗口，需要管理策略。

**滑动窗口**——保留最近 N 轮对话：

```python
def sliding_window(history: list[dict], max_turns: int = 10) -> list[dict]:
    """保留最近 N 轮对话（每轮 = user + assistant）"""
    if len(history) <= max_turns * 2:
        return history
    return history[-(max_turns * 2):]
```

**Token 预算控制**——按 Token 数量截断：

```python
import tiktoken

def trim_by_tokens(
    history: list[dict], max_tokens: int = 4000, model: str = "gpt-4o-mini"
) -> list[dict]:
    """按 Token 预算从旧到新保留消息"""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    trimmed = []
    for msg in reversed(history):
        tokens = len(enc.encode(msg["content"]))
        if total + tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total += tokens
    return trimmed
```

**摘要压缩**——用 LLM 总结早期对话：

```python
async def summarize_and_trim(
    history: list[dict], max_turns: int = 6
) -> list[dict]:
    """将早期对话压缩为摘要"""
    if len(history) <= max_turns * 2:
        return history

    old_messages = history[:-(max_turns * 2)]
    recent_messages = history[-(max_turns * 2):]

    summary_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "请用 2-3 句话总结以下对话的要点。"},
            *old_messages,
        ],
        max_tokens=200,
    )
    summary = summary_resp.choices[0].message.content

    return [
        {"role": "system", "content": f"之前的对话摘要：{summary}"},
        *recent_messages,
    ]
```

> **实战建议：** 简单场景用滑动窗口即可；对话轮次多且需要长期记忆时，用摘要压缩。Token 预算控制适合需要精确成本管理的场景。

### 1.3 系统提示词设计

系统提示词决定了机器人的"人格"和行为边界。一个好的客服系统提示词应包含：

```python
SYSTEM_PROMPT = """你是 [公司名] 的智能客服助手。

## 角色设定
- 语气友好、专业，使用简洁的中文回答
- 称呼用户为"您"

## 行为规范
- 只回答与 [公司名] 产品和服务相关的问题
- 不确定的信息不要编造，告知用户"我需要确认后回复您"
- 涉及账户安全、退款等敏感操作时，引导用户联系人工客服

## 输出约束
- 回答控制在 200 字以内
- 如果需要分步骤说明，使用编号列表
- 在回答末尾询问"还有其他问题吗？"

## 知识边界
- 你只了解截至 {knowledge_cutoff} 的信息
- 对于超出知识范围的问题，诚实告知并建议查阅官方文档
"""
```

**提示词模板化**——支持动态注入上下文：

```python
from string import Template

def build_system_prompt(company: str, knowledge_cutoff: str, extra_context: str = "") -> str:
    template = Template(SYSTEM_PROMPT)
    prompt = template.safe_substitute(
        company=company,
        knowledge_cutoff=knowledge_cutoff,
    )
    if extra_context:
        prompt += f"\n\n## 补充信息\n{extra_context}"
    return prompt
```

---

## 2. 知识库问答（RAG 实战）

### 2.1 知识库构建

完整的知识库构建流程：文档导入 → 分块 → 向量化 → 存储索引。

```python
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载文档（支持多种格式）
def load_documents(file_paths: list[str]):
    docs = []
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    for path in file_paths:
        ext = path[path.rfind("."):]
        loader_cls = loaders.get(ext)
        if loader_cls:
            docs.extend(loader_cls(path).load())
    return docs

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", " "],
)

# 3. 向量化 + 存储
def build_index(file_paths: list[str], persist_dir: str = "./chroma_db"):
    docs = load_documents(file_paths)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_dir,
    )
    return vectorstore
```

### 2.2 检索与生成

**混合检索**——结合向量检索和关键词检索，提升召回率：

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(vectorstore, documents, k: int = 4):
    """向量检索 + BM25 关键词检索"""
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],  # 向量检索权重更高
    )
```

**带引用标注的 RAG 问答**：

```python
RAG_PROMPT = """基于以下参考资料回答用户问题。

## 参考资料
{context}

## 规则
- 只基于参考资料回答，不要编造信息
- 在回答中标注引用来源，格式：[来源: 文档名]
- 如果参考资料中没有相关信息，回答"抱歉，我在知识库中没有找到相关信息"

## 用户问题
{question}
"""

def rag_chat(question: str, retriever) -> str:
    # 检索相关文档
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[文档: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RAG_PROMPT.format(
                context=context, question=question
            )},
        ],
        temperature=0.3,  # RAG 场景用低 temperature
    )
    return response.choices[0].message.content
```

### 2.3 增量更新

生产环境中知识库需要持续更新：

```python
class KnowledgeBaseManager:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

    def add_documents(self, file_paths: list[str]):
        """增量添加新文档"""
        docs = load_documents(file_paths)
        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)

    def delete_by_source(self, source: str):
        """按来源删除文档"""
        self.vectorstore._collection.delete(
            where={"source": source}
        )

    def update_document(self, source: str, new_path: str):
        """更新文档 = 删除旧版 + 添加新版"""
        self.delete_by_source(source)
        self.add_documents([new_path])
```

---

## 3. 多轮对话

### 3.1 上下文理解

多轮对话的核心挑战是**指代消解**和**话题追踪**。用户经常用"它"、"这个"、"上面那个"等代词引用之前的内容。

**查询改写**——将依赖上下文的问题改写为独立问题：

```python
REWRITE_PROMPT = """根据对话历史，将用户的最新问题改写为一个独立的、完整的问题。
如果最新问题已经是独立的，直接返回原问题。

对话历史：
{history}

最新问题：{question}

改写后的独立问题："""

def rewrite_query(question: str, history: list[dict]) -> str:
    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-6:]
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(
            history=history_text, question=question
        )}],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content
```

示例效果：

| 对话历史 | 用户输入 | 改写结果 |
|---------|---------|---------|
| "我想了解你们的退货政策" → AI 回答 | "时间限制是多久？" | "退货政策的时间限制是多久？" |
| "帮我查一下订单 #12345" → AI 回答 | "它什么时候到？" | "订单 #12345 的预计到达时间是什么时候？" |

### 3.2 对话状态管理

复杂业务场景（如订单查询、退换货）需要用状态机管理对话流程：

```python
from enum import Enum
from pydantic import BaseModel

class ConversationState(str, Enum):
    IDLE = "idle"
    COLLECTING_ORDER_ID = "collecting_order_id"
    COLLECTING_REASON = "collecting_reason"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

class ConversationContext(BaseModel):
    state: ConversationState = ConversationState.IDLE
    order_id: str | None = None
    reason: str | None = None
    slots: dict = {}  # 通用槽位存储

class StatefulChatbot:
    def __init__(self):
        self.sessions: dict[str, ConversationContext] = {}

    def get_context(self, session_id: str) -> ConversationContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext()
        return self.sessions[session_id]

    async def handle(self, session_id: str, message: str) -> str:
        ctx = self.get_context(session_id)

        if ctx.state == ConversationState.IDLE:
            return await self._handle_idle(ctx, message)
        elif ctx.state == ConversationState.COLLECTING_ORDER_ID:
            return await self._handle_collect_order(ctx, message)
        elif ctx.state == ConversationState.CONFIRMING:
            return await self._handle_confirm(ctx, message)
        return "抱歉，我遇到了问题，请重新开始。"

    async def _handle_idle(self, ctx: ConversationContext, message: str) -> str:
        # 用 LLM 判断意图，决定进入哪个流程
        intent = await self._classify_intent(message)
        if intent == "return_request":
            ctx.state = ConversationState.COLLECTING_ORDER_ID
            return "好的，我来帮您处理退货。请提供您的订单号。"
        return await self._general_chat(message)

    async def _handle_collect_order(self, ctx: ConversationContext, msg: str) -> str:
        ctx.order_id = msg.strip()
        ctx.state = ConversationState.COLLECTING_REASON
        return f"已记录订单号 {ctx.order_id}，请问退货原因是什么？"
```

### 3.3 澄清与确认

当用户查询模糊时，主动提问比猜测更好：

```python
CLARIFICATION_PROMPT = """分析用户的问题，判断是否需要澄清。

如果问题清晰明确，返回 JSON：{"needs_clarification": false}
如果问题模糊或有多种理解，返回 JSON：
{
  "needs_clarification": true,
  "reason": "模糊原因",
  "suggestions": ["可能的理解1", "可能的理解2"]
}

用户问题：{question}"""

async def maybe_clarify(question: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": CLARIFICATION_PROMPT.format(
            question=question
        )}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
```

---

## 4. 意图识别与路由

### 4.1 意图分类

用 LLM 做意图分类，比传统 NLU 模型更灵活：

```python
INTENT_PROMPT = """将用户消息分类为以下意图之一，返回 JSON 格式。

可选意图：
- faq: 常见问题咨询（产品功能、价格、政策等）
- order_query: 订单查询（物流、状态、退换货）
- complaint: 投诉建议
- chitchat: 闲聊寒暄
- human_agent: 明确要求转人工

返回格式：{"intent": "意图名", "confidence": 0.0-1.0}

用户消息：{message}"""

async def classify_intent(message: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": INTENT_PROMPT.format(message=message)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
```

### 4.2 多技能路由

根据意图将请求分发到不同处理模块：

```python
class SkillRouter:
    def __init__(self, retriever, order_service):
        self.retriever = retriever
        self.order_service = order_service

    async def route(self, message: str, history: list[dict]) -> str:
        result = await classify_intent(message)
        intent = result["intent"]
        confidence = result["confidence"]

        # 低置信度时走降级策略
        if confidence < 0.6:
            return await self._fallback(message, history)

        handlers = {
            "faq": self._handle_faq,
            "order_query": self._handle_order,
            "complaint": self._handle_complaint,
            "chitchat": self._handle_chitchat,
            "human_agent": self._handle_transfer,
        }
        handler = handlers.get(intent, self._fallback)
        return await handler(message, history)

    async def _handle_faq(self, message: str, history: list[dict]) -> str:
        """FAQ 走 RAG 检索"""
        return rag_chat(message, self.retriever)

    async def _handle_order(self, message: str, history: list[dict]) -> str:
        """订单查询调用业务 API"""
        order_info = await self.order_service.query(message)
        return f"您的订单状态：{order_info}"

    async def _handle_transfer(self, message: str, history: list[dict]) -> str:
        """转人工"""
        return "正在为您转接人工客服，请稍候..."

    async def _fallback(self, message: str, history: list[dict]) -> str:
        """降级：用通用对话处理"""
        return chat(message, history)
```

### 4.3 降级策略

当系统无法处理请求时，需要优雅降级：

```python
class FallbackChain:
    """多级降级链"""

    async def handle(self, message: str, history: list[dict]) -> str:
        # 第一级：尝试 RAG 检索
        docs = self.retriever.invoke(message)
        if docs and self._is_relevant(docs[0], message):
            return rag_chat(message, self.retriever)

        # 第二级：通用 LLM 对话
        response = chat(message, history)
        if self._is_confident(response):
            return response

        # 第三级：引导用户
        return (
            "抱歉，这个问题我暂时无法准确回答。您可以：\n"
            "1. 换个方式描述您的问题\n"
            "2. 输入「转人工」联系人工客服\n"
            "3. 访问帮助中心：https://help.example.com"
        )
```

---

## 5. 部署与集成

### 5.1 Web 应用

完整的 FastAPI 后端服务结构：

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="智能客服 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

router = SkillRouter(retriever=retriever, order_service=order_service)

@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    """HTTP 接口——适合简单集成"""
    response = await router.route(req.message, req.history)
    return {"reply": response}

@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    """WebSocket 接口——适合实时对话"""
    await ws.accept()
    history = []
    while True:
        message = await ws.receive_text()
        response = await router.route(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        await ws.send_text(response)
```

### 5.2 即时通讯集成

以飞书机器人为例：

```python
from fastapi import Request
import httpx

FEISHU_APP_ID = "cli_xxx"
FEISHU_APP_SECRET = "xxx"

@app.post("/webhook/feishu")
async def feishu_webhook(request: Request):
    body = await request.json()

    # 飞书验证回调
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    message = event.get("message", {})
    text = json.loads(message.get("content", "{}")).get("text", "")
    chat_id = message.get("chat_id")

    # 调用客服路由
    reply = await router.route(text, [])

    # 回复消息
    async with httpx.AsyncClient() as http:
        await http.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            headers={"Authorization": f"Bearer {await get_tenant_token()}"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": reply}),
            },
            params={"receive_id_type": "chat_id"},
        )
    return {"code": 0}
```

### 5.3 API 服务

生产级 API 需要考虑认证、限流和监控：

```python
from fastapi import Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
import time

limiter = Limiter(key_func=get_remote_address)

async def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/v1/chat", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def chat_v1(req: ChatRequest, request: Request):
    start = time.time()
    response = await router.route(req.message, req.history)
    latency = time.time() - start

    # 记录指标
    logger.info(f"chat latency={latency:.2f}s tokens={count_tokens(response)}")

    return {
        "reply": response,
        "metadata": {"latency_ms": int(latency * 1000)},
    }
```

---

## 练习

1. **构建知识库客服机器人**：加载一组 Markdown 文档构建向量索引，实现带引用标注的 RAG 问答
2. **实现多轮对话追踪**：用查询改写解决指代消解问题，对比改写前后的检索效果
3. **添加意图路由**：实现 FAQ / 订单查询 / 转人工三个技能的路由分发

## 延伸阅读

- [LangChain Chatbot 教程](https://python.langchain.com/docs/tutorials/chatbot/) — 官方对话机器人构建指南
- [Vercel AI Chatbot 模板](https://github.com/vercel/ai-chatbot) — 全栈 AI 聊天应用参考实现
- [OpenAI Cookbook: How to build a RAG chatbot](https://cookbook.openai.com/) — RAG 问答最佳实践
- [RAG 评估框架 Ragas](https://docs.ragas.io/) — 系统化评估 RAG 系统质量