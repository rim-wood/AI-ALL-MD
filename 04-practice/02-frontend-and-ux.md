# 前端与用户体验

> AI 应用的用户界面与交互设计

## 学习目标

- 实现流式输出与实时交互，掌握 SSE 和 WebSocket 方案
- 掌握对话式 UI 设计模式，包括 Markdown 渲染和富媒体展示
- 使用 Vercel AI SDK 快速构建 AI 前端
- 建立用户反馈收集与利用机制

---

## 1. 流式输出

### 1.1 SSE（Server-Sent Events）

SSE 是 AI 应用最常用的流式方案——服务端单向推送，实现简单，兼容性好。

**服务端实现（FastAPI）**：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

@app.post("/api/chat/stream")
async def chat_stream(message: str):
    """SSE 流式对话接口"""
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                # SSE 格式：data: {json}\n\n
                yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

**客户端消费（JavaScript）**：

```javascript
async function streamChat(message) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // 保留不完整的行

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;
        const { content } = JSON.parse(data);
        appendToMessage(content); // 追加到 UI
      }
    }
  }
}
```

### 1.2 WebSocket

WebSocket 适合需要双向通信的场景（如语音对话、协作编辑）：

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    history = []

    try:
        while True:
            data = await ws.receive_json()
            message = data["message"]
            history.append({"role": "user", "content": message})

            # 流式返回
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=history,
                stream=True,
            )
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    await ws.send_json({"type": "delta", "content": content})

            await ws.send_json({"type": "done"})
            history.append({"role": "assistant", "content": full_response})
    except WebSocketDisconnect:
        pass
```

```javascript
// 客户端 WebSocket
const ws = new WebSocket("ws://localhost:8000/ws/chat");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "delta") {
    appendToMessage(data.content);
  } else if (data.type === "done") {
    finalizeMessage();
  }
};

function sendMessage(text) {
  ws.send(JSON.stringify({ message: text }));
}
```

**SSE vs WebSocket 选型**：

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 单向（服务端→客户端） | 双向 |
| 协议 | HTTP | 独立协议 |
| 自动重连 | 浏览器内置 | 需手动实现 |
| 适用场景 | 对话流式输出 | 语音对话、实时协作 |
| 复杂度 | 低 | 中 |

> **实战建议：** 大多数 AI 对话应用用 SSE 就够了。只有需要客户端持续发送数据（如语音流）时才用 WebSocket。

### 1.3 前端流式渲染

逐字显示效果的实现：

```tsx
// React 流式消息组件
import { useState, useRef, useEffect } from "react";

function StreamingMessage({ content }: { content: string }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [content]);

  return (
    <div className="message assistant">
      <MarkdownRenderer content={content} />
      {/* 打字光标动画 */}
      {content && <span className="cursor blink">▊</span>}
      <div ref={endRef} />
    </div>
  );
}

// CSS 打字光标动画
const cursorStyle = `
  .cursor.blink {
    animation: blink 1s step-end infinite;
  }
  @keyframes blink {
    50% { opacity: 0; }
  }
`;
```

**流式 Markdown 渲染的挑战**：流式输出时 Markdown 可能处于不完整状态（如代码块未闭合），需要特殊处理：

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

function MarkdownRenderer({ content }: { content: string }) {
  // 处理未闭合的代码块
  const processedContent = ensureClosedCodeBlocks(content);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const isInline = !match;
          return isInline ? (
            <code className="inline-code" {...props}>{children}</code>
          ) : (
            <SyntaxHighlighter language={match[1]} PreTag="div">
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          );
        },
      }}
    >
      {processedContent}
    </ReactMarkdown>
  );
}

function ensureClosedCodeBlocks(content: string): string {
  const codeBlockCount = (content.match(/```/g) || []).length;
  if (codeBlockCount % 2 !== 0) {
    return content + "\n```";
  }
  return content;
}
```

---

## 2. 对话式 UI

### 2.1 消息组件

一个完整的对话 UI 需要区分不同类型的消息：

```tsx
type MessageRole = "user" | "assistant" | "system" | "tool";

interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  metadata?: {
    model?: string;
    tokens?: number;
    tool_calls?: ToolCall[];
    sources?: Source[];  // RAG 引用来源
  };
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${message.role}`}>
      {/* 头像 */}
      <div className="avatar">
        {isUser ? "👤" : "🤖"}
      </div>

      {/* 消息内容 */}
      <div className="content">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <MarkdownRenderer content={message.content} />
        )}

        {/* RAG 引用来源 */}
        {message.metadata?.sources && (
          <SourceList sources={message.metadata.sources} />
        )}

        {/* 工具调用状态 */}
        {message.metadata?.tool_calls && (
          <ToolCallDisplay calls={message.metadata.tool_calls} />
        )}
      </div>

      {/* 操作按钮 */}
      <div className="actions">
        <CopyButton text={message.content} />
        {!isUser && <FeedbackButtons messageId={message.id} />}
      </div>
    </div>
  );
}
```

### 2.2 Markdown 渲染

AI 回复通常包含 Markdown 格式，需要支持代码高亮、表格和 LaTeX：

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

function FullMarkdownRenderer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        // 代码块：带语言标签和复制按钮
        code({ className, children }) {
          const match = /language-(\w+)/.exec(className || "");
          if (!match) {
            return <code className="inline-code">{children}</code>;
          }
          return (
            <div className="code-block">
              <div className="code-header">
                <span className="language-tag">{match[1]}</span>
                <CopyButton text={String(children)} />
              </div>
              <SyntaxHighlighter language={match[1]} style={oneDark}>
                {String(children).replace(/\n$/, "")}
              </SyntaxHighlighter>
            </div>
          );
        },
        // 表格：添加横向滚动
        table({ children }) {
          return (
            <div className="table-wrapper">
              <table>{children}</table>
            </div>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

### 2.3 富媒体展示

AI 回复可能包含图片、图表和文件：

```tsx
interface RichContent {
  type: "text" | "image" | "chart" | "file" | "code_execution";
  content: string;
  metadata?: Record<string, unknown>;
}

function RichMessageContent({ parts }: { parts: RichContent[] }) {
  return (
    <div className="rich-content">
      {parts.map((part, i) => {
        switch (part.type) {
          case "text":
            return <MarkdownRenderer key={i} content={part.content} />;
          case "image":
            return (
              <figure key={i}>
                <img src={part.content} alt="" className="generated-image" />
                <figcaption>{part.metadata?.caption as string}</figcaption>
              </figure>
            );
          case "chart":
            return (
              <div key={i} className="chart-container"
                dangerouslySetInnerHTML={{ __html: part.content }}
              />
            );
          case "file":
            return (
              <a key={i} href={part.content} download className="file-download">
                📎 {part.metadata?.filename as string}
              </a>
            );
          case "code_execution":
            return (
              <div key={i} className="code-execution">
                <details>
                  <summary>代码执行结果</summary>
                  <pre>{part.content}</pre>
                </details>
              </div>
            );
        }
      })}
    </div>
  );
}
```

### 2.4 工具调用可视化

展示 Agent 的工具调用过程，增加透明度：

```tsx
interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "running" | "completed" | "failed";
  result?: string;
  duration_ms?: number;
}

function ToolCallDisplay({ calls }: { calls: ToolCall[] }) {
  return (
    <div className="tool-calls">
      {calls.map((call) => (
        <div key={call.id} className={`tool-call ${call.status}`}>
          <div className="tool-header">
            <StatusIcon status={call.status} />
            <span className="tool-name">{call.name}</span>
            {call.duration_ms && (
              <span className="duration">{call.duration_ms}ms</span>
            )}
          </div>

          {/* 可折叠的详情 */}
          <details>
            <summary>查看详情</summary>
            <div className="tool-args">
              <strong>参数：</strong>
              <pre>{JSON.stringify(call.arguments, null, 2)}</pre>
            </div>
            {call.result && (
              <div className="tool-result">
                <strong>结果：</strong>
                <pre>{call.result}</pre>
              </div>
            )}
          </details>
        </div>
      ))}
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  const icons = {
    running: "⏳",
    completed: "✅",
    failed: "❌",
  };
  return <span>{icons[status] || "❓"}</span>;
}
```

---

## 3. UI 组件库

### 3.1 Vercel AI SDK

Vercel AI SDK 是构建 AI 前端最快的方式，提供了开箱即用的 React Hooks：

```tsx
// Next.js + Vercel AI SDK
// app/api/chat/route.ts — 服务端
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai("gpt-4o-mini"),
    messages,
    system: "你是一个友好的中文助手。",
  });

  return result.toDataStreamResponse();
}
```

```tsx
// app/page.tsx — 客户端
"use client";
import { useChat } from "@ai-sdk/react";

export default function ChatPage() {
  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } =
    useChat({ api: "/api/chat" });

  return (
    <div className="chat-container">
      {/* 消息列表 */}
      <div className="messages">
        {messages.map((m) => (
          <div key={m.id} className={`message ${m.role}`}>
            <MarkdownRenderer content={m.content} />
          </div>
        ))}
      </div>

      {/* 输入框 */}
      <form onSubmit={handleSubmit} className="input-area">
        <textarea
          value={input}
          onChange={handleInputChange}
          placeholder="输入消息..."
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        {isLoading ? (
          <button type="button" onClick={stop}>停止</button>
        ) : (
          <button type="submit">发送</button>
        )}
      </form>
    </div>
  );
}
```

**带工具调用的 AI SDK**：

```tsx
// 服务端：定义工具
import { streamText, tool } from "ai";
import { z } from "zod";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai("gpt-4o"),
    messages,
    tools: {
      getWeather: tool({
        description: "获取指定城市的天气",
        parameters: z.object({
          city: z.string().describe("城市名称"),
        }),
        execute: async ({ city }) => {
          // 调用天气 API
          return { city, temperature: 22, condition: "晴" };
        },
      }),
    },
  });

  return result.toDataStreamResponse();
}
```

### 3.2 开源 Chat UI

快速搭建完整的对话界面：

| 项目 | 技术栈 | 特点 |
|------|--------|------|
| [Open WebUI](https://github.com/open-webui/open-webui) | Svelte | 功能最全，支持多模型、RAG、插件 |
| [Lobe Chat](https://github.com/lobehub/lobe-chat) | Next.js | UI 精美，插件生态丰富 |
| [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) | Next.js | 简洁，适合二次开发 |

**基于 Open WebUI 快速部署**：

```bash
# Docker 一键部署
docker run -d -p 3000:8080 \
  -e OPENAI_API_KEY=sk-xxx \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### 3.3 自定义组件

从零构建 React 对话组件：

```tsx
import { useState, useCallback } from "react";

interface ChatHookOptions {
  api: string;
  onError?: (error: Error) => void;
}

function useSimpleChat({ api, onError }: ChatHookOptions) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await fetch(api, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...messages, userMessage].map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
        signal: controller.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";
      const assistantId = crypto.randomUUID();

      // 添加空的 assistant 消息
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", timestamp: new Date() },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        for (const line of text.split("\n")) {
          if (line.startsWith("data: ") && line.slice(6) !== "[DONE]") {
            const { content } = JSON.parse(line.slice(6));
            assistantContent += content;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: assistantContent } : m
              )
            );
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError?.(err as Error);
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  }, [api, messages, onError]);

  const stop = useCallback(() => {
    abortController?.abort();
  }, [abortController]);

  return { messages, sendMessage, isLoading, stop };
}
```

---

## 4. 交互模式

### 4.1 加载状态

AI 响应通常需要几秒，良好的加载状态能显著提升体验：

```tsx
function LoadingIndicator({ type }: { type: "dots" | "skeleton" | "typing" }) {
  if (type === "dots") {
    return (
      <div className="loading-dots">
        <span>●</span><span>●</span><span>●</span>
      </div>
    );
  }

  if (type === "skeleton") {
    return (
      <div className="skeleton-message">
        <div className="skeleton-line" style={{ width: "80%" }} />
        <div className="skeleton-line" style={{ width: "60%" }} />
        <div className="skeleton-line" style={{ width: "70%" }} />
      </div>
    );
  }

  // typing: 显示"AI 正在输入..."
  return <div className="typing-indicator">AI 正在思考...</div>;
}
```

```css
/* 骨架屏动画 */
.skeleton-line {
  height: 16px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 8px;
}
@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* 三点跳动动画 */
.loading-dots span {
  animation: bounce 1.4s infinite ease-in-out;
  display: inline-block;
  margin: 0 2px;
}
.loading-dots span:nth-child(1) { animation-delay: 0s; }
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-8px); }
}
```

### 4.2 错误处理

友好的错误提示和自动重试：

```tsx
function ErrorMessage({
  error,
  onRetry,
}: {
  error: Error;
  onRetry: () => void;
}) {
  const errorMessages: Record<string, string> = {
    "429": "请求太频繁，请稍后再试",
    "500": "服务暂时不可用，请稍后再试",
    "503": "模型正在加载中，请稍等",
    network: "网络连接失败，请检查网络",
  };

  const message = errorMessages[getErrorCode(error)] || "出了点问题，请重试";

  return (
    <div className="error-message">
      <span className="error-icon">⚠️</span>
      <span>{message}</span>
      <button onClick={onRetry} className="retry-button">
        重试
      </button>
    </div>
  );
}

// 自动重试逻辑
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 3
): Promise<Response> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);
      if (response.ok || response.status < 500) return response;
    } catch (err) {
      if (i === maxRetries - 1) throw err;
    }
    // 指数退避
    await new Promise((r) => setTimeout(r, Math.pow(2, i) * 1000));
  }
  throw new Error("Max retries exceeded");
}
```

### 4.3 中断与取消

用户应该能随时停止生成：

```tsx
function ChatInput({
  onSend,
  onStop,
  isLoading,
}: {
  onSend: (msg: string) => void;
  onStop: () => void;
  isLoading: boolean;
}) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input.trim());
      setInput("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="输入消息... (Shift+Enter 换行)"
        disabled={isLoading}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
          }
        }}
      />
      {isLoading ? (
        <button type="button" onClick={onStop} className="stop-btn">
          ⏹ 停止
        </button>
      ) : (
        <button type="submit" disabled={!input.trim()}>
          发送 ↑
        </button>
      )}
    </form>
  );
}
```

---

## 5. 用户反馈

### 5.1 反馈收集

收集用户对 AI 回复的评价：

```tsx
function FeedbackButtons({ messageId }: { messageId: string }) {
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const [showDetail, setShowDetail] = useState(false);

  const submitFeedback = async (type: "up" | "down", detail?: string) => {
    setFeedback(type);
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messageId, type, detail }),
    });
  };

  return (
    <div className="feedback">
      <button
        className={feedback === "up" ? "active" : ""}
        onClick={() => submitFeedback("up")}
        aria-label="有帮助"
      >
        👍
      </button>
      <button
        className={feedback === "down" ? "active" : ""}
        onClick={() => {
          setFeedback("down");
          setShowDetail(true);
        }}
        aria-label="没帮助"
      >
        👎
      </button>

      {/* 负面反馈时收集详细原因 */}
      {showDetail && (
        <FeedbackDetailForm
          onSubmit={(detail) => {
            submitFeedback("down", detail);
            setShowDetail(false);
          }}
          onCancel={() => setShowDetail(false)}
        />
      )}
    </div>
  );
}

function FeedbackDetailForm({
  onSubmit,
  onCancel,
}: {
  onSubmit: (detail: string) => void;
  onCancel: () => void;
}) {
  const reasons = [
    "回答不准确",
    "回答不完整",
    "回答与问题无关",
    "格式或表达不好",
    "其他",
  ];

  return (
    <div className="feedback-detail">
      <p>请选择原因：</p>
      {reasons.map((reason) => (
        <button key={reason} onClick={() => onSubmit(reason)}>
          {reason}
        </button>
      ))}
      <button onClick={onCancel}>取消</button>
    </div>
  );
}
```

### 5.2 反馈数据利用

将用户反馈转化为评估数据集和改进依据：

```python
# 服务端：反馈收集与分析
from pydantic import BaseModel
from datetime import datetime

class FeedbackEntry(BaseModel):
    message_id: str
    type: str  # "up" or "down"
    detail: str | None = None
    timestamp: datetime = datetime.now()
    conversation_context: dict = {}  # 对话上下文

class FeedbackAnalyzer:
    """分析反馈数据，指导优化"""

    def get_negative_patterns(self, feedbacks: list[FeedbackEntry]) -> dict:
        """分析负面反馈的模式"""
        negatives = [f for f in feedbacks if f.type == "down"]
        # 按原因分组统计
        reason_counts = {}
        for f in negatives:
            reason = f.detail or "未说明"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return dict(sorted(reason_counts.items(), key=lambda x: -x[1]))

    def build_eval_dataset(self, feedbacks: list[FeedbackEntry]) -> list[dict]:
        """从反馈构建评估数据集"""
        dataset = []
        for f in feedbacks:
            if f.conversation_context:
                dataset.append({
                    "input": f.conversation_context.get("question", ""),
                    "expected_quality": "good" if f.type == "up" else "bad",
                    "actual_response": f.conversation_context.get("response", ""),
                    "feedback_reason": f.detail,
                })
        return dataset
```

### 5.3 对话导出

支持用户导出和分享对话：

```tsx
function ExportButton({ messages }: { messages: Message[] }) {
  const exportAsMarkdown = () => {
    const md = messages
      .map((m) => {
        const role = m.role === "user" ? "**用户**" : "**AI**";
        return `${role}：\n\n${m.content}\n`;
      })
      .join("\n---\n\n");

    downloadFile(`chat-${Date.now()}.md`, md, "text/markdown");
  };

  const exportAsJSON = () => {
    const data = JSON.stringify(messages, null, 2);
    downloadFile(`chat-${Date.now()}.json`, data, "application/json");
  };

  const copyShareLink = async () => {
    // 上传对话到服务端，获取分享链接
    const res = await fetch("/api/share", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    const { shareUrl } = await res.json();
    await navigator.clipboard.writeText(shareUrl);
    alert("分享链接已复制到剪贴板");
  };

  return (
    <div className="export-menu">
      <button onClick={exportAsMarkdown}>导出 Markdown</button>
      <button onClick={exportAsJSON}>导出 JSON</button>
      <button onClick={copyShareLink}>复制分享链接</button>
    </div>
  );
}

function downloadFile(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

---

## 练习

1. **用 Vercel AI SDK 构建流式对话界面**：实现完整的 Next.js 对话应用，包含流式输出、Markdown 渲染和代码高亮
2. **实现 Markdown + 代码高亮的消息渲染**：支持 GFM 表格、LaTeX 公式、代码块复制，处理流式输出时的未闭合标签
3. **添加用户反馈收集功能**：实现 👍👎 反馈、负面反馈原因收集、反馈数据统计面板

## 延伸阅读

- [Vercel AI SDK 文档](https://sdk.vercel.ai/) — 官方 AI 前端开发工具包
- [Next.js AI Chatbot](https://github.com/vercel/ai-chatbot) — Vercel 官方全栈 AI 聊天模板
- [Open WebUI](https://github.com/open-webui/open-webui) — 功能最全的开源 Chat UI
- [Lobe Chat](https://github.com/lobehub/lobe-chat) — 高颜值开源 AI 对话应用
- [React Markdown](https://github.com/remarkjs/react-markdown) — React Markdown 渲染组件