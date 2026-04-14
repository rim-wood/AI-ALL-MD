# TypeScript 编程基础

> AI 应用的全栈语言——用 TypeScript 构建前端交互与后端服务

## 学习目标

- 掌握 TypeScript 类型系统在 AI 开发中的应用
- 理解 Node.js 异步编程与流式处理
- 能够使用 Vercel AI SDK 构建 AI 前端应用
- 熟悉 AI 开发中的 TypeScript 工具链

## 为什么需要 TypeScript

虽然 Python 是 AI/ML 的主力语言，但 TypeScript 在 AI 应用层有不可替代的优势：
- **前端交互**：对话式 UI、流式渲染、实时反馈
- **全栈开发**：Next.js + Vercel AI SDK 一站式方案
- **类型安全**：LLM 输出解析、API 接口定义
- **生态丰富**：LangChain.js、Vercel AI SDK、MCP TypeScript SDK

---

## 1. 类型系统

TypeScript 的类型系统在 AI 开发中用于定义 API 接口、LLM 响应结构和工具参数。

### 1.1 基础类型

```typescript
// 基本类型
const model: string = "gpt-4o";
const temperature: number = 0.7;
const stream: boolean = true;
const maxTokens: number | undefined = undefined;

// 函数签名
function chat(message: string, temperature?: number): Promise<string> {
  // ...
}
```

### 1.2 接口与类型别名

```typescript
// 消息接口 —— LLM 对话的基础结构
interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

// LLM 配置
interface LLMConfig {
  model: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

// 类型别名
type MessageRole = "system" | "user" | "assistant";
type ToolResult = { success: true; data: unknown } | { success: false; error: string };
```

### 1.3 泛型

```typescript
// 通用 API 响应
interface ApiResponse<T> {
  data: T;
  status: number;
  timestamp: Date;
}

// LLM 结构化输出解析
async function parseOutput<T>(response: string, schema: z.ZodType<T>): Promise<T> {
  return schema.parse(JSON.parse(response));
}

// 使用
const result = await parseOutput(llmResponse, UserInfoSchema);
```

### 1.4 Zod —— TypeScript 的 Pydantic

```typescript
import { z } from "zod";

// 定义 Schema（类似 Python 的 Pydantic）
const ChatRequestSchema = z.object({
  message: z.string().min(1),
  model: z.string().default("gpt-4o"),
  temperature: z.number().min(0).max(2).default(0.7),
});

type ChatRequest = z.infer<typeof ChatRequestSchema>;

// 验证输入
const request = ChatRequestSchema.parse(rawInput);

// 用于 LLM 结构化输出
const ExtractedInfoSchema = z.object({
  name: z.string(),
  age: z.number(),
  skills: z.array(z.string()),
});
```

## 2. 异步编程

### 2.1 Promise 与 async/await

```typescript
// 基础异步调用
async function callLLM(prompt: string): Promise<string> {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer <api-key>`,
    },
    body: JSON.stringify({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const data = await response.json();
  return data.choices[0].message.content;
}
```

### 2.2 并发控制

```typescript
// Promise.all 并发调用
async function batchCall(prompts: string[]): Promise<string[]> {
  return Promise.all(prompts.map((p) => callLLM(p)));
}

// 带并发限制
async function batchCallLimited(
  prompts: string[],
  maxConcurrent: number = 5
): Promise<string[]> {
  const results: string[] = [];
  for (let i = 0; i < prompts.length; i += maxConcurrent) {
    const batch = prompts.slice(i, i + maxConcurrent);
    const batchResults = await Promise.all(batch.map((p) => callLLM(p)));
    results.push(...batchResults);
  }
  return results;
}
```

### 2.3 错误处理

```typescript
async function callWithRetry(
  prompt: string,
  maxAttempts: number = 3
): Promise<string> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await callLLM(prompt);
    } catch (error) {
      if (attempt === maxAttempts - 1) throw error;
      await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
    }
  }
  throw new Error("Unreachable");
}
```

## 3. 流式处理

流式输出是 AI 应用的核心体验，TypeScript 在浏览器和 Node.js 中都有完善的 Stream API。

### 3.1 ReadableStream

```typescript
// 消费 SSE 流式响应
async function* streamChat(prompt: string): AsyncGenerator<string> {
  const response = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message: prompt }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value);
  }
}

// 使用
for await (const chunk of streamChat("Hello")) {
  process.stdout.write(chunk);
}
```

### 3.2 Server-Sent Events

```typescript
// 服务端 SSE（Node.js / Express）
import { Router } from "express";

const router = Router();

router.post("/chat/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  for await (const chunk of streamFromLLM(req.body.message)) {
    res.write(`data: ${JSON.stringify({ content: chunk })}\n\n`);
  }
  res.write("data: [DONE]\n\n");
  res.end();
});
```

## 4. Vercel AI SDK

TypeScript AI 开发的核心工具包，提供统一的模型接口和前端集成。

### 4.1 基础调用

```typescript
import { generateText, streamText } from "ai";
import { openai } from "@ai-sdk/openai";

// 同步生成
const { text } = await generateText({
  model: openai("gpt-4o"),
  prompt: "什么是 RAG？",
});

// 流式生成
const result = streamText({
  model: openai("gpt-4o"),
  prompt: "解释 Transformer 架构",
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

### 4.2 结构化输出

```typescript
import { generateObject } from "ai";
import { z } from "zod";

const { object } = await generateObject({
  model: openai("gpt-4o"),
  schema: z.object({
    name: z.string(),
    summary: z.string(),
    tags: z.array(z.string()),
  }),
  prompt: "分析这篇文章：...",
});
```

### 4.3 工具调用

```typescript
import { generateText, tool } from "ai";

const result = await generateText({
  model: openai("gpt-4o"),
  tools: {
    weather: tool({
      description: "获取天气信息",
      parameters: z.object({
        city: z.string().describe("城市名称"),
      }),
      execute: async ({ city }) => {
        return `${city}今天晴，25°C`;
      },
    }),
  },
  prompt: "北京今天天气怎么样？",
});
```

### 4.4 React 集成（useChat）

```tsx
"use client";
import { useChat } from "@ai-sdk/react";

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();

  return (
    <div>
      {messages.map((m) => (
        <div key={m.id}>
          <strong>{m.role}:</strong> {m.content}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit">发送</button>
      </form>
    </div>
  );
}
```

## 5. Node.js 运行时

### 5.1 文件操作

```typescript
import { readFile, writeFile } from "fs/promises";

// 读取文档用于 RAG
async function loadDocument(path: string): Promise<string> {
  return readFile(path, "utf-8");
}

// 保存对话历史
async function saveHistory(messages: Message[], path: string): Promise<void> {
  await writeFile(path, JSON.stringify(messages, null, 2));
}
```

### 5.2 环境变量与配置

```typescript
// .env 文件
// OPENAI_API_KEY=sk-...
// ANTHROPIC_API_KEY=sk-ant-...

import "dotenv/config";

const config = {
  openaiKey: process.env.OPENAI_API_KEY!,
  model: process.env.MODEL ?? "gpt-4o",
  port: parseInt(process.env.PORT ?? "3000"),
};
```

## 6. 包管理与项目配置

### 6.1 pnpm（推荐）

```bash
# 创建项目
pnpm init
pnpm add ai @ai-sdk/openai zod
pnpm add -D typescript @types/node tsx

# 运行
pnpm tsx src/index.ts
```

### 6.2 tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "outDir": "dist"
  },
  "include": ["src"]
}
```

### 6.3 常用 AI 开发依赖

```json
{
  "dependencies": {
    "ai": "^6.0.0",
    "@ai-sdk/openai": "^1.0.0",
    "@ai-sdk/anthropic": "^1.0.0",
    "zod": "^3.23.0",
    "next": "^15.0.0"
  }
}
```

---

## Python vs TypeScript 选型

| 维度 | Python | TypeScript |
|------|--------|------------|
| 模型训练/微调 | ✅ 首选 | ❌ 不适合 |
| 后端 API | ✅ FastAPI | ✅ Next.js / Express |
| 前端 UI | ❌ | ✅ 首选 |
| AI 框架生态 | ✅ 最丰富 | ✅ 快速追赶 |
| 数据处理 | ✅ pandas/numpy | ⚠️ 有限 |
| 全栈开发 | ⚠️ 需配合前端 | ✅ 一站式 |

**建议**：两者都学，Python 做后端和 AI 逻辑，TypeScript 做前端和全栈应用。

---

## 练习

1. 用 Vercel AI SDK 构建一个流式对话应用
2. 用 Zod 定义 LLM 结构化输出 Schema 并解析响应
3. 实现一个带工具调用的 TypeScript Agent

## 延伸阅读

- [TypeScript 官方文档](https://www.typescriptlang.org/docs/)
- [Vercel AI SDK 文档](https://ai-sdk.dev/)
- [Zod 文档](https://zod.dev/)
- [LangChain.js 文档](https://js.langchain.com/)
- [Node.js 官方文档](https://nodejs.org/docs/)
