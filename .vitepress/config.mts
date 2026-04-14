import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/',
  title: 'AI 应用开发指南',
  description: '从基础知识到生产落地，系统掌握 AI 应用开发全栈技能',
  lang: 'zh-CN',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '开始学习', link: '/01-fundamentals/01-python-basics' }
    ],
    sidebar: [
      {
        text: '第一部分：基础知识',
        items: [
          { text: 'Python 编程基础', link: '/01-fundamentals/01-python-basics' },
          { text: 'TypeScript 编程基础', link: '/01-fundamentals/02-typescript-basics' },
          { text: 'AI/ML 基础', link: '/01-fundamentals/03-ai-ml-basics' },
          { text: 'Prompt Engineering', link: '/01-fundamentals/04-prompt-engineering' }
        ]
      },
      {
        text: '第二部分：核心技术栈',
        items: [
          { text: 'LLM API 与模型选型', link: '/02-core-tech/01-llm-api-and-models' },
          { text: 'Function Calling', link: '/02-core-tech/02-function-calling' },
          { text: 'RAG', link: '/02-core-tech/03-rag' },
          { text: 'Embedding 与向量搜索', link: '/02-core-tech/04-embeddings-and-vector-search' }
        ]
      },
      {
        text: '第三部分：框架与工具',
        items: [
          { text: 'LangChain / LangGraph', link: '/03-frameworks/01-langchain-langgraph' },
          { text: 'LlamaIndex', link: '/03-frameworks/02-llamaindex' },
          { text: 'CrewAI / AutoGen', link: '/03-frameworks/03-crewai-autogen' },
          { text: 'AI Agent 开发', link: '/03-frameworks/04-ai-agent-development' },
          { text: 'MCP 协议', link: '/03-frameworks/05-mcp-protocol' }
        ]
      },
      {
        text: '第四部分：应用实践',
        items: [
          { text: '智能客服与问答', link: '/04-practice/01-chatbot-and-qa' },
          { text: '代码助手', link: '/04-practice/02-code-assistant' },
          { text: '数据分析 Agent', link: '/04-practice/03-data-analysis-agent' },
          { text: '多模态应用', link: '/04-practice/04-multimodal-apps' },
          { text: '前端与用户体验', link: '/04-practice/05-frontend-and-ux' }
        ]
      },
      {
        text: '第五部分：生产部署与运维',
        items: [
          { text: 'LLMOps', link: '/05-production/01-llmops' },
          { text: '监控与可观测性', link: '/05-production/02-monitoring' },
          { text: '安全与合规', link: '/05-production/03-security-and-compliance' },
          { text: '性能优化与成本控制', link: '/05-production/04-performance-and-cost' }
        ]
      },
      {
        text: '第六部分：进阶与前沿',
        items: [
          { text: 'Agentic 系统设计', link: '/06-advanced/01-agentic-system-design' },
          { text: '微调与模型定制', link: '/06-advanced/02-fine-tuning' },
          { text: '前沿方向', link: '/06-advanced/03-frontier-topics' }
        ]
      }
    ],
    outline: { level: [2, 3], label: '目录' },
    search: { provider: 'local' },
    lastUpdated: { text: '最后更新' }
  }
})
