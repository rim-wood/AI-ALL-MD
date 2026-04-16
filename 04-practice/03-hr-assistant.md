# HR 智能助手

> 用 AI 重塑人力资源——简历筛选、员工问答与入职自动化

## 学习目标

- 构建简历智能筛选与候选人匹配系统
- 实现基于 RAG 的 HR 政策问答助手
- 掌握入职流程自动化 Agent 的设计
- 了解 HR 场景的数据隐私与合规要求

---

## 1. 简历筛选与候选人匹配

### 1.1 简历解析

将非结构化的简历文件（PDF/Word）转为结构化数据：

```python
from openai import OpenAI
import json
from pathlib import Path

client = OpenAI()

RESUME_PARSE_PROMPT = """解析以下简历内容，提取结构化信息。返回 JSON 格式。

简历内容：
{resume_text}

返回格式：
{{
  "name": "姓名",
  "email": "邮箱",
  "phone": "电话",
  "education": [
    {{"school": "", "degree": "", "major": "", "graduation_year": ""}}
  ],
  "experience": [
    {{"company": "", "title": "", "start_date": "", "end_date": "", "description": ""}}
  ],
  "skills": ["技能1", "技能2"],
  "years_of_experience": 0,
  "summary": "一句话概括候选人背景"
}}
"""

def parse_resume(resume_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": RESUME_PARSE_PROMPT.format(
            resume_text=resume_text
        )}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
```

**从 PDF 提取文本**：

```python
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def process_resume_file(file_path: str) -> dict:
    """完整的简历处理流程"""
    text = extract_text_from_pdf(file_path)
    return parse_resume(text)
```

### 1.2 职位匹配评分

根据 JD（职位描述）对候选人进行匹配评分：

```python
MATCH_PROMPT = """你是一个资深 HR。根据职位要求评估候选人的匹配度。

## 职位要求
{job_description}

## 候选人信息
{candidate_info}

## 评估维度
1. 技能匹配度（必备技能和加分技能）
2. 经验匹配度（行业经验、年限）
3. 教育背景匹配度
4. 综合潜力

返回 JSON：
{{
  "overall_score": 0-100,
  "skill_match": {{"score": 0-100, "matched": ["匹配的技能"], "missing": ["缺失的技能"]}},
  "experience_match": {{"score": 0-100, "comment": "说明"}},
  "education_match": {{"score": 0-100, "comment": "说明"}},
  "recommendation": "推荐/待定/不推荐",
  "summary": "一段话总结评估结论",
  "interview_suggestions": ["建议面试中重点考察的方向"]
}}
"""

def match_candidate(job_description: str, candidate_info: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": MATCH_PROMPT.format(
            job_description=job_description,
            candidate_info=json.dumps(candidate_info, ensure_ascii=False),
        )}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)
```

### 1.3 批量筛选流水线

处理大量简历的自动化流水线：

```python
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

class ScreeningResult(BaseModel):
    file_name: str
    candidate_name: str
    overall_score: int
    recommendation: str
    summary: str

def batch_screen(
    resume_dir: str, job_description: str, top_n: int = 10
) -> list[ScreeningResult]:
    """批量筛选简历，返回 Top N 候选人"""
    results = []

    pdf_files = list(Path(resume_dir).glob("*.pdf"))

    def process_one(pdf_path: Path) -> ScreeningResult | None:
        try:
            candidate = process_resume_file(str(pdf_path))
            match = match_candidate(job_description, candidate)
            return ScreeningResult(
                file_name=pdf_path.name,
                candidate_name=candidate.get("name", "未知"),
                overall_score=match["overall_score"],
                recommendation=match["recommendation"],
                summary=match["summary"],
            )
        except Exception as e:
            print(f"处理 {pdf_path.name} 失败: {e}")
            return None

    # 并发处理
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_one, pdf_files))

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results[:top_n]
```

> **公平性提醒：** AI 简历筛选可能继承训练数据中的偏见。建议：(1) 不要将姓名、性别、年龄等信息传入评分模型；(2) 定期审计筛选结果的多样性；(3) AI 评分仅作为参考，最终决策由人类做出。

---

## 2. HR 政策问答助手

### 2.1 知识库构建

将公司的 HR 政策文档（员工手册、考勤制度、福利政策等）构建为 RAG 知识库：

```python
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def build_hr_knowledge_base(docs_dir: str, persist_dir: str = "./hr_kb"):
    """构建 HR 知识库"""
    loaders = {
        ".pdf": PyPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }

    docs = []
    for path in Path(docs_dir).rglob("*"):
        loader_cls = loaders.get(path.suffix)
        if loader_cls:
            loaded = loader_cls(str(path)).load()
            # 添加元数据：文档类别
            category = path.parent.name  # 如 "考勤制度"、"福利政策"
            for doc in loaded:
                doc.metadata["category"] = category
                doc.metadata["source"] = path.name
            docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " "],
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_dir,
    )
    return vectorstore
```

### 2.2 政策问答

带权限控制的 HR 问答——不同角色看到不同范围的信息：

```python
from enum import Enum

class EmployeeRole(str, Enum):
    EMPLOYEE = "employee"       # 普通员工
    MANAGER = "manager"         # 经理
    HR = "hr"                   # HR
    EXECUTIVE = "executive"     # 高管

# 文档访问权限
CATEGORY_ACCESS = {
    EmployeeRole.EMPLOYEE: ["考勤制度", "福利政策", "行为规范", "常见问题"],
    EmployeeRole.MANAGER: ["考勤制度", "福利政策", "行为规范", "常见问题", "绩效管理", "团队管理"],
    EmployeeRole.HR: ["*"],  # 全部访问
    EmployeeRole.EXECUTIVE: ["*"],
}

HR_QA_PROMPT = """你是公司的 HR 政策助手。根据参考资料回答员工的问题。

## 参考资料
{context}

## 规则
- 只基于参考资料回答，不要编造政策内容
- 引用具体的制度条款，格式：[来源: 文档名]
- 涉及薪资、个人隐私等敏感信息时，引导员工联系 HR 部门
- 如果问题超出知识范围，回答"建议您联系 HR 部门（分机 8888）获取准确信息"

## 员工问题
{question}
"""

class HRAssistant:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def answer(self, question: str, role: EmployeeRole) -> str:
        # 1. 按权限过滤检索范围
        allowed = CATEGORY_ACCESS.get(role, [])
        if "*" in allowed:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5, "filter": {"category": {"$in": allowed}}}
            )

        # 2. 检索
        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[文档: {d.metadata['source']}]\n{d.page_content}" for d in docs
        )

        # 3. 生成回答
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": HR_QA_PROMPT.format(
                context=context, question=question
            )}],
            temperature=0.3,
        )
        return response.choices[0].message.content
```

### 2.3 常见问题场景

HR 助手需要处理的典型问题类型：

| 类别 | 示例问题 | 处理方式 |
|------|---------|---------|
| 考勤 | "迟到怎么扣款？" | RAG 检索考勤制度 |
| 假期 | "年假还剩几天？" | 调用 HR 系统 API |
| 福利 | "体检怎么预约？" | RAG + 流程引导 |
| 薪资 | "我的工资明细？" | 拒绝回答，引导联系 HR |
| 流程 | "怎么申请调岗？" | RAG + 表单链接 |

**集成 HR 系统 API**——查询个人数据：

```python
class HRSystemClient:
    """HR 系统 API 客户端"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def get_leave_balance(self, employee_id: str) -> dict:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                f"{self.base_url}/employees/{employee_id}/leave-balance",
                headers=self.headers,
            )
            return resp.json()

    async def get_attendance(self, employee_id: str, month: str) -> dict:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                f"{self.base_url}/employees/{employee_id}/attendance",
                headers=self.headers,
                params={"month": month},
            )
            return resp.json()
```

---

## 3. 入职流程自动化

### 3.1 入职 Agent 设计

用 Agent 自动化新员工入职流程——从 Offer 签署到第一天报到：

```python
from enum import Enum
from pydantic import BaseModel
from datetime import date

class OnboardingStep(str, Enum):
    OFFER_SENT = "offer_sent"
    OFFER_ACCEPTED = "offer_accepted"
    DOCS_COLLECTED = "docs_collected"
    ACCOUNT_CREATED = "account_created"
    EQUIPMENT_READY = "equipment_ready"
    MENTOR_ASSIGNED = "mentor_assigned"
    ORIENTATION_SCHEDULED = "orientation_scheduled"
    COMPLETED = "completed"

class OnboardingTask(BaseModel):
    step: OnboardingStep
    description: str
    assignee: str          # 负责人
    due_date: date | None
    completed: bool = False
    notes: str = ""

class OnboardingAgent:
    """入职流程自动化 Agent"""

    def __init__(self, hr_client: HRSystemClient):
        self.hr_client = hr_client
        self.tasks: dict[str, list[OnboardingTask]] = {}

    def create_onboarding_plan(
        self, employee_name: str, position: str, start_date: date
    ) -> list[OnboardingTask]:
        """根据职位自动生成入职计划"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"为以下新员工生成入职任务清单：\n"
                f"姓名：{employee_name}\n职位：{position}\n入职日期：{start_date}\n\n"
                "返回 JSON 数组，每个任务包含：step, description, assignee, due_date\n"
                "assignee 从以下角色选择：HR, IT, 直属经理, 行政\n"
                "任务应覆盖：证件收集、系统账号、设备准备、导师分配、入职培训"
            )}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        tasks_data = json.loads(response.choices[0].message.content)
        tasks = [OnboardingTask(**t) for t in tasks_data.get("tasks", [])]
        self.tasks[employee_name] = tasks
        return tasks

    async def check_and_notify(self, employee_name: str):
        """检查任务进度，发送提醒"""
        tasks = self.tasks.get(employee_name, [])
        overdue = [
            t for t in tasks
            if not t.completed and t.due_date and t.due_date < date.today()
        ]
        for task in overdue:
            await self._send_reminder(task)

    async def _send_reminder(self, task: OnboardingTask):
        """发送任务提醒（邮件/IM）"""
        message = (
            f"⏰ 入职任务逾期提醒\n"
            f"任务：{task.description}\n"
            f"负责人：{task.assignee}\n"
            f"截止日期：{task.due_date}"
        )
        # 调用通知服务...
        print(message)
```

### 3.2 入职文档自动生成

根据员工信息自动生成入职所需文档：

```python
OFFER_TEMPLATE = """根据以下信息生成 Offer Letter 内容。

员工信息：
- 姓名：{name}
- 职位：{position}
- 部门：{department}
- 入职日期：{start_date}
- 薪资：{salary}
- 汇报对象：{manager}

要求：
- 正式、专业的语气
- 包含职位、薪资、入职日期、试用期等关键信息
- 包含公司福利概述
- 包含签署截止日期（入职日期前 7 天）
"""

def generate_offer_letter(employee_info: dict) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": OFFER_TEMPLATE.format(**employee_info)}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def generate_welcome_email(employee_info: dict, mentor_name: str) -> str:
    """生成入职欢迎邮件"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"为新员工 {employee_info['name']} 生成入职欢迎邮件。\n"
            f"职位：{employee_info['position']}，部门：{employee_info['department']}\n"
            f"导师：{mentor_name}\n"
            f"入职日期：{employee_info['start_date']}\n\n"
            "邮件应包含：欢迎语、第一天安排、导师介绍、常用链接、联系方式"
        )}],
        temperature=0.5,
    )
    return response.choices[0].message.content
```

### 3.3 入职问答机器人

新员工专属的问答机器人，回答入职相关问题：

```python
ONBOARDING_QA_PROMPT = """你是新员工入职助手。帮助新员工了解公司和入职流程。

## 员工信息
姓名：{name}，职位：{position}，部门：{department}
入职日期：{start_date}，导师：{mentor}

## 入职进度
{onboarding_status}

## 公司信息
{company_info}

## 规则
- 友好、热情地回答新员工的问题
- 涉及具体薪资数字时，引导联系 HR
- 主动提供有用的信息（如 WiFi 密码、食堂位置等）
"""

class OnboardingChatbot:
    def __init__(self, employee_info: dict, knowledge_base):
        self.employee_info = employee_info
        self.kb = knowledge_base
        self.history: list[dict] = []

    def chat(self, question: str) -> str:
        # 检索相关入职信息
        docs = self.kb.as_retriever(search_kwargs={"k": 3}).invoke(question)
        company_info = "\n".join(d.page_content for d in docs)

        messages = [
            {"role": "system", "content": ONBOARDING_QA_PROMPT.format(
                **self.employee_info,
                onboarding_status=self._get_status(),
                company_info=company_info,
            )},
            *self.history,
            {"role": "user", "content": question},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.5,
        )
        reply = response.choices[0].message.content

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": reply})
        return reply
```

---

## 4. 员工自助服务

### 4.1 请假与考勤查询

将 HR 系统 API 封装为 Agent 工具，让员工通过自然语言完成自助操作：

```python
from datetime import datetime

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_leave_balance",
            "description": "查询员工的剩余假期天数",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "员工工号"},
                    "leave_type": {
                        "type": "string",
                        "enum": ["annual", "sick", "personal"],
                        "description": "假期类型：年假/病假/事假",
                    },
                },
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_leave_request",
            "description": "提交请假申请",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "leave_type": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "reason": {"type": "string"},
                },
                "required": ["employee_id", "leave_type", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_attendance",
            "description": "查询员工考勤记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "month": {"type": "string", "description": "YYYY-MM"},
                },
                "required": ["employee_id", "month"],
            },
        },
    },
]

async def handle_tool_call(name: str, args: dict, hr_client: HRSystemClient) -> str:
    """执行工具调用"""
    if name == "query_leave_balance":
        result = await hr_client.get_leave_balance(args["employee_id"])
        return json.dumps(result, ensure_ascii=False)
    elif name == "submit_leave_request":
        # 提交前需要员工确认
        return json.dumps({"status": "pending_confirmation", "details": args})
    elif name == "query_attendance":
        result = await hr_client.get_attendance(args["employee_id"], args["month"])
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": "未知工具"})

class EmployeeSelfService:
    """员工自助服务 Agent"""

    def __init__(self, employee_id: str, hr_client: HRSystemClient):
        self.employee_id = employee_id
        self.hr_client = hr_client

    async def chat(self, message: str, history: list[dict]) -> str:
        messages = [
            {"role": "system", "content": (
                f"你是员工自助服务助手。当前员工工号：{self.employee_id}。\n"
                "帮助员工查询假期、考勤，提交请假申请。\n"
                "提交请假前必须先确认信息，得到员工确认后再提交。"
            )},
            *history,
            {"role": "user", "content": message},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS,
        )
        msg = response.choices[0].message

        # 处理工具调用
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                args["employee_id"] = self.employee_id  # 强制使用当前员工 ID
                result = await handle_tool_call(tc.function.name, args, self.hr_client)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })

            final = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages,
            )
            return final.choices[0].message.content

        return msg.content
```

### 4.2 绩效辅助

帮助经理撰写绩效评语和发展建议：

```python
PERFORMANCE_PROMPT = """根据以下绩效数据，帮助经理撰写绩效评语。

## 员工信息
姓名：{name}，职位：{position}，入职时间：{hire_date}

## 绩效数据
- KPI 完成率：{kpi_completion}%
- 项目贡献：{projects}
- 同事评价摘要：{peer_feedback}
- 上期改进目标：{previous_goals}

## 要求
返回 JSON：
{{
  "overall_rating": "优秀/良好/合格/待改进",
  "strengths": ["优势1", "优势2"],
  "improvements": ["改进方向1", "改进方向2"],
  "review_text": "正式的绩效评语（200字以内）",
  "development_plan": ["下期发展目标1", "下期发展目标2"]
}}
"""

def generate_performance_review(employee_data: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": PERFORMANCE_PROMPT.format(**employee_data)}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)
```

---

## 5. 数据隐私与合规

### 5.1 PII 脱敏

HR 数据包含大量个人敏感信息，传入 LLM 前必须脱敏：

```python
import re

class PIIMasker:
    """个人信息脱敏"""

    PATTERNS = [
        (r"\b\d{17}[\dXx]\b", "[身份证号]"),                    # 身份证
        (r"\b1[3-9]\d{9}\b", "[手机号]"),                       # 手机号
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[邮箱]"),
        (r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", None),           # 日期保留
        (r"\b\d{16,19}\b", "[银行卡号]"),                       # 银行卡
    ]

    def mask(self, text: str) -> tuple[str, dict]:
        """脱敏文本，返回 (脱敏后文本, 映射表)"""
        mapping = {}
        masked = text
        for pattern, replacement in self.PATTERNS:
            if replacement is None:
                continue
            for match in re.finditer(pattern, masked):
                original = match.group()
                key = f"{replacement}_{len(mapping)}"
                mapping[key] = original
                masked = masked.replace(original, key, 1)
        return masked, mapping

    def unmask(self, text: str, mapping: dict) -> str:
        """还原脱敏文本"""
        result = text
        for key, original in mapping.items():
            result = result.replace(key, original)
        return result

# 使用示例
masker = PIIMasker()
text = "员工张三，手机 13800138000，身份证 110101199001011234"
masked_text, mapping = masker.mask(text)
# masked_text: "员工张三，手机 [手机号]_0，身份证 [身份证号]_1"
```

### 5.2 审计与合规

记录所有 AI 操作，满足合规审计要求：

```python
from datetime import datetime
from pydantic import BaseModel

class AuditLog(BaseModel):
    timestamp: datetime
    operator_id: str       # 操作人
    action: str            # 操作类型
    target_employee: str   # 涉及的员工
    data_accessed: list[str]  # 访问的数据类别
    ai_model: str
    input_masked: bool     # 输入是否已脱敏
    result_summary: str

class HRAuditLogger:
    def __init__(self, db_path: str = "./hr_audit.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, operator_id TEXT, action TEXT,
                target_employee TEXT, data_accessed TEXT,
                ai_model TEXT, input_masked INTEGER, result_summary TEXT
            )
        """)

    def log(self, entry: AuditLog):
        self.conn.execute(
            "INSERT INTO audit_logs VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp.isoformat(), entry.operator_id,
                entry.action, entry.target_employee,
                json.dumps(entry.data_accessed), entry.ai_model,
                int(entry.input_masked), entry.result_summary,
            ),
        )
        self.conn.commit()
```

### 5.3 安全边界

HR 助手必须严格遵守的安全规则：

```python
SAFETY_RULES = """
## 安全边界（不可违反）
1. 绝不透露其他员工的薪资、绩效等个人信息
2. 绝不代替 HR 做出雇佣/解雇/晋升决策
3. 涉及劳动纠纷时，建议咨询法务部门
4. 不对公司未公开的政策变更做出承诺
5. 所有涉及金额的回答必须标注"以实际发放为准"
"""

def build_hr_system_prompt(employee_role: EmployeeRole) -> str:
    base = "你是公司的 HR 智能助手。" + SAFETY_RULES
    if employee_role == EmployeeRole.EMPLOYEE:
        base += "\n当前用户是普通员工，只能查看自己的信息和公开政策。"
    elif employee_role == EmployeeRole.MANAGER:
        base += "\n当前用户是经理，可以查看团队成员的考勤和绩效概况。"
    elif employee_role == EmployeeRole.HR:
        base += "\n当前用户是 HR，可以访问所有员工信息和政策文档。"
    return base
```

---

## 练习

1. **构建简历筛选系统**：实现 PDF 简历解析 → 结构化提取 → JD 匹配评分的完整流程，批量处理 10 份简历并排名
2. **实现 HR 政策问答助手**：用公司员工手册构建 RAG 知识库，实现带角色权限控制的问答
3. **构建入职自动化 Agent**：根据新员工信息自动生成入职计划、Offer Letter 和欢迎邮件

## 延伸阅读

- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) — 构建知识库问答的基础
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — 工具调用实现 HR 系统集成
- [EU AI Act — HR 应用合规](https://artificialintelligenceact.eu/) — 欧盟 AI 法案对 HR 场景的要求
- [Responsible AI in Hiring](https://www.shrm.org/) — SHRM 关于 AI 招聘的最佳实践指南