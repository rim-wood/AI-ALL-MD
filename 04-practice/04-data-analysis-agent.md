# 数据分析 Agent

> 用自然语言驱动数据查询与可视化

## 学习目标

- 实现 Text-to-SQL 自然语言查询，掌握 Schema 提供与 SQL 安全策略
- 构建数据可视化 Agent，自动选择图表类型并生成代码
- 实现 CSV/Excel 文件的沙箱分析
- 了解与 BI 工具的集成方案及数据安全控制

---

## 1. Text-to-SQL

### 1.1 基本原理

Text-to-SQL 的核心流程：

```
用户自然语言 → 理解意图 → 生成 SQL → 验证 SQL → 执行查询 → 解读结果
```

最小可用实现：

```python
from openai import OpenAI
import sqlite3
import json

client = OpenAI()

TEXT2SQL_PROMPT = """你是一个 SQL 专家。根据用户的自然语言问题生成 SQL 查询。

## 数据库 Schema
{schema}

## 规则
- 只生成 SELECT 查询，禁止 INSERT/UPDATE/DELETE
- 返回 JSON 格式：{{"sql": "SELECT ...", "explanation": "查询说明"}}
- 如果问题无法用当前 Schema 回答，返回 {{"sql": null, "explanation": "原因"}}

## 用户问题
{question}
"""

def text_to_sql(question: str, schema: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": TEXT2SQL_PROMPT.format(
            schema=schema, question=question
        )}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

def query_database(db_path: str, question: str, schema: str) -> str:
    """完整的 Text-to-SQL 查询流程"""
    # 1. 生成 SQL
    result = text_to_sql(question, schema)
    if not result.get("sql"):
        return f"无法生成查询：{result['explanation']}"

    sql = result["sql"]

    # 2. 验证 SQL 安全性
    if not is_safe_sql(sql):
        return "生成的 SQL 包含不安全操作，已拒绝执行。"

    # 3. 执行查询
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()

    # 4. 用 LLM 解读结果
    return interpret_results(question, sql, columns, rows)
```

### 1.2 Schema 提供策略

Schema 的提供方式直接影响 SQL 生成质量。全量 Schema 在表多时会超出上下文，需要筛选策略。

**自动提取 Schema**：

```python
def extract_schema(db_path: str) -> str:
    """从数据库自动提取 Schema 信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        # 获取建表语句
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}'")
        create_sql = cursor.fetchone()[0]

        # 获取示例数据（帮助 LLM 理解数据格式）
        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
        sample_rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        schema_parts.append(f"-- 表: {table}\n{create_sql};")
        if sample_rows:
            sample = "\n".join(
                str(dict(zip(columns, row))) for row in sample_rows
            )
            schema_parts.append(f"-- 示例数据:\n{sample}")

    conn.close()
    return "\n\n".join(schema_parts)
```

**相关表筛选**——大型数据库只提供相关表的 Schema：

```python
TABLE_SELECT_PROMPT = """从以下数据库表列表中，选出回答用户问题所需的表。

表列表：
{table_list}

用户问题：{question}

返回 JSON：{{"tables": ["table1", "table2"]}}
"""

def select_relevant_tables(question: str, db_path: str) -> list[str]:
    """用 LLM 筛选相关表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    # 获取每个表的列信息作为摘要
    table_list = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        table_list.append(f"- {table}: {', '.join(cols)}")
    conn.close()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": TABLE_SELECT_PROMPT.format(
            table_list="\n".join(table_list), question=question
        )}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)["tables"]
```

### 1.3 SQL 验证与安全

生成的 SQL 必须经过安全验证才能执行：

```python
import sqlparse

def is_safe_sql(sql: str) -> bool:
    """验证 SQL 安全性"""
    parsed = sqlparse.parse(sql)
    for statement in parsed:
        stmt_type = statement.get_type()
        # 只允许 SELECT 查询
        if stmt_type and stmt_type.upper() != "SELECT":
            return False

    sql_upper = sql.upper()
    # 禁止危险操作
    dangerous_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "CREATE", "TRUNCATE", "EXEC", "EXECUTE",
    ]
    for keyword in dangerous_keywords:
        # 确保是独立关键词，不是列名的一部分
        if f" {keyword} " in f" {sql_upper} ":
            return False

    return True

def execute_with_limits(db_path: str, sql: str, timeout: int = 10, max_rows: int = 1000):
    """带限制的安全执行"""
    conn = sqlite3.connect(db_path)
    conn.execute(f"PRAGMA max_page_count = 1000")  # 限制读取量

    # 自动添加 LIMIT
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {max_rows}"

    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()
```

### 1.4 复杂查询处理

对于复杂问题，可以分步生成 SQL：

```python
COMPLEX_SQL_PROMPT = """用户的问题可能需要复杂的 SQL 查询。请分步思考：

## 数据库 Schema
{schema}

## 用户问题
{question}

## 请按以下步骤分析：
1. 理解用户意图
2. 确定需要的表和字段
3. 确定 JOIN 关系
4. 确定过滤条件和聚合方式
5. 生成最终 SQL

返回 JSON：
{{
  "thinking": "分步思考过程",
  "sql": "最终 SQL",
  "explanation": "查询说明"
}}
"""

def interpret_results(
    question: str, sql: str, columns: list, rows: list
) -> str:
    """用 LLM 解读查询结果"""
    # 限制结果大小
    display_rows = rows[:20]
    result_text = f"列: {columns}\n数据({len(rows)}行):\n"
    result_text += "\n".join(str(row) for row in display_rows)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"用户问题：{question}\n"
            f"执行的 SQL：{sql}\n"
            f"查询结果：\n{result_text}\n\n"
            "请用自然语言总结查询结果，回答用户的问题。"
        )}],
        temperature=0.3,
    )
    return response.choices[0].message.content
```

---

## 2. 数据可视化

### 2.1 自动图表生成

根据数据特征自动选择合适的图表类型：

```python
CHART_PROMPT = """根据数据分析结果，选择合适的图表类型并生成 Python 可视化代码。

## 数据
列名: {columns}
数据样本（前5行）:
{sample_data}
总行数: {total_rows}

## 用户需求
{requirement}

## 规则
- 使用 matplotlib 或 plotly 生成图表
- 自动选择最合适的图表类型（柱状图、折线图、饼图、散点图等）
- 包含标题、轴标签、图例
- 中文显示需设置字体

返回 JSON：
{{
  "chart_type": "图表类型",
  "reason": "选择原因",
  "code": "完整的 Python 代码"
}}
"""

def generate_chart(
    columns: list, rows: list, requirement: str = "生成合适的可视化图表"
) -> dict:
    sample = rows[:5]
    sample_text = "\n".join(str(row) for row in sample)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": CHART_PROMPT.format(
            columns=columns,
            sample_data=sample_text,
            total_rows=len(rows),
            requirement=requirement,
        )}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)
```

### 2.2 可视化库选择

不同场景适合不同的可视化库：

| 库 | 适用场景 | 特点 |
|---|---------|------|
| matplotlib | 静态报告、论文图表 | 最灵活，学习曲线陡 |
| plotly | 交互式 Web 图表 | 支持缩放、悬停、导出 |
| echarts (pyecharts) | 仪表盘、BI 集成 | 丰富的图表类型，中文友好 |

**Plotly 交互式图表示例**：

```python
import plotly.express as px
import pandas as pd

def create_interactive_chart(data: dict, chart_type: str) -> str:
    """生成交互式 Plotly 图表，返回 HTML"""
    df = pd.DataFrame(data["rows"], columns=data["columns"])

    chart_funcs = {
        "bar": px.bar,
        "line": px.line,
        "scatter": px.scatter,
        "pie": px.pie,
        "histogram": px.histogram,
    }

    func = chart_funcs.get(chart_type, px.bar)
    fig = func(df, x=data["columns"][0], y=data["columns"][1])
    return fig.to_html(include_plotlyjs="cdn")
```

### 2.3 交互式探索

支持用户对数据进行追问和钻取：

```python
class DataExplorer:
    """交互式数据探索 Agent"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = extract_schema(db_path)
        self.history: list[dict] = []  # 查询历史

    async def explore(self, question: str) -> dict:
        """处理用户的数据探索请求"""
        # 带上查询历史，支持追问
        context = ""
        if self.history:
            context = "之前的查询：\n" + "\n".join(
                f"Q: {h['question']}\nSQL: {h['sql']}"
                for h in self.history[-3:]
            )

        result = text_to_sql(
            question,
            schema=f"{self.schema}\n\n{context}",
        )

        if not result.get("sql"):
            return {"type": "error", "message": result["explanation"]}

        data = execute_with_limits(self.db_path, result["sql"])
        if "error" in data:
            return {"type": "error", "message": data["error"]}

        # 记录历史
        self.history.append({"question": question, "sql": result["sql"]})

        # 判断是否需要可视化
        if len(data["rows"]) > 1 and len(data["columns"]) >= 2:
            chart = generate_chart(
                data["columns"], data["rows"], requirement=question
            )
            return {
                "type": "chart",
                "data": data,
                "chart": chart,
                "interpretation": interpret_results(
                    question, result["sql"], data["columns"], data["rows"]
                ),
            }

        return {
            "type": "text",
            "data": data,
            "interpretation": interpret_results(
                question, result["sql"], data["columns"], data["rows"]
            ),
        }
```

---

## 3. CSV/Excel 分析

### 3.1 文件解析

用 pandas 加载文件并自动生成数据概览：

```python
import pandas as pd

def load_and_preview(file_path: str) -> dict:
    """加载数据文件并生成概览"""
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(file_path)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    preview = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "head": df.head().to_dict(),
        "describe": df.describe().to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
    }
    return {"df": df, "preview": preview}

def preview_to_prompt(preview: dict) -> str:
    """将数据概览转为 LLM 可理解的文本"""
    return (
        f"数据集: {preview['shape'][0]} 行 × {preview['shape'][1]} 列\n"
        f"列名和类型: {preview['dtypes']}\n"
        f"前5行: {preview['head']}\n"
        f"统计摘要: {preview['describe']}\n"
        f"空值统计: {preview['null_counts']}"
    )
```

### 3.2 代码生成执行

让 LLM 生成 pandas 分析代码并执行：

```python
ANALYSIS_PROMPT = """你是一个数据分析师。根据用户需求，生成 pandas 分析代码。

## 数据概览
{data_preview}

## 用户需求
{requirement}

## 规则
- 使用 pandas 进行数据分析
- 变量 `df` 已经加载好，直接使用
- 将最终结果赋值给变量 `result`
- 如果需要可视化，使用 matplotlib，保存到 `output.png`
- 只返回可执行的 Python 代码，不要包含 markdown 标记

代码：
"""

def generate_analysis_code(data_preview: str, requirement: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
            data_preview=data_preview, requirement=requirement
        )}],
        temperature=0.2,
    )
    # 提取代码块
    content = response.choices[0].message.content
    if "```python" in content:
        content = content.split("```python")[1].split("```")[0]
    return content.strip()
```

### 3.3 沙箱执行

生成的代码必须在隔离环境中执行，防止恶意操作：

```python
import subprocess
import tempfile
import os

class PythonSandbox:
    """安全的 Python 代码执行沙箱"""

    FORBIDDEN_MODULES = {"os", "subprocess", "shutil", "socket", "http", "requests"}

    def validate_code(self, code: str) -> list[str]:
        """检查代码安全性"""
        warnings = []
        for module in self.FORBIDDEN_MODULES:
            if f"import {module}" in code or f"from {module}" in code:
                warnings.append(f"禁止导入模块: {module}")
        if "open(" in code and "output.png" not in code:
            warnings.append("禁止文件操作（图表输出除外）")
        if "eval(" in code or "exec(" in code:
            warnings.append("禁止使用 eval/exec")
        return warnings

    def execute(self, code: str, df: pd.DataFrame, timeout: int = 30) -> dict:
        """在子进程中执行代码"""
        warnings = self.validate_code(code)
        if warnings:
            return {"error": f"安全检查未通过: {'; '.join(warnings)}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存数据
            data_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(data_path, index=False)

            # 构建执行脚本
            script = f"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("{data_path}")
{code}
if 'result' in dir():
    print("__RESULT__")
    print(result)
"""
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(script)

            try:
                proc = subprocess.run(
                    ["python", script_path],
                    capture_output=True, text=True,
                    timeout=timeout, cwd=tmpdir,
                )
                output = proc.stdout
                if "__RESULT__" in output:
                    result = output.split("__RESULT__\n", 1)[1]
                else:
                    result = output

                # 检查是否生成了图表
                chart_path = os.path.join(tmpdir, "output.png")
                chart_data = None
                if os.path.exists(chart_path):
                    with open(chart_path, "rb") as f:
                        import base64
                        chart_data = base64.b64encode(f.read()).decode()

                return {
                    "result": result,
                    "chart": chart_data,
                    "stderr": proc.stderr if proc.returncode != 0 else None,
                }
            except subprocess.TimeoutExpired:
                return {"error": f"执行超时（{timeout}秒）"}
```

> **生产建议：** 真实环境中应使用 Docker 容器或 AWS Lambda 等更强的隔离方案，而非简单的子进程。

---

## 4. BI 工具集成

### 4.1 与现有 BI 协作

将 AI 查询能力嵌入现有 BI 工具：

```python
class BIIntegration:
    """BI 工具集成层"""

    def __init__(self, db_url: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(db_url)
        self.schema = self._extract_schema()

    def natural_language_query(self, question: str) -> dict:
        """自然语言查询接口，供 BI 工具调用"""
        result = text_to_sql(question, self.schema)
        if not result.get("sql"):
            return {"error": result["explanation"]}

        import pandas as pd
        df = pd.read_sql(result["sql"], self.engine)

        return {
            "sql": result["sql"],
            "explanation": result["explanation"],
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "row_count": len(df),
        }

    def create_dashboard_query(self, description: str) -> dict:
        """为仪表盘生成查询配置"""
        prompt = f"""根据描述生成仪表盘查询配置。

Schema: {self.schema}
描述: {description}

返回 JSON：
{{
  "queries": [
    {{"name": "指标名", "sql": "SQL", "chart_type": "图表类型", "refresh_interval": 60}}
  ]
}}"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
```

### 4.2 自然语言仪表盘

对话式数据探索界面：

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI(title="数据分析 Agent")

@app.post("/api/query")
async def query_endpoint(question: str, db_name: str = "default"):
    """自然语言查询接口"""
    explorer = DataExplorer(db_path=f"./databases/{db_name}.db")
    result = await explorer.explore(question)
    return result

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile):
    """上传 CSV 文件进行分析"""
    content = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    preview = load_and_preview_from_df(df)
    return {
        "file_name": file.filename,
        "preview": preview,
        "suggested_questions": generate_suggested_questions(preview),
    }

def generate_suggested_questions(preview: dict) -> list[str]:
    """根据数据特征生成推荐问题"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"数据概览：{preview_to_prompt(preview)}\n\n"
            "根据这个数据集，生成 5 个有价值的分析问题。返回 JSON 数组。"
        )}],
        response_format={"type": "json_object"},
        temperature=0.5,
    )
    return json.loads(response.choices[0].message.content).get("questions", [])
```

---

## 5. 数据安全

### 5.1 权限控制

数据分析 Agent 必须遵守数据访问权限：

```python
from enum import Enum

class AccessLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"

class DataAccessControl:
    """数据访问控制"""

    def __init__(self):
        # 表级权限配置
        self.table_access = {
            "products": AccessLevel.PUBLIC,
            "orders": AccessLevel.INTERNAL,
            "users": AccessLevel.CONFIDENTIAL,
            "payments": AccessLevel.CONFIDENTIAL,
        }
        # 列级脱敏配置
        self.masked_columns = {
            "users.email": lambda v: v[:3] + "***@***",
            "users.phone": lambda v: v[:3] + "****" + v[-4:],
            "payments.card_number": lambda _: "****-****-****-****",
        }

    def filter_schema(self, full_schema: str, user_level: AccessLevel) -> str:
        """根据用户权限过滤可见的 Schema"""
        level_order = [AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.CONFIDENTIAL]
        max_idx = level_order.index(user_level)
        allowed_tables = [
            table for table, level in self.table_access.items()
            if level_order.index(level) <= max_idx
        ]
        # 只返回允许访问的表的 Schema
        return filter_schema_by_tables(full_schema, allowed_tables)

    def mask_results(self, columns: list, rows: list, table: str) -> list:
        """对查询结果进行脱敏"""
        masked_rows = []
        for row in rows:
            masked_row = list(row)
            for i, col in enumerate(columns):
                key = f"{table}.{col}"
                if key in self.masked_columns:
                    masked_row[i] = self.masked_columns[key](str(row[i]))
            masked_rows.append(masked_row)
        return masked_rows
```

### 5.2 审计日志

记录所有查询操作，便于安全审计：

```python
from datetime import datetime
from pydantic import BaseModel

class QueryLog(BaseModel):
    timestamp: datetime
    user_id: str
    question: str
    generated_sql: str
    executed: bool
    row_count: int | None = None
    error: str | None = None

class AuditLogger:
    """查询审计日志"""

    def __init__(self, log_db_path: str = "./audit.db"):
        self.conn = sqlite3.connect(log_db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, user_id TEXT, question TEXT,
                generated_sql TEXT, executed INTEGER,
                row_count INTEGER, error TEXT
            )
        """)

    def log(self, entry: QueryLog):
        self.conn.execute(
            "INSERT INTO query_logs VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp.isoformat(), entry.user_id,
                entry.question, entry.generated_sql,
                int(entry.executed), entry.row_count, entry.error,
            ),
        )
        self.conn.commit()

    def get_user_queries(self, user_id: str, limit: int = 50) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM query_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

---

## 练习

1. **实现 Text-to-SQL 查询助手**：用 SQLite 示例数据库（如 Chinook），实现自然语言查询，包含 SQL 验证和结果解读
2. **构建 CSV 分析 Agent**：实现文件上传 → 数据预览 → 自然语言分析 → 可视化的完整流程，使用沙箱执行
3. **添加权限控制和审计**：为查询助手添加用户权限分级和查询日志记录

## 延伸阅读

- [LangChain SQL Agent](https://python.langchain.com/docs/tutorials/sql_qa/) — 官方 Text-to-SQL 教程
- [Vanna.ai](https://vanna.ai/) — 开源 Text-to-SQL 框架，支持自动训练
- [Spider 2.0 Benchmark](https://spider2-sql.github.io/) — Text-to-SQL 评估基准
- [Plotly 官方文档](https://plotly.com/python/) — 交互式可视化库