# LLMOps

> AI 应用的持续集成、评估与发布

## 学习目标

- 理解 LLMOps 与传统 MLOps 的区别
- 建立提示词 CI/CD 流程
- 掌握评估驱动的迭代方法

---

## 1. LLMOps 概述

### 1.1 什么是 LLMOps

LLMOps（Large Language Model Operations）是一套专门针对大语言模型应用的工程实践，覆盖从开发、评估、部署到监控的完整生命周期。

与传统软件不同，LLM 应用的核心逻辑不在代码里，而在**提示词**和**模型配置**中。这意味着：

- **输出不确定**：相同输入可能产生不同输出，传统单元测试失效
- **质量难量化**：回答的"好坏"需要多维度评估，不是简单的 pass/fail
- **迭代对象变了**：优化的不是代码逻辑，而是提示词、模型选择、检索策略

因此，LLM 应用需要一套专门的运维体系来保障生产质量。

### 1.2 与 MLOps 的区别

| 维度 | MLOps | LLMOps |
|------|-------|--------|
| 核心产物 | 训练好的模型 | 提示词 + 模型配置 |
| 数据需求 | 大量标注数据 | 少量评估数据集 |
| 训练过程 | 必须（天/周级别） | 通常不需要（直接调用 API） |
| 评估方式 | 精确指标（AUC、F1） | 模糊评估（相关性、有用性） |
| 迭代速度 | 慢（重新训练） | 快（修改提示词即生效） |
| 部署复杂度 | 高（模型服务化） | 低（API 调用） |
| 版本管理 | 模型权重 + 代码 | 提示词 + 配置 + 代码 |
| 监控重点 | 数据漂移、模型性能 | 输出质量、幻觉率、成本 |

### 1.3 核心流程

LLMOps 的核心是一个持续迭代的闭环：

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  开发    │───>│  评估    │───>│  部署    │───>│  监控    │───>│  迭代    │
│         │    │         │    │         │    │         │    │         │
│ 提示词   │    │ 数据集   │    │ 灰度发布 │    │ 质量追踪 │    │ 分析改进 │
│ 工具定义 │    │ 自动评估 │    │ A/B 测试 │    │ 成本监控 │    │ 回归测试 │
│ RAG 配置 │    │ 人工审核 │    │ 回滚机制 │    │ 告警响应 │    │ 版本更新 │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └────┬────┘
      ^                                                          │
      └──────────────────────────────────────────────────────────┘
```

每个环节的关键动作：

1. **开发**：编写/修改提示词、定义工具、配置 RAG 管道
2. **评估**：在标准数据集上运行自动评估，必要时人工审核
3. **部署**：灰度发布到生产环境，配置 A/B 测试
4. **监控**：追踪输出质量、延迟、成本等核心指标
5. **迭代**：基于监控数据分析问题，回到开发阶段改进

---

## 2. 提示词 CI/CD

### 2.1 版本管理

提示词是 LLM 应用的核心资产，必须像代码一样进行版本管理。推荐的目录结构：

```
prompts/
├── chat/
│   ├── system.yaml          # 系统提示词
│   ├── rag_answer.yaml      # RAG 回答模板
│   └── summarize.yaml       # 摘要模板
├── evaluation/
│   ├── relevance.yaml       # 相关性评估
│   └── faithfulness.yaml    # 忠实度评估
└── config.yaml              # 模型和参数配置
```

使用 YAML 格式存储提示词，便于管理变量和元数据：

```yaml
# prompts/chat/system.yaml
name: customer_support_system
version: "2.3"
model: gpt-4o
temperature: 0.3
max_tokens: 1024

template: |
  你是一个专业的客服助手。请根据以下知识库内容回答用户问题。

  ## 规则
  - 只基于提供的知识库内容回答
  - 如果知识库中没有相关信息，明确告知用户
  - 回答要简洁专业，不超过 200 字

  ## 知识库内容
  {context}

  ## 用户问题
  {question}

metadata:
  author: team-ai
  updated: "2026-04-15"
  description: "客服系统提示词 v2.3 - 增加长度限制"
```

加载和使用提示词的代码：

```python
import yaml
from pathlib import Path
from openai import OpenAI

def load_prompt(prompt_path: str) -> dict:
    """从 YAML 文件加载提示词配置"""
    with open(prompt_path) as f:
        return yaml.safe_load(f)

def render_prompt(config: dict, **variables) -> str:
    """渲染提示词模板，填充变量"""
    return config["template"].format(**variables)

# 使用示例
client = OpenAI()
config = load_prompt("prompts/chat/system.yaml")
prompt = render_prompt(config, context="退货政策：7天无理由退货", question="怎么退货？")

response = client.chat.completions.create(
    model=config["model"],
    temperature=config["temperature"],
    max_tokens=config["max_tokens"],
    messages=[{"role": "system", "content": prompt}],
)
```

### 2.2 自动化测试

提示词的每次变更都应通过自动化测试验证。用 pytest 编写回归测试：

```python
# tests/test_prompts.py
import pytest
import yaml
from openai import OpenAI

client = OpenAI()

def load_prompt(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# 测试用例数据
TEST_CASES = [
    {
        "context": "退货政策：购买后7天内可无理由退货，需保持商品完好。",
        "question": "买了3天的东西能退吗？",
        "expected_keywords": ["可以", "7天", "退货"],
        "forbidden_keywords": ["不能", "无法"],
    },
    {
        "context": "营业时间：周一至周五 9:00-18:00",
        "question": "周末上班吗？",
        "expected_keywords": ["周一", "周五"],
    },
]

@pytest.mark.parametrize("case", TEST_CASES)
def test_prompt_quality(case):
    """测试提示词输出质量"""
    config = load_prompt("prompts/chat/system.yaml")
    prompt = config["template"].format(
        context=case["context"], question=case["question"]
    )

    response = client.chat.completions.create(
        model=config["model"],
        temperature=0,  # 测试时用 0 保证确定性
        messages=[{"role": "system", "content": prompt}],
    )
    answer = response.choices[0].message.content

    # 检查必须包含的关键词
    for keyword in case.get("expected_keywords", []):
        assert keyword in answer, f"回答中缺少关键词: {keyword}"

    # 检查不应包含的关键词
    for keyword in case.get("forbidden_keywords", []):
        assert keyword not in answer, f"回答中包含禁止词: {keyword}"

    # 检查回答长度
    assert len(answer) < 500, "回答过长"


def test_prompt_refusal():
    """测试知识库外问题的拒答能力"""
    config = load_prompt("prompts/chat/system.yaml")
    prompt = config["template"].format(
        context="退货政策：7天无理由退货。",
        question="帮我写一首诗",
    )

    response = client.chat.completions.create(
        model=config["model"],
        temperature=0,
        messages=[{"role": "system", "content": prompt}],
    )
    answer = response.choices[0].message.content

    # 应该拒绝回答无关问题
    refusal_indicators = ["无法", "不能", "超出", "抱歉", "没有相关"]
    assert any(kw in answer for kw in refusal_indicators), "未能拒绝知识库外问题"
```

### 2.3 发布流程

使用 GitHub Actions 实现提示词的 CI/CD：

```yaml
# .github/workflows/prompt-ci.yaml
name: Prompt CI/CD

on:
  push:
    paths: ['prompts/**']
  pull_request:
    paths: ['prompts/**']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install pytest openai pyyaml

      - name: Run prompt tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/test_prompts.py -v

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy prompts (canary 10%)
        run: |
          python scripts/deploy_prompts.py \
            --env production \
            --canary-percent 10

      - name: Wait and validate
        run: |
          sleep 300  # 等待 5 分钟收集数据
          python scripts/validate_canary.py \
            --min-quality-score 0.85 \
            --max-error-rate 0.02

      - name: Full rollout
        run: |
          python scripts/deploy_prompts.py \
            --env production \
            --canary-percent 100
```

灰度发布的核心逻辑：

```python
import hashlib
import yaml

class PromptRouter:
    """基于用户 ID 的提示词灰度路由"""

    def __init__(self, canary_percent: int = 10):
        self.canary_percent = canary_percent
        self.stable = load_prompt("prompts/chat/system.yaml")
        self.canary = load_prompt("prompts/chat/system_canary.yaml")

    def get_prompt(self, user_id: str) -> dict:
        """根据用户 ID 决定使用稳定版还是灰度版"""
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if hash_val % 100 < self.canary_percent:
            return self.canary
        return self.stable

def load_prompt(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

---

## 3. 评估框架

评估是 LLMOps 的核心环节。没有可靠的评估，就无法判断提示词修改是改进还是退化。

### 3.1 LangSmith

LangSmith 是 LangChain 团队推出的 LLM 应用开发平台，提供追踪、评估、数据集管理等功能。

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 创建评估数据集
dataset = client.create_dataset("customer-support-qa")

# 添加测试用例
examples = [
    {
        "inputs": {"question": "怎么退货？", "context": "7天无理由退货，需保持商品完好。"},
        "outputs": {"answer": "购买后7天内可退货，商品需保持完好。"},
    },
    {
        "inputs": {"question": "运费谁出？", "context": "退货运费由买家承担，质量问题除外。"},
        "outputs": {"answer": "一般由买家承担运费，质量问题由卖家承担。"},
    },
]
for ex in examples:
    client.create_example(
        inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id
    )

# 定义被评估的目标函数
def my_rag_app(inputs: dict) -> dict:
    # 你的 RAG 应用逻辑
    response = call_llm(inputs["question"], inputs["context"])
    return {"answer": response}

# 定义评估器
def correctness(run, example) -> dict:
    """用 LLM 判断回答是否正确"""
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]

    grade_prompt = f"""判断预测答案是否与参考答案语义一致。
参考答案：{reference}
预测答案：{prediction}
只回答 correct 或 incorrect。"""

    result = call_llm(grade_prompt)
    return {"key": "correctness", "score": 1 if "correct" in result.lower() else 0}

# 运行评估
results = evaluate(my_rag_app, data=dataset.name, evaluators=[correctness])
print(f"正确率: {results.aggregate_metrics['correctness']:.2%}")
```

### 3.2 Promptfoo

Promptfoo 是一个轻量级的 CLI 评估工具，特别适合快速对比不同提示词和模型的效果。

配置文件示例：

```yaml
# promptfooconfig.yaml
prompts:
  - id: v1
    raw: |
      根据以下内容回答问题：
      内容：{{context}}
      问题：{{question}}

  - id: v2
    raw: |
      你是专业客服。请严格基于知识库回答，不要编造信息。
      知识库：{{context}}
      用户问题：{{question}}

providers:
  - id: openai:gpt-4o-mini
  - id: openai:gpt-4o

tests:
  - vars:
      context: "退货政策：7天无理由退货"
      question: "能退货吗？"
    assert:
      - type: contains
        value: "7天"
      - type: llm-rubric
        value: "回答准确引用了退货政策"

  - vars:
      context: "营业时间：周一至周五 9:00-18:00"
      question: "周六营业吗？"
    assert:
      - type: llm-rubric
        value: "明确告知周六不营业"

  - vars:
      context: "退货政策：7天无理由退货"
      question: "帮我写代码"
    assert:
      - type: llm-rubric
        value: "拒绝回答与知识库无关的问题"
```

运行评估：

```bash
# 安装
npm install -g promptfoo

# 运行评估，生成对比报告
promptfoo eval

# 打开 Web UI 查看结果
promptfoo view
```

Promptfoo 会生成一个矩阵视图，横轴是 prompt × model 的组合，纵轴是测试用例，每个格子显示通过/失败状态和详细输出。

### 3.3 Braintrust

Braintrust 是一个在线评估平台，支持实验追踪和团队协作。

```python
import braintrust

# 初始化实验
experiment = braintrust.init(project="customer-support", experiment="prompt-v2.3")

# 记录评估结果
for test_case in test_cases:
    output = my_rag_app(test_case["input"])

    experiment.log(
        input=test_case["input"],
        output=output,
        expected=test_case["expected"],
        scores={
            "correctness": score_correctness(output, test_case["expected"]),
            "relevance": score_relevance(output, test_case["input"]),
        },
    )

# 查看实验摘要
print(experiment.summarize())
```

### 3.4 自定义评估

当现有工具不满足需求时，可以构建自定义评估框架。LLM-as-Judge 是最常用的方法：

```python
from openai import OpenAI
from dataclasses import dataclass

client = OpenAI()

@dataclass
class EvalResult:
    score: float        # 0-1
    reasoning: str      # 评分理由

def llm_judge(question: str, answer: str, context: str, criteria: str) -> EvalResult:
    """通用 LLM-as-Judge 评估器"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""请评估以下回答的质量。

## 评估标准
{criteria}

## 原始问题
{question}

## 参考上下文
{context}

## 待评估回答
{answer}

请按以下 JSON 格式输出：
{{"score": 0.0到1.0的分数, "reasoning": "评分理由"}}"""
        }],
        response_format={"type": "json_object"},
    )

    import json
    result = json.loads(response.choices[0].message.content)
    return EvalResult(**result)


# 定义多维度评估标准
EVAL_CRITERIA = {
    "faithfulness": "回答是否忠实于提供的上下文，没有编造信息。1.0=完全忠实，0.0=完全编造。",
    "relevance": "回答是否与问题相关且有帮助。1.0=高度相关，0.0=完全无关。",
    "safety": "回答是否安全，不包含有害、歧视或不当内容。1.0=完全安全，0.0=有害。",
}

def run_evaluation(question: str, answer: str, context: str) -> dict[str, EvalResult]:
    """对一条回答运行多维度评估"""
    results = {}
    for name, criteria in EVAL_CRITERIA.items():
        results[name] = llm_judge(question, answer, context, criteria)
    return results

# 使用示例
results = run_evaluation(
    question="怎么退货？",
    answer="您可以在购买后7天内申请退货，需保持商品完好。",
    context="退货政策：购买后7天内可无理由退货，需保持商品完好。",
)
for name, result in results.items():
    print(f"{name}: {result.score:.2f} - {result.reasoning}")
```

---

## 4. A/B 测试

### 4.1 实验设计

A/B 测试用于在生产环境中对比不同提示词或模型配置的效果。

关键要素：

| 要素 | 说明 | 示例 |
|------|------|------|
| 主要指标 | 直接衡量目标的指标 | 用户满意度评分 |
| 护栏指标 | 不能恶化的底线指标 | 幻觉率、响应延迟 |
| 流量分配 | 实验组/对照组的比例 | 50/50 或 10/90 |
| 最小样本量 | 达到统计显著性所需的样本数 | 通常 1000+ 次交互 |
| 实验周期 | 收集足够数据的时间 | 通常 1-2 周 |

```python
import hashlib
from dataclasses import dataclass

@dataclass
class Variant:
    name: str
    prompt_version: str
    model: str
    weight: int  # 流量权重

class ABTest:
    def __init__(self, experiment_name: str, variants: list[Variant]):
        self.experiment_name = experiment_name
        self.variants = variants
        self.total_weight = sum(v.weight for v in variants)

    def assign(self, user_id: str) -> Variant:
        """确定性地将用户分配到实验组"""
        key = f"{self.experiment_name}:{user_id}"
        hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        bucket = hash_val % self.total_weight

        cumulative = 0
        for variant in self.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant
        return self.variants[-1]

# 配置实验
experiment = ABTest(
    experiment_name="prompt-v2.3-vs-v2.2",
    variants=[
        Variant(name="control", prompt_version="v2.2", model="gpt-4o", weight=50),
        Variant(name="treatment", prompt_version="v2.3", model="gpt-4o", weight=50),
    ],
)

# 在请求处理中使用
variant = experiment.assign(user_id="user_123")
prompt_config = load_prompt(f"prompts/chat/system_{variant.prompt_version}.yaml")
```

### 4.2 统计分析

实验结束后，需要进行统计分析来判断结果是否显著：

```python
import numpy as np
from scipy import stats

def analyze_ab_test(
    control_scores: list[float],
    treatment_scores: list[float],
    alpha: float = 0.05,
) -> dict:
    """分析 A/B 测试结果"""
    control = np.array(control_scores)
    treatment = np.array(treatment_scores)

    # 基本统计量
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    lift = (treatment_mean - control_mean) / control_mean

    # 双样本 t 检验
    t_stat, p_value = stats.ttest_ind(control, treatment)

    # 置信区间（差值）
    diff = treatment_mean - control_mean
    se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    return {
        "control_mean": round(control_mean, 4),
        "treatment_mean": round(treatment_mean, 4),
        "lift": f"{lift:.2%}",
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "confidence_interval": (round(ci_lower, 4), round(ci_upper, 4)),
        "recommendation": "采用 treatment" if p_value < alpha and lift > 0 else "保持 control",
    }

# 使用示例
result = analyze_ab_test(
    control_scores=[0.8, 0.7, 0.9, 0.85, 0.75],   # 对照组满意度
    treatment_scores=[0.9, 0.85, 0.95, 0.88, 0.92], # 实验组满意度
)
print(result)
```

### 4.3 决策框架

根据实验结果做出决策的流程：

```
实验结束
  │
  ├─ 主要指标显著提升 & 护栏指标无恶化 → ✅ 全量发布 treatment
  │
  ├─ 主要指标无显著差异 → 🔄 延长实验 或 放弃变更
  │
  ├─ 主要指标提升但护栏指标恶化 → ⚠️ 分析权衡，可能需要调整
  │
  └─ 主要指标显著下降 → ❌ 回滚到 control
```

关键原则：

- **护栏指标优先**：即使主要指标提升，如果幻觉率上升或延迟恶化，也不应发布
- **考虑实际意义**：统计显著不等于业务显著，0.1% 的提升可能不值得增加的复杂度
- **记录决策**：每次实验的结论和决策理由都应归档，便于后续参考

---

## 5. 模型管理

### 5.1 模型切换

生产环境需要支持灵活的模型切换，避免硬编码模型名称：

```python
from dataclasses import dataclass, field
from openai import OpenAI

@dataclass
class ModelConfig:
    name: str
    provider: str
    max_tokens: int = 4096
    temperature: float = 0.7
    fallback: str | None = None  # 降级模型

# 模型注册表
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "default": ModelConfig(
        name="gpt-4o", provider="openai", fallback="fast"
    ),
    "fast": ModelConfig(
        name="gpt-4o-mini", provider="openai", temperature=0.3
    ),
    "reasoning": ModelConfig(
        name="o3-mini", provider="openai", temperature=1.0
    ),
}

class ModelManager:
    def __init__(self):
        self.client = OpenAI()

    def call(self, model_key: str, messages: list[dict], **kwargs) -> str:
        """调用模型，失败时自动降级"""
        config = MODEL_REGISTRY[model_key]
        try:
            response = self.client.chat.completions.create(
                model=config.name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", config.max_tokens),
                temperature=kwargs.get("temperature", config.temperature),
            )
            return response.choices[0].message.content
        except Exception as e:
            if config.fallback:
                print(f"模型 {config.name} 调用失败: {e}，降级到 {config.fallback}")
                return self.call(config.fallback, messages, **kwargs)
            raise

# 使用
manager = ModelManager()
answer = manager.call("default", [{"role": "user", "content": "你好"}])
```

### 5.2 版本回滚

快速回滚机制是生产安全的关键保障：

```python
import json
from datetime import datetime
from pathlib import Path

class PromptVersionManager:
    """提示词版本管理，支持快速回滚"""

    def __init__(self, history_dir: str = "prompt_history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)

    def deploy(self, prompt_config: dict, version: str):
        """部署新版本，保存历史"""
        record = {
            "version": version,
            "config": prompt_config,
            "deployed_at": datetime.now().isoformat(),
        }
        # 保存到历史
        history_file = self.history_dir / f"{version}.json"
        history_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))

        # 更新当前版本指针
        current_file = self.history_dir / "current.json"
        current_file.write_text(json.dumps({"version": version}, indent=2))

    def rollback(self, target_version: str):
        """回滚到指定版本"""
        history_file = self.history_dir / f"{target_version}.json"
        if not history_file.exists():
            raise ValueError(f"版本 {target_version} 不存在")

        current_file = self.history_dir / "current.json"
        current_file.write_text(json.dumps({"version": target_version}, indent=2))
        print(f"已回滚到版本 {target_version}")

    def get_current(self) -> dict:
        """获取当前版本的配置"""
        current_file = self.history_dir / "current.json"
        current = json.loads(current_file.read_text())
        history_file = self.history_dir / f"{current['version']}.json"
        record = json.loads(history_file.read_text())
        return record["config"]
```

### 5.3 配置管理

将模型参数外部化，实现环境隔离：

```yaml
# config/production.yaml
llm:
  default_model: gpt-4o
  temperature: 0.3
  max_tokens: 2048

rag:
  chunk_size: 512
  top_k: 5
  similarity_threshold: 0.75

guardrails:
  max_input_length: 2000
  blocked_topics: ["政治", "暴力"]
```

```python
import yaml
import os

def load_config() -> dict:
    """根据环境加载配置"""
    env = os.getenv("APP_ENV", "development")
    config_path = f"config/{env}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 环境变量覆盖（优先级最高）
    if api_key := os.getenv("OPENAI_API_KEY"):
        config.setdefault("llm", {})["api_key"] = api_key

    return config
```

---

## 练习

1. 为一个 RAG 应用建立 Promptfoo 评估流水线
2. 实现提示词的 Git 版本管理与自动化测试
3. 设计一个 A/B 测试方案并分析结果

## 延伸阅读

- [LangSmith 文档](https://docs.smith.langchain.com/)
- [Promptfoo 文档](https://www.promptfoo.dev/)
- [Braintrust 文档](https://www.braintrust.dev/docs)
