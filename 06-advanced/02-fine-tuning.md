# 微调与模型定制

> 当 Prompt Engineering 和 RAG 不够时

## 学习目标

- 理解何时需要微调
- 掌握 LoRA/QLoRA 参数高效微调
- 建立数据准备与评估流程

---

## 1. 何时微调

微调（Fine-tuning）是在预训练模型的基础上，使用特定领域数据继续训练，使模型更好地适应目标任务。但微调并非万能药——在投入时间和资源之前，需要明确判断是否真的需要微调。

### 1.1 决策框架

选择技术方案时，应按成本从低到高依次尝试：

```
Prompt Engineering → RAG → 微调 → 从头训练
```

**决策树：**

| 问题 | 是 → 方案 | 否 → 继续 |
|------|-----------|-----------|
| 调整提示词能解决吗？ | Prompt Engineering | ↓ |
| 需要外部知识吗？ | RAG | ↓ |
| 需要特定风格/格式/行为？ | 微调 | ↓ |
| 现有模型架构不够？ | 从头训练 | 重新评估需求 |

实际项目中，这些技术经常组合使用。例如：微调模型 + RAG 检索，让模型既掌握领域风格，又能获取最新知识。

### 1.2 适用场景

微调在以下场景中效果显著：

| 场景 | 说明 | 示例 |
|------|------|------|
| **特定输出格式** | 模型始终按固定格式输出 | 医疗报告生成、法律文书格式 |
| **领域术语和风格** | 掌握行业特有的表达方式 | 金融分析报告、技术文档 |
| **行为一致性** | 稳定的角色扮演和语气 | 品牌客服、虚拟助教 |
| **复杂指令遵循** | 多约束条件下的精确执行 | 结构化数据提取、代码转换 |
| **延迟优化** | 用小模型替代大模型 | 将 GPT-4 级别的任务蒸馏到 7B 模型 |

### 1.3 不适用场景

以下场景不应使用微调：

- **知识注入**：微调不擅长记忆大量事实性知识，应使用 RAG。微调 100 条产品信息不如建一个向量知识库
- **通用能力提升**：微调通常会让模型在目标任务上变好，但可能在其他任务上变差（灾难性遗忘）
- **数据不足**：高质量训练数据少于 50 条时，Few-shot Prompting 通常效果更好
- **快速变化的知识**：产品价格、政策法规等频繁更新的信息，应使用 RAG 实时检索

---

## 2. 数据准备

数据质量是微调成功的最关键因素。"Garbage in, garbage out"——低质量数据训练出的模型不仅无用，还可能比基础模型更差。

### 2.1 数据收集

训练数据的来源通常有三类：

**真实数据** 是最有价值的数据来源，直接来自生产环境：

```python
# 从客服对话日志中提取高质量样本
def extract_training_samples(chat_logs: list[dict]) -> list[dict]:
    samples = []
    for log in chat_logs:
        # 筛选条件：用户满意度高、对话完整、无敏感信息
        if (log["satisfaction_score"] >= 4
            and log["resolved"]
            and not log["contains_pii"]):
            samples.append({
                "messages": [
                    {"role": "system", "content": "你是一个专业的客服助手..."},
                    {"role": "user", "content": log["user_query"]},
                    {"role": "assistant", "content": log["agent_response"]},
                ]
            })
    return samples
```

**合成数据** 使用强模型（如 GPT-4o）生成训练数据，适合冷启动阶段：

```python
from openai import OpenAI

client = OpenAI()

def generate_synthetic_data(task_description: str, n: int = 50) -> list[dict]:
    """用 GPT-4o 生成合成训练数据"""
    samples = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""为以下任务生成一个训练样本（输入+理想输出）：
任务：{task_description}
要求：输入要多样化，输出要高质量。
以 JSON 格式返回：{{"input": "...", "output": "..."}}"""
            }],
            temperature=0.9,  # 高温度增加多样性
        )
        samples.append(response.choices[0].message.content)
    return samples
```

**数据增强** 通过改写、翻译、同义替换等方式扩充数据集，但要注意不要引入噪声。

### 2.2 数据格式

不同的微调平台和目标要求不同的数据格式：

**对话格式（Chat Format）** — 最常用，适合对话类任务：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的医疗助手，回答要准确、简洁。"},
    {"role": "user", "content": "头痛应该挂什么科？"},
    {"role": "assistant", "content": "头痛建议挂神经内科。如果伴有外伤，可挂神经外科。"}
  ]
}
```

**指令格式（Instruction Format）** — 适合 Alpaca 风格的微调：

```json
{
  "instruction": "将以下英文翻译为中文，保持专业术语准确",
  "input": "The transformer architecture uses self-attention mechanisms.",
  "output": "Transformer 架构使用自注意力机制。"
}
```

**偏好对格式（Preference Pairs）** — 用于 DPO/RLHF 训练：

```json
{
  "prompt": "解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习规律...",
  "rejected": "机器学习就是让电脑自己学东西，很简单的。"
}
```

### 2.3 数据质量

数据质量控制是微调流程中最容易被忽视但最重要的环节：

```python
import hashlib
from collections import Counter

def clean_dataset(samples: list[dict]) -> list[dict]:
    """数据清洗流水线"""
    cleaned = []
    seen_hashes = set()

    for sample in samples:
        text = str(sample)

        # 1. 去重：基于内容哈希
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # 2. 长度过滤：过短或过长的样本
        messages = sample.get("messages", [])
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs:
            continue
        response_len = len(assistant_msgs[0]["content"])
        if response_len < 10 or response_len > 10000:
            continue

        # 3. 质量检查：确保回复不是空话
        response = assistant_msgs[0]["content"]
        low_quality_patterns = ["我不确定", "作为AI", "我无法"]
        if any(p in response for p in low_quality_patterns):
            continue

        cleaned.append(sample)

    print(f"清洗前: {len(samples)} 条, 清洗后: {len(cleaned)} 条")
    return cleaned
```

### 2.4 数据规模

不同任务类型对数据量的需求差异很大：

| 任务类型 | 最小数据量 | 推荐数据量 | 说明 |
|----------|-----------|-----------|------|
| 格式调整 | 50-100 | 200-500 | 如 JSON 输出格式 |
| 风格迁移 | 100-300 | 500-1000 | 如品牌语气 |
| 领域适配 | 500-1000 | 2000-5000 | 如医疗、法律 |
| 复杂推理 | 1000+ | 5000-10000 | 如数学解题 |

> 经验法则：先用 50-100 条高质量数据做一次快速实验，验证微调方向是否正确，再逐步扩大数据集。

---

## 3. 微调方法

### 3.1 全量微调

全量微调（Full Fine-tuning）更新模型的所有参数。效果最好，但资源需求极高，且容易过拟合和灾难性遗忘。

| 模型规模 | 显存需求（FP16） | 显存需求（FP32） | 推荐 GPU |
|----------|-----------------|-----------------|----------|
| 7B | ~14 GB | ~28 GB | 1× A100 80GB |
| 13B | ~26 GB | ~52 GB | 2× A100 80GB |
| 70B | ~140 GB | ~280 GB | 8× A100 80GB |

由于资源需求高昂，全量微调在实际项目中已较少使用，大多数场景被参数高效微调（PEFT）方法取代。

### 3.2 LoRA

**LoRA（Low-Rank Adaptation）** 是目前最主流的参数高效微调方法。核心思想：冻结原始模型权重，在每一层注入可训练的低秩矩阵。

**原理：** 对于原始权重矩阵 W（d×d），LoRA 不直接更新 W，而是学习两个小矩阵 A（d×r）和 B（r×d），其中 r << d。前向传播时：

```
输出 = W·x + α·B·A·x
```

这样只需训练 A 和 B 的参数，大幅减少可训练参数量。

| 对比项 | 全量微调 | LoRA (r=16) |
|--------|---------|-------------|
| 7B 模型可训练参数 | 70 亿 | ~1700 万（0.24%） |
| 显存需求 | ~14 GB | ~8 GB |
| 训练速度 | 基准 | 快 2-3× |
| 效果 | 最优 | 接近全量微调 |

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # 秩：越大表达能力越强，但参数越多
    lora_alpha=32,                 # 缩放因子，通常设为 2×r
    lora_dropout=0.05,             # Dropout 防过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标层
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 16,777,216 || all params: 8,030,261,248 || trainable%: 0.2090
```

### 3.3 QLoRA

**QLoRA** 在 LoRA 的基础上引入量化技术，将基础模型量化为 4-bit 精度，进一步降低显存需求。这使得在单张消费级 GPU（如 RTX 4090 24GB）上微调 70B 模型成为可能。

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时用 bf16
    bnb_4bit_use_double_quant=True,       # 双重量化，进一步压缩
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
```

**三种微调方法对比：**

| 方法 | 7B 显存 | 70B 显存 | 训练速度 | 效果 |
|------|---------|---------|----------|------|
| 全量微调 | ~14 GB | ~140 GB | 慢 | 最优 |
| LoRA | ~8 GB | ~80 GB | 快 | 接近全量 |
| QLoRA | ~5 GB | ~24 GB | 中等 | 略低于 LoRA |

### 3.4 关键超参数

微调效果对超参数非常敏感，以下是关键参数的调优建议：

| 超参数 | 推荐范围 | 说明 |
|--------|---------|------|
| 学习率 | 1e-5 ~ 2e-4 | QLoRA 通常用 2e-4，全量微调用 1e-5 |
| Epoch | 1-5 | 数据量少时 3-5 轮，数据量大时 1-2 轮 |
| Batch Size | 4-32 | 受显存限制，可用梯度累积等效增大 |
| LoRA Rank (r) | 8-64 | 简单任务 8-16，复杂任务 32-64 |
| LoRA Alpha | 2×r | 通常设为 Rank 的 2 倍 |
| Warmup Ratio | 0.03-0.1 | 学习率预热比例 |
| Weight Decay | 0.01-0.1 | 正则化，防止过拟合 |

> 调参建议：先用默认参数跑一次基线，再逐个调整。学习率是最敏感的参数，建议优先调整。

---

## 4. 训练平台

### 4.1 本地训练

本地训练适合快速实验和小规模微调。**Unsloth** 是目前最受欢迎的本地微调工具，提供 2× 加速和 60% 显存节省。

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 加载模型（Unsloth 自动优化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 添加 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# 准备数据集
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

def format_chat(example):
    """将数据格式化为对话模板"""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = dataset.map(format_chat)

# 训练配置
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    ),
)

trainer.train()
model.save_pretrained("./fine-tuned-model")
```

**Hugging Face Transformers + PEFT** 是更灵活的选择，适合需要自定义训练流程的场景。

### 4.2 云端训练

当本地 GPU 资源不足时，云端训练是更好的选择：

| 平台 | 特点 | 适用场景 | 参考价格 |
|------|------|----------|----------|
| AWS SageMaker | 企业级，与 AWS 生态集成 | 大规模生产训练 | 按实例计费 |
| Together AI | 简单易用，支持多种模型 | 快速实验 | ~$0.5/小时 (7B) |
| Modal | Serverless GPU，按秒计费 | 间歇性训练 | 按 GPU 秒计费 |
| Lambda Cloud | 高性价比 A100/H100 | 长时间训练 | ~$1.1/小时 (A100) |

使用 Modal 进行 Serverless 微调的示例：

```python
import modal

app = modal.App("fine-tuning")
image = modal.Image.debian_slim().pip_install(
    "unsloth", "transformers", "trl", "datasets"
)

@app.function(gpu="A100", image=image, timeout=3600)
def train(data_path: str):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    # ... 训练代码同本地训练
    return "训练完成"
```

### 4.3 API 微调

对于不想管理 GPU 基础设施的团队，API 微调是最简单的选择。OpenAI 提供了开箱即用的微调 API：

```python
from openai import OpenAI

client = OpenAI()

# 1. 上传训练数据
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune",
)

# 2. 创建微调任务
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 1.8,
        "batch_size": 4,
    },
)

# 3. 监控训练进度
import time
while True:
    status = client.fine_tuning.jobs.retrieve(job.id)
    print(f"状态: {status.status}")
    if status.status in ["succeeded", "failed"]:
        break
    time.sleep(60)

# 4. 使用微调后的模型
if status.status == "succeeded":
    response = client.chat.completions.create(
        model=status.fine_tuned_model,  # ft:gpt-4o-mini-2024-07-18:org::xxx
        messages=[{"role": "user", "content": "你好"}],
    )
```

**三种训练方式对比：**

| 方式 | 灵活性 | 成本 | 上手难度 | 适用场景 |
|------|--------|------|----------|----------|
| 本地训练 | 最高 | 硬件一次性投入 | 中 | 频繁实验、数据敏感 |
| 云端训练 | 高 | 按需付费 | 中 | 大模型、偶尔训练 |
| API 微调 | 低 | 按 token 计费 | 低 | 快速验证、小团队 |

---

## 5. 评估与迭代

微调不是一次性工作，而是一个持续迭代的过程。科学的评估体系是迭代优化的基础。

### 5.1 评估指标

评估微调模型需要同时关注目标任务表现和通用能力保持：

**任务特定指标：**

| 任务类型 | 指标 | 说明 |
|----------|------|------|
| 分类 | Accuracy, F1, Precision, Recall | 标准分类指标 |
| 生成 | BLEU, ROUGE, BERTScore | 与参考答案的相似度 |
| 对话 | 用户满意度, 解决率 | 业务指标 |
| 格式遵循 | 格式正确率 | JSON 解析成功率等 |
| 综合 | LLM-as-Judge | 用 GPT-4o 评分 |

**通用能力保持** — 微调后需要检查模型在通用任务上是否退化：

```python
from openai import OpenAI

client = OpenAI()

def evaluate_model(model_id: str, test_cases: list[dict]) -> dict:
    """评估微调模型"""
    results = {"correct": 0, "total": len(test_cases), "scores": []}

    for case in test_cases:
        response = client.chat.completions.create(
            model=model_id,
            messages=case["messages"],
            temperature=0,
        )
        output = response.choices[0].message.content

        # 自动评估（精确匹配或 LLM-as-Judge）
        if case.get("expected"):
            is_correct = case["expected"].lower() in output.lower()
            results["correct"] += int(is_correct)

        # LLM-as-Judge 评分
        judge_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""评估以下回答的质量（1-5分）：
问题：{case['messages'][-1]['content']}
回答：{output}
评分标准：准确性、完整性、格式规范性
只返回数字分数。"""
            }],
        )
        score = float(judge_response.choices[0].message.content.strip())
        results["scores"].append(score)

    results["accuracy"] = results["correct"] / results["total"]
    results["avg_score"] = sum(results["scores"]) / len(results["scores"])
    return results
```

### 5.2 基准对比

科学的评估需要对比多个基线：

```python
def benchmark_comparison(test_cases: list[dict]) -> dict:
    """对比不同方案的效果"""
    models = {
        "基础模型": "gpt-4o-mini",
        "基础模型+提示词优化": "gpt-4o-mini",  # 使用优化后的 system prompt
        "微调模型": "ft:gpt-4o-mini:org::xxx",
        "RAG 方案": "gpt-4o-mini",  # 配合 RAG 检索
    }

    results = {}
    for name, model_id in models.items():
        results[name] = evaluate_model(model_id, test_cases)
        print(f"{name}: 准确率={results[name]['accuracy']:.2%}, "
              f"平均分={results[name]['avg_score']:.2f}")

    return results
```

对比结果示例：

| 方案 | 准确率 | 平均分 | 延迟 | 成本/千次 |
|------|--------|--------|------|-----------|
| GPT-4o-mini 基础 | 72% | 3.2 | 800ms | $0.15 |
| + 提示词优化 | 81% | 3.8 | 900ms | $0.18 |
| + RAG | 85% | 4.0 | 1200ms | $0.25 |
| 微调模型 | 91% | 4.3 | 600ms | $0.12 |

### 5.3 迭代策略

微调效果不理想时的排查和优化路径：

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 训练 loss 不下降 | 学习率太低/太高 | 调整学习率，尝试 1e-4 ~ 3e-4 |
| 过拟合（验证 loss 上升） | 数据量不足或 epoch 过多 | 增加数据、减少 epoch、增大 dropout |
| 格式不稳定 | 训练数据格式不一致 | 统一数据格式，增加格式示例 |
| 通用能力下降 | 灾难性遗忘 | 降低学习率、减少 epoch、混入通用数据 |
| 效果不如提示词 | 数据质量差 | 重新审查和清洗数据 |

---

## 6. 部署

微调完成后，需要将模型部署为可用的推理服务。

### 6.1 模型合并

LoRA 微调产生的是适配器权重（通常只有几十 MB），部署时需要与基础模型合并：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# 加载并合并 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./fine-tuned-lora")
merged_model = model.merge_and_unload()

# 保存合并后的完整模型
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

### 6.2 量化部署

合并后的模型通常较大，量化可以显著减小模型体积和推理显存需求：

| 量化格式 | 精度 | 7B 模型大小 | 推理显存 | 适用场景 |
|----------|------|------------|----------|----------|
| FP16 | 16-bit | ~14 GB | ~16 GB | GPU 推理基准 |
| GPTQ | 4-bit | ~4 GB | ~6 GB | GPU 推理，速度快 |
| AWQ | 4-bit | ~4 GB | ~6 GB | GPU 推理，精度略优 |
| GGUF | 2-8 bit | 3-7 GB | ~5 GB | CPU/混合推理，llama.cpp |

使用 llama.cpp 转换为 GGUF 格式（适合本地部署）：

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 转换为 GGUF 格式
python convert_hf_to_gguf.py ./merged-model --outfile model.gguf --outtype q4_k_m
```

### 6.3 推理服务

生产环境推荐使用专业的推理框架部署微调模型：

**vLLM** — 高性能推理引擎，支持 PagedAttention 和连续批处理：

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="./merged-model",
    tensor_parallel_size=1,    # GPU 并行数
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# 批量推理
prompts = ["你好，请介绍一下自己", "什么是机器学习？"]
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(output.outputs[0].text)
```

**使用 vLLM 启动 OpenAI 兼容的 API 服务：**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./merged-model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

启动后可以用标准的 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="./merged-model",
    messages=[{"role": "user", "content": "你好"}],
)
```

---

## 练习

1. 用 QLoRA 微调一个 7B 模型完成特定任务
2. 对比微调前后的任务表现
3. 将微调模型部署为 API 服务

## 延伸阅读

- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [Unsloth](https://github.com/unslothai/unsloth)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
