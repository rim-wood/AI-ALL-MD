# 安全与合规

> 保护 AI 应用免受攻击，满足合规要求

## 学习目标

- 理解 AI 应用面临的安全威胁
- 掌握 Guardrails 的实现方法
- 了解数据隐私与合规要求

---

## 1. Prompt Injection

### 1.1 攻击类型

<!-- 直接注入、间接注入、越狱 -->

### 1.2 防护策略

<!-- 输入过滤、指令隔离、权限最小化 -->

### 1.3 检测方法

<!-- 规则匹配、分类模型、LLM 检测 -->

## 2. Guardrails

### 2.1 输入防护

<!-- 内容过滤、话题限制、长度限制 -->

### 2.2 输出防护

<!-- 敏感信息检测、格式验证、事实核查 -->

### 2.3 工具与框架

<!-- NeMo Guardrails、Guardrails AI、Lakera -->

### 2.4 自定义 Guardrails

<!-- 基于规则、基于 LLM 的防护 -->

## 3. 数据隐私

### 3.1 PII 检测与脱敏

<!-- 个人信息识别、自动脱敏 -->

### 3.2 数据传输安全

<!-- 加密、API 密钥管理 -->

### 3.3 本地部署 vs 云端 API

<!-- 隐私敏感场景的部署选择 -->

## 4. 合规要求

### 4.1 GDPR

<!-- 数据主体权利、数据处理协议 -->

### 4.2 SOC 2

<!-- 安全控制、审计要求 -->

### 4.3 行业特定要求

<!-- 金融、医疗、教育 -->

## 5. 审计与治理

### 5.1 审计日志

<!-- 操作记录、决策追踪 -->

### 5.2 模型治理

<!-- 模型清单、风险评估 -->

### 5.3 内容安全策略

<!-- 内容分级、过滤规则 -->

---

## 练习

1. 实现一个 Prompt Injection 检测管道
2. 用 NeMo Guardrails 为聊天机器人添加防护
3. 实现 PII 检测与自动脱敏

## 延伸阅读

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Guardrails AI](https://www.guardrailsai.com/)
