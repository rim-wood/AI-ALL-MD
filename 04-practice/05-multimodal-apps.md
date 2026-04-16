# 多模态应用

> 超越文本——视觉、语音与图像生成

## 学习目标

- 掌握视觉理解与图像分析的 API 调用与应用场景
- 了解语音交互（STT/TTS/实时对话）的实现方案
- 理解图像生成与编辑的技术栈
- 构建跨模态工作流的多模态 Agent

---

## 1. 视觉理解

### 1.1 图片描述与分析

GPT-4o、Claude、Gemini 等模型都支持图片输入。核心是将图片编码为 base64 或传入 URL。

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

def encode_image(image_path: str) -> str:
    """将图片编码为 base64"""
    return base64.b64encode(Path(image_path).read_bytes()).decode()

def analyze_image(image_path: str, question: str = "描述这张图片的内容") -> str:
    """图片分析"""
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",  # high / low / auto
                    },
                },
            ],
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content
```

**多图对比分析**：

```python
def compare_images(image_paths: list[str], question: str) -> str:
    """多图对比分析"""
    content = [{"type": "text", "text": question}]
    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return response.choices[0].message.content

# 使用示例
result = compare_images(
    ["design_v1.png", "design_v2.png"],
    "对比这两个 UI 设计稿，列出主要差异和各自的优缺点",
)
```

**实际应用场景**：

| 场景 | 提示词示例 | detail 参数 |
|------|-----------|------------|
| 商品识别 | "识别图中的商品，给出名称和预估价格" | high |
| 图表解读 | "解读这张数据图表的关键趋势" | high |
| UI 审查 | "检查这个界面的可访问性问题" | high |
| 缩略图分类 | "这张图片属于什么类别？" | low（省 Token） |

### 1.2 OCR 与文档解析

利用视觉模型提取图片中的文字和结构化信息：

```python
def extract_text_from_image(image_path: str) -> str:
    """OCR：从图片提取文字"""
    return analyze_image(
        image_path,
        "提取图片中的所有文字内容，保持原始排版格式。"
    )

def parse_table_from_image(image_path: str) -> str:
    """从图片中提取表格数据"""
    return analyze_image(
        image_path,
        "提取图片中的表格数据，以 Markdown 表格格式输出。"
    )

def parse_invoice(image_path: str) -> dict:
    """发票/收据解析"""
    import json
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "解析这张发票/收据，提取结构化信息。返回 JSON：\n"
                    '{"vendor": "", "date": "", "items": [{"name": "", '
                    '"quantity": 0, "price": 0}], "total": 0, "tax": 0}'
                )},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                },
            ],
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

### 1.3 视频分析

视频模型尚未普及，当前主流方案是**关键帧提取 + 逐帧分析**：

```python
import cv2
import tempfile
import os

def extract_key_frames(video_path: str, interval_sec: int = 5) -> list[str]:
    """从视频中按固定间隔提取关键帧"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    frames = []
    tmpdir = tempfile.mkdtemp()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            path = os.path.join(tmpdir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
        frame_idx += 1

    cap.release()
    return frames

def analyze_video(video_path: str, question: str) -> str:
    """视频分析：提取关键帧 → 批量分析 → 综合总结"""
    frames = extract_key_frames(video_path, interval_sec=10)

    # 构建多图消息
    content = [{"type": "text", "text": (
        f"以下是一段视频的关键帧截图（每 10 秒一帧，共 {len(frames)} 帧）。\n"
        f"请分析视频内容并回答：{question}"
    )}]
    for frame_path in frames[:20]:  # 限制帧数避免超出上下文
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame_path)}"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
    )
    return response.choices[0].message.content
```

> **成本提示：** 图片输入按 Token 计费。`detail: high` 模式下一张 1024×1024 图片约消耗 765 Token。批量处理图片时注意成本控制。

---

## 2. 语音交互

### 2.1 语音转文字（STT）

OpenAI Whisper 是目前最流行的 STT 方案，支持 API 调用和本地部署：

```python
def speech_to_text(audio_path: str, language: str = "zh") -> str:
    """语音转文字"""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="text",
        )
    return transcript

def speech_to_text_with_timestamps(audio_path: str) -> dict:
    """带时间戳的语音转文字（适合字幕生成）"""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return transcript
```

**本地部署 Whisper**（适合隐私敏感场景）：

```python
# pip install openai-whisper
import whisper

model = whisper.load_model("large-v3")  # tiny/base/small/medium/large

def local_stt(audio_path: str) -> str:
    result = model.transcribe(audio_path, language="zh")
    return result["text"]
```

### 2.2 文字转语音（TTS）

```python
def text_to_speech(
    text: str,
    output_path: str = "output.mp3",
    voice: str = "alloy",  # alloy/echo/fable/onyx/nova/shimmer
) -> str:
    """文字转语音"""
    response = client.audio.speech.create(
        model="tts-1",       # tts-1（快）或 tts-1-hd（高质量）
        voice=voice,
        input=text,
    )
    response.stream_to_file(output_path)
    return output_path

def stream_tts(text: str):
    """流式 TTS——边生成边播放"""
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    # 流式写入，可配合音频播放器实时播放
    for chunk in response.iter_bytes(chunk_size=4096):
        yield chunk
```

**TTS 方案对比**：

| 方案 | 延迟 | 质量 | 中文支持 | 成本 |
|------|------|------|---------|------|
| OpenAI TTS | 低 | 高 | 好 | $15/1M 字符 |
| ElevenLabs | 低 | 极高 | 一般 | $5-330/月 |
| Azure TTS | 低 | 高 | 极好 | $16/1M 字符 |
| Edge TTS（免费） | 低 | 中 | 好 | 免费 |
| Bark（本地） | 高 | 中 | 一般 | 免费（需 GPU） |

### 2.3 实时语音对话

OpenAI Realtime API 支持低延迟的语音到语音对话：

```python
import asyncio
import websockets
import json

async def realtime_voice_chat():
    """使用 OpenAI Realtime API 进行语音对话"""
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        # 配置会话
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "你是一个友好的中文语音助手。",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",  # 服务端语音活动检测
                    "threshold": 0.5,
                },
            },
        }))

        # 发送音频数据
        async def send_audio(audio_stream):
            async for chunk in audio_stream:
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))

        # 接收响应
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "response.audio.delta":
                audio_chunk = base64.b64decode(event["delta"])
                yield audio_chunk  # 播放音频
            elif event["type"] == "response.text.delta":
                print(event["delta"], end="", flush=True)
```

**WebRTC 集成**——在浏览器中实现实时语音对话：

```javascript
// 前端 WebRTC 音频采集
async function startVoiceChat() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: "audio/webm;codecs=opus",
  });

  const ws = new WebSocket("wss://your-server.com/ws/voice");

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      ws.send(event.data); // 发送音频到服务端
    }
  };

  ws.onmessage = (event) => {
    // 播放服务端返回的音频
    const audioBlob = new Blob([event.data], { type: "audio/mp3" });
    const audio = new Audio(URL.createObjectURL(audioBlob));
    audio.play();
  };

  mediaRecorder.start(250); // 每 250ms 发送一次
}
```

---

## 3. 图像生成

### 3.1 文生图

主流文生图 API 调用：

```python
def generate_image(
    prompt: str,
    size: str = "1024x1024",  # 1024x1024 / 1792x1024 / 1024x1792
    quality: str = "standard",  # standard / hd
    style: str = "natural",  # natural / vivid
) -> str:
    """DALL-E 文生图，返回图片 URL"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        n=1,
    )
    return response.data[0].url

# Stable Diffusion（通过 API 或本地部署）
import httpx

def generate_image_sd(
    prompt: str,
    negative_prompt: str = "blurry, low quality",
    steps: int = 30,
) -> bytes:
    """调用 Stable Diffusion WebUI API"""
    response = httpx.post(
        "http://localhost:7860/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "width": 1024,
            "height": 1024,
            "cfg_scale": 7,
        },
        timeout=120,
    )
    import base64
    return base64.b64decode(response.json()["images"][0])
```

### 3.2 图像编辑

**Inpainting**——局部修改图片：

```python
def edit_image(
    image_path: str,
    mask_path: str,
    prompt: str,
) -> str:
    """图像编辑（Inpainting）"""
    response = client.images.edit(
        model="dall-e-2",
        image=open(image_path, "rb"),
        mask=open(mask_path, "rb"),  # 白色区域为需要编辑的部分
        prompt=prompt,
        size="1024x1024",
    )
    return response.data[0].url
```

**图像变体**——基于原图生成风格变体：

```python
def create_variation(image_path: str) -> str:
    """生成图像变体"""
    response = client.images.create_variation(
        model="dall-e-2",
        image=open(image_path, "rb"),
        size="1024x1024",
        n=1,
    )
    return response.data[0].url
```

### 3.3 提示词工程

图像生成的提示词技巧与文本生成有很大不同：

```python
IMAGE_PROMPT_ENHANCE = """优化以下图像生成提示词，使其更适合 AI 图像生成。

原始描述：{description}

优化规则：
1. 添加具体的视觉细节（光线、角度、材质）
2. 指定艺术风格（摄影、插画、3D 渲染等）
3. 描述构图和色彩
4. 添加质量修饰词

返回 JSON：{{"enhanced_prompt": "优化后的英文提示词", "style_tags": ["标签"]}}
"""

def enhance_image_prompt(description: str) -> dict:
    """优化图像生成提示词"""
    import json
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": IMAGE_PROMPT_ENHANCE.format(
            description=description
        )}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# 示例
result = enhance_image_prompt("一只猫坐在窗台上")
# 输出: "A fluffy orange tabby cat sitting on a sunlit windowsill,
#         warm golden hour lighting, soft bokeh background of a garden,
#         photorealistic style, shot with 85mm lens, shallow depth of field"
```

**提示词模板库**：

```python
STYLE_TEMPLATES = {
    "product_photo": (
        "{subject}, professional product photography, "
        "white background, studio lighting, high resolution, 8k"
    ),
    "ui_mockup": (
        "{subject}, modern UI design, clean layout, "
        "Figma style, light theme, minimal"
    ),
    "illustration": (
        "{subject}, digital illustration, flat design, "
        "vibrant colors, vector art style"
    ),
    "architecture": (
        "{subject}, architectural visualization, "
        "photorealistic 3D render, golden hour, dramatic lighting"
    ),
}

def generate_styled_image(subject: str, style: str) -> str:
    template = STYLE_TEMPLATES.get(style, "{subject}")
    prompt = template.format(subject=subject)
    return generate_image(prompt)
```

---

## 4. 多模态 Agent

### 4.1 架构设计

多模态 Agent 需要处理不同类型的输入，并路由到合适的处理模块：

```python
from enum import Enum
from pydantic import BaseModel

class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class MultimodalInput(BaseModel):
    modality: ModalityType
    content: str  # 文本内容或文件路径
    metadata: dict = {}

class MultimodalAgent:
    """多模态 Agent"""

    def __init__(self):
        self.history: list[dict] = []

    async def process(self, inputs: list[MultimodalInput]) -> str:
        """处理多模态输入"""
        # 1. 预处理各模态输入
        message_content = []
        for inp in inputs:
            if inp.modality == ModalityType.TEXT:
                message_content.append({"type": "text", "text": inp.content})
            elif inp.modality == ModalityType.IMAGE:
                b64 = encode_image(inp.content)
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            elif inp.modality == ModalityType.AUDIO:
                text = speech_to_text(inp.content)
                message_content.append({"type": "text", "text": f"[语音转文字] {text}"})
            elif inp.modality == ModalityType.VIDEO:
                analysis = analyze_video(inp.content, "描述视频内容")
                message_content.append({"type": "text", "text": f"[视频分析] {analysis}"})

        # 2. 调用多模态模型
        messages = [
            {"role": "system", "content": (
                "你是一个多模态 AI 助手，能理解文字、图片、语音和视频。"
                "根据用户提供的多模态输入，给出有帮助的回答。"
            )},
            *self.history,
            {"role": "user", "content": message_content},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content

        # 3. 更新历史
        self.history.append({"role": "user", "content": message_content})
        self.history.append({"role": "assistant", "content": reply})

        return reply
```

### 4.2 跨模态工作流

将多个模态串联成完整的工作流：

```python
class ContentCreationPipeline:
    """跨模态内容创作流水线：图片 → 分析 → 文案 → 语音"""

    async def create_product_content(self, image_path: str) -> dict:
        """从产品图片自动生成营销内容"""
        # 1. 图片分析
        analysis = analyze_image(
            image_path,
            "详细描述这个产品的外观、材质、颜色和特点",
        )

        # 2. 生成营销文案
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": (
                f"产品描述：{analysis}\n\n"
                "请生成：\n"
                "1. 一句吸引人的标题（20字以内）\n"
                "2. 产品卖点（3个要点）\n"
                "3. 社交媒体文案（100字以内）"
            )}],
        )
        copywriting = response.choices[0].message.content

        # 3. 生成语音播报
        audio_path = text_to_speech(copywriting, "product_intro.mp3", voice="nova")

        # 4. 生成配图变体
        enhanced = enhance_image_prompt(analysis)
        banner_url = generate_image(
            f"Marketing banner: {enhanced['enhanced_prompt']}",
            size="1792x1024",
        )

        return {
            "analysis": analysis,
            "copywriting": copywriting,
            "audio": audio_path,
            "banner": banner_url,
        }
```

### 4.3 应用场景

**智能客服（图片+文字）**——用户发送商品图片咨询：

```python
@app.post("/api/multimodal-chat")
async def multimodal_chat(
    message: str = "",
    image: UploadFile | None = None,
    audio: UploadFile | None = None,
):
    """多模态客服接口"""
    agent = MultimodalAgent()
    inputs = []

    if message:
        inputs.append(MultimodalInput(modality=ModalityType.TEXT, content=message))
    if image:
        path = f"/tmp/{image.filename}"
        with open(path, "wb") as f:
            f.write(await image.read())
        inputs.append(MultimodalInput(modality=ModalityType.IMAGE, content=path))
    if audio:
        path = f"/tmp/{audio.filename}"
        with open(path, "wb") as f:
            f.write(await audio.read())
        inputs.append(MultimodalInput(modality=ModalityType.AUDIO, content=path))

    reply = await agent.process(inputs)
    return {"reply": reply}
```

**文档理解 Agent**——处理包含图表的 PDF：

```python
class DocumentUnderstandingAgent:
    """文档理解 Agent：处理文字 + 图表混合内容"""

    async def analyze_document(self, pdf_path: str, question: str) -> str:
        # 1. 提取文字
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() for page in reader.pages)

        # 2. 提取图片（图表）
        images = self._extract_images(pdf_path)

        # 3. 分析图表
        chart_analyses = []
        for img_path in images[:5]:
            analysis = analyze_image(img_path, "解读这张图表的数据和趋势")
            chart_analyses.append(analysis)

        # 4. 综合回答
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": (
                f"文档文字内容：\n{text[:5000]}\n\n"
                f"图表分析：\n{''.join(chart_analyses)}\n\n"
                f"问题：{question}"
            )}],
        )
        return response.choices[0].message.content
```

---

## 练习

1. **构建图片分析助手**：实现图片上传 → 自动描述 → 多轮问答的完整流程
2. **实现语音对话机器人**：用 Whisper STT + GPT-4o + TTS 构建语音对话，支持中文
3. **构建多模态内容创作 Agent**：输入产品图片，自动生成标题、文案、配图和语音介绍

## 延伸阅读

- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision) — 官方视觉能力指南
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 实时语音对话 API
- [OpenAI Image Generation](https://platform.openai.com/docs/guides/images) — DALL-E 图像生成指南
- [Whisper 论文](https://arxiv.org/abs/2212.04356) — Robust Speech Recognition via Large-Scale Weak Supervision
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) — 最流行的本地图像生成工具