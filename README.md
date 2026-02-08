# aimanju

本仓库用于把 **剧本 → 分镜镜头（shotlist+prompt）→ 分镜图** 跑通一个最小可用的本地 workflow（开源/可本地部署）。

## 推荐的开源组合（2026 实战）

- **剧本 → 镜头拆解（文字）**：`Qwen2.5 Instruct`（Apache-2.0，结构化输出稳定，适合产出 JSON/CSV 的 shotlist）
  - 最省事的本地运行方式：`Ollama`（提供 OpenAI 兼容接口，脚本可直接调用）
- **镜头 → 分镜图（图片）**：`SDXL`（本地可跑、质量稳定，16GB 显存可用）
  - 更强但通常 **非商用**：`FLUX.1-dev`（需要更高显存，且许可更严格）

> 说明：目前并不存在“单一开源模型”能直接从完整剧本一键产出高质量分镜图且可控；行业常用做法是 **LLM 产出镜头表 + diffusion 产出镜头图** 的组合式 pipeline。

---

## 本地快速开始（最短路径）

### 0) 环境准备

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt
```

### 1) 从剧本生成 shotlist（不依赖 LLM，也能跑）

```bash
.venv/bin/python -m workflows.storyboard_from_script \
  --script scripts/episode_01_thanks_network.md \
  --episode episode_01_thanks_network
```

输出目录：`storyboards/<episode>/`（会生成 `shotlist.json` / `shotlist.csv` / `prompts.md`）

### 2) 下载 SDXL（可选，后续出图要用）

先装出图依赖：

```bash
.venv/bin/pip install -r requirements-sdxl.txt
```

```bash
.venv/bin/python -m workflows.download_models sdxl
```

### 3) 用 SDXL 批量生成分镜图（diffusers）

```bash
.venv/bin/python -m workflows.render_sdxl_diffusers \
  --shot-json storyboards/episode_01_thanks_network/shotlist.json \
  --out-dir storyboards/episode_01_thanks_network/frames
```

---

## 模块化入口（推荐）

本仓库采用：
- `libs/`：纯库代码（infrastructure/application/domain/common）
- `apps/`：可执行入口（每个 app 自带自己的 `config.yml`）

### 低层（infra）出图：只认 prompt → 图片

```bash
.venv/bin/python -m apps.image_gen
```

配置文件：`apps/image_gen/config.yml`

### 上层（wrapper）分镜出图：读 shotlist → 调用 image_gen 批量出图

```bash
.venv/bin/python -m apps.story_gen
```

配置文件：`apps/story_gen/config.yml`

---

## 角色/场景参考图（assets_gen，用于一致性）

你现在已经把 **ControlNet（openpose/depth/canny）+ IP-Adapter（SDXL + image_encoder）** 放进了：
- `models/controlnet_sdxl/{openpose,depth,canny}/`
- `models/ip_adapter_sdxl/sdxl_models/...`

接下来要补的“缺口”主要是两件事：

1) **准备 Control 图片**（你自己提供 PNG）：
- OpenPose：`assets/poses/*.png`（姿势骨架图）
- Depth：`assets/controls/*_depth*.png`
- Canny：`assets/controls/*_canny*.png`

2) **确保 SDXL base 可用**（ControlNet pipeline 推荐用 diffusers 格式）：
- 推荐：`stabilityai/stable-diffusion-xl-base-1.0`（首次运行会自动下载到 HF cache，或你也可以提前手动下载到本地 diffusers 目录）

### 运行 assets_gen

编辑配置：`apps/assets_gen/config.yml`（把 `characters:` / `backgrounds:` 填起来，并指向你的 control PNG）

运行：

```bash
.venv/bin/python -m apps.assets_gen -c apps/assets_gen/config.yml
```

输出目录（默认）：
- `outputs/assets/episode_01_thanks_network/characters/...`
- `outputs/assets/episode_01_thanks_network/backgrounds/...`

生成完后，你可以把其中某一张角色定妆图填到：
- `apps/assets_gen/config.yml` → `ip_adapter.reference_image`

这样后续出分镜时（story_gen）就能更稳定地保持同一个人。

## 使用 Ollama（可选：让 LLM 自动补全镜头语言 + prompt）

1) 安装 Ollama 并下载模型（示例：Qwen2.5 7B Instruct）：

```bash
ollama pull qwen2.5:7b-instruct
```

2) 生成 shotlist 时加上参数（会调用 `http://localhost:11434`）：

```bash
.venv/bin/python -m workflows.storyboard_from_script \
  --script scripts/episode_01_thanks_network.md \
  --episode episode_01_thanks_network \
  --llm ollama \
  --ollama-model qwen2.5:7b-instruct
```

