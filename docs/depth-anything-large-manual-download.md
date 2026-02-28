# Depth Anything Large (LiheYoung/depth-anything-large-hf) – Manual Download

ComfyUI (via **comfyui_controlnet_aux**) is trying to load this model but cannot reach Hugging Face. Download the files below in a browser, then place them in the Hugging Face cache so the node finds them.

---

## 1. What to download

| File | Direct download URL | Size (approx.) |
|------|--------------------|----------------|
| **config.json** | https://huggingface.co/LiheYoung/depth-anything-large-hf/resolve/main/config.json?download=true | ~1.4 KB |
| **preprocessor_config.json** | https://huggingface.co/LiheYoung/depth-anything-large-hf/resolve/main/preprocessor_config.json?download=true | ~0.4 KB |
| **model.safetensors** | https://huggingface.co/LiheYoung/depth-anything-large-hf/resolve/main/model.safetensors?download=true | ~1.34 GB |

**Repo (browse / download from UI):**  
https://huggingface.co/LiheYoung/depth-anything-large-hf/tree/main

---

## 2. Where to save (Hugging Face cache)

The **comfyui_controlnet_aux** Depth Anything node uses the `transformers` library, which looks for this model in the Hugging Face cache.

### Cache base directory

- **Linux / WSL:** `~/.cache/huggingface/hub/`
- **Full path example:** `/home/dalu/.cache/huggingface/hub/`

### Target folder structure

Create this path (use the **exact** snapshot hash so the cache format is correct):

```
~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6/
```

Put the three downloaded files **inside** that `snapshots/27ccb09.../` folder:

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`

Then create the `refs` file so the cache knows which snapshot to use:

- **File:** `~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/refs/main`
- **Content (single line):**  
  `27ccb0920352c0c37b3a96441873c8d37bd52fb6`

---

## 3. Step-by-step commands (Linux / WSL)

```bash
# 1. Create the snapshot directory
mkdir -p ~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6

# 2. Go there (then copy or move your downloaded files into this folder)
cd ~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6

# 3. After you have copied config.json, preprocessor_config.json, and model.safetensors here, create the refs file:
mkdir -p ~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/refs
echo -n 27ccb0920352c0c37b3a96441873c8d37bd52fb6 > ~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/refs/main
```

If you downloaded the files to e.g. `~/Downloads/depth-anything-large-hf/`:

```bash
cp ~/Downloads/depth-anything-large-hf/config.json \
   ~/Downloads/depth-anything-large-hf/preprocessor_config.json \
   ~/Downloads/depth-anything-large-hf/model.safetensors \
   ~/.cache/huggingface/hub/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6/
```

---

## 4. Verify

Restart ComfyUI and run a workflow that uses the **Depth Anything** preprocessor (e.g. from comfyui_controlnet_aux). It should load the model from cache and no longer try to download from Hugging Face.

---

## Optional: custom cache location

If you prefer a different cache directory, set one of these **before** starting ComfyUI:

```bash
export HF_HOME=/path/to/your/hf_cache
# or
export HUGGINGFACE_HUB_CACHE=/path/to/your/hf_cache
```

Then use the same structure under that path:  
`<cache>/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6/` and `refs/main` as above.
