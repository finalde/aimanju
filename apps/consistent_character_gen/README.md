# consistent_character_gen (InstantID workflow)

This module calls a running ComfyUI server and executes the InstantID workflow from
`fofr/cog-consistent-character` to get stronger character consistency than plain img2img.

Source workflow and repo:
- https://github.com/fofr/cog-consistent-character
- workflow JSON: https://github.com/fofr/cog-consistent-character/blob/main/workflow_api.json
- custom nodes list: https://github.com/fofr/cog-consistent-character/blob/main/custom_nodes.json
- weights list: https://github.com/fofr/cog-consistent-character/blob/main/weights.json

## 1) Install ComfyUI + required custom nodes
From the repo list, install these custom nodes (at minimum):
- ComfyUI_InstantID
- ComfyUI_IPAdapter_plus
- comfyui_controlnet_aux
- ComfyUI-Impact-Pack
- ComfyUI_KJNodes
- ComfyUI_essentials
- was-node-suite-comfyui

The official list (with commit pins) is in `custom_nodes.json`.

## 2) Required models (download yourself)
The workflow needs (place into your ComfyUI `models/` folders):

Checkpoints:
- `dreamshaperXL_lightningDPMSDE.safetensors` -> `models/checkpoints/`

InstantID:
- `instantid-ip-adapter.bin` -> `models/instantid/`
- `instantid-controlnet.safetensors` -> `models/controlnet/`

OpenPose ControlNet:
- `thibaud_xl_openpose.safetensors` -> `models/controlnet/`

IP-Adapter (for PLUS FACE preset):
- `ip-adapter-plus-face_sdxl_vit-h.safetensors` -> `models/ipadapter/`
- `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` -> `models/clip_vision/`

InsightFace (for face analysis):
- `antelopev2` (folder) -> `models/insightface/`

All of the above names come from the workflow repo’s `weights.json`.

## 3) Prepare inputs
You need 3 images:
- `subject_image`: reference portrait/full-body image of the character
- `pose_image`: OpenPose/DW pose image (for body pose)
- `keypoints_image`: face keypoints image (for InstantID)

Set them in `apps/consistent_character_gen/config.yml`.

## 4) Run
Start ComfyUI (default port 8188), then run:

```bash
.venv/bin/python -m apps.consistent_character_gen -c apps/consistent_character_gen/config.yml
```

Outputs are saved under `output.out_dir`.
