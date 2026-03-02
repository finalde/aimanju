# comfyui_controlnet_aux – Models to Download

Base path for all targets:  
`/home/dalu/workspace/aimanju/external/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts`

---

## OpenPose annotator (body, hand, face)

When you see `model_path is .../body_pose_model.pth` (and similar) in the terminal, these are the files to download manually:

| Name | Download URL | Save path |
|------|---------------|-----------|
| body_pose_model.pth | https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth | `ckpts/lllyasviel/Annotators/body_pose_model.pth` |
| hand_pose_model.pth | https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth | `ckpts/lllyasviel/Annotators/hand_pose_model.pth` |
| facenet.pth | https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth | `ckpts/lllyasviel/Annotators/facenet.pth` |

Full directory: `.../comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/` (create it if missing).

---

## ZoeDepth (Intel/zoedepth-nyu-kitti) – Hugging Face cache

When you see a request for `Intel/zoedepth-nyu-kitti` (e.g. `config.json`) and "Network is unreachable", the **ZoeDepth** depth estimator is being loaded via the **transformers** library. It uses the **Hugging Face cache**, not the `ckpts` folder.

- **Repo:** https://huggingface.co/Intel/zoedepth-nyu-kitti  
- **Cache path:** `~/.cache/huggingface/hub/models--Intel--zoedepth-nyu-kitti/`  
- The code was updated to use `local_files_only=True`, so it will **only** read from this cache (no network). If the cache is empty, the node will fail until you either:
  1. Download the repo files (e.g. `config.json`, `preprocessor_config.json`, `*.safetensors` or `pytorch_model.bin`) and place them in the cache under `snapshots/<revision>/`, with `refs/main` pointing to that revision, or  
  2. Run once with network access so the transformers library can populate the cache.

---

## All annotator models (full table)

| Name | Download URL | Target save path |
|------|--------------|------------------|
| **hed** | https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth | `.../ckpts/lllyasviel/Annotators/ControlNetHED.pth` |
| **leres (res101)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth | `.../ckpts/lllyasviel/Annotators/res101.pth` |
| **leres (latest_net_G)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/latest_net_G.pth | `.../ckpts/lllyasviel/Annotators/latest_net_G.pth` |
| **lineart (sk_model)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth | `.../ckpts/lllyasviel/Annotators/sk_model.pth` |
| **lineart (sk_model2)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth | `.../ckpts/lllyasviel/Annotators/sk_model2.pth` |
| **lineart_anime** | https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth | `.../ckpts/lllyasviel/Annotators/netG.pth` |
| **manga_line** | https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth | `.../ckpts/lllyasviel/Annotators/erika.pth` |
| **midas** | https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt | `.../ckpts/lllyasviel/Annotators/dpt_hybrid-midas-501f0c75.pt` |
| **mlsd** | https://huggingface.co/lllyasviel/Annotators/resolve/main/mlsd_large_512_fp32.pth | `.../ckpts/lllyasviel/Annotators/mlsd_large_512_fp32.pth` |
| **normalbae** | https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt | `.../ckpts/lllyasviel/Annotators/scannet.pt` |
| **oneformer** | https://huggingface.co/lllyasviel/Annotators/resolve/main/250_16_swin_l_oneformer_ade20k_160k.pth | `.../ckpts/lllyasviel/Annotators/250_16_swin_l_oneformer_ade20k_160k.pth` |
| **open_pose (body)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth | `.../ckpts/lllyasviel/Annotators/body_pose_model.pth` |
| **open_pose (hand)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth | `.../ckpts/lllyasviel/Annotators/hand_pose_model.pth` |
| **open_pose (face)** | https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth | `.../ckpts/lllyasviel/Annotators/facenet.pth` |
| **pidi** | https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth | `.../ckpts/lllyasviel/Annotators/table5_pidinet.pth` |
| **sam (MobileSAM)** | https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt | `.../ckpts/dhkim2810/MobileSAM/mobile_sam.pt` |
| **uniformer** | https://huggingface.co/lllyasviel/Annotators/resolve/main/upernet_global_small.pth | `.../ckpts/lllyasviel/Annotators/upernet_global_small.pth` |
| **zoe** | https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt | `.../ckpts/lllyasviel/Annotators/ZoeD_M12_N.pt` |
| **teed** | https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth | `.../ckpts/bdsqlsz/qinglong_controlnet-lllite/Annotators/7_model.pth` |
| **anime_face_segment (UNet)** | https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/UNet.pth | `.../ckpts/bdsqlsz/qinglong_controlnet-lllite/Annotators/UNet.pth` |
| **anime_face_segment (isnet)** | https://huggingface.co/anime-seg/isnetis/resolve/main/isnetis.ckpt | `.../ckpts/anime-seg/isnetis/isnetis.ckpt` |
| **densepose** | https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r50_fpn_dl.torchscript | `.../ckpts/LayerNorm/DensePose-TorchScript-with-hint-image/densepose_r50_fpn_dl.torchscript` |
| **densepose (hrnet)** | https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/hrnetv2_w64_imagenet_pretrained.pth | `.../ckpts/LayerNorm/DensePose-TorchScript-with-hint-image/hrnetv2_w64_imagenet_pretrained.pth` |
| **dwpose bbox (yolox)** | https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx | `.../ckpts/yzd-v/DWPose/yolox_l.onnx` |
| **dwpose pose** | https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx | `.../ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx` |
| **dwpose pose (TorchScript)** | https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt | `.../ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt` |
| **dwpose bbox (TorchScript)** | https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt | `.../ckpts/hr16/yolox-onnx/yolox_l.torchscript.pt` |
| **animal_pose bbox** | (same as dwpose bbox above) | (same as dwpose bbox) |
| **animal_pose pose** | https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/rtmpose-m_ap10k_256_bs5.torchscript.pt | `.../ckpts/hr16/DWPose-TorchScript-BatchSize5/rtmpose-m_ap10k_256_bs5.torchscript.pt` |
| **animal_pose pose (onnx)** | https://huggingface.co/hr16/UnJIT-DWPose/resolve/main/rtmpose-m_ap10k_256.onnx | `.../ckpts/hr16/UnJIT-DWPose/rtmpose-m_ap10k_256.onnx` |
| **mesh_graphormer (graphormer)** | https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/graphormer_hand_state_dict.bin | `.../ckpts/hr16/ControlNet-HandRefiner-pruned/graphormer_hand_state_dict.bin` |
| **mesh_graphormer (hrnet)** | https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/resolve/main/hrnetv2_w64_imagenet_pretrained.pth | `.../ckpts/hr16/ControlNet-HandRefiner-pruned/hrnetv2_w64_imagenet_pretrained.pth` |
| **depth_anything (vitl14)** | https://huggingface.co/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth | `.../ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitl14.pth` |
| **depth_anything (vitb14)** | https://huggingface.co/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth | `.../ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitb14.pth` |
| **depth_anything (vits14)** | https://huggingface.co/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth | `.../ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vits14.pth` |
| **diffusion_edge (indoor)** | https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_indoor.pt | `.../ckpts/hr16/Diffusion-Edge/diffusion_edge_indoor.pt` |
| **diffusion_edge (urban)** | https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_urban.pt | `.../ckpts/hr16/Diffusion-Edge/diffusion_edge_urban.pt` |
| **diffusion_edge (natural)** | https://huggingface.co/hr16/Diffusion-Edge/resolve/main/diffusion_edge_natrual.pt | `.../ckpts/hr16/Diffusion-Edge/diffusion_edge_natrual.pt` |
| **unimatch (scale2-regrefine6)** | https://huggingface.co/hr16/Unimatch/resolve/main/gmflow-scale2-regrefine6-mixdata.pth | `.../ckpts/hr16/Unimatch/gmflow-scale2-regrefine6-mixdata.pth` |
| **unimatch (scale2)** | https://huggingface.co/hr16/Unimatch/resolve/main/gmflow-scale2-mixdata.pth | `.../ckpts/hr16/Unimatch/gmflow-scale2-mixdata.pth` |
| **unimatch (scale1)** | https://huggingface.co/hr16/Unimatch/resolve/main/gmflow-scale1-mixdata.pth | `.../ckpts/hr16/Unimatch/gmflow-scale1-mixdata.pth` |
| **zoe_depth_anything (indoor)** | https://huggingface.co/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt | `.../ckpts/LiheYoung/Depth-Anything/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt` |
| **zoe_depth_anything (outdoor)** | https://huggingface.co/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt | `.../ckpts/LiheYoung/Depth-Anything/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt` |

---

## Notes

- **Base path:** Replace `.../ckpts` with  
  `/home/dalu/workspace/aimanju/external/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts`
- **Depth Anything (HF transformers):** The README also references **LiheYoung/depth-anything-large-hf** (and base/small). Those are used via Hugging Face cache, not the `ckpts` folder; see `docs/depth-anything-large-manual-download.md` if you need to install them manually.
- **Optional DWPose/YOLO variants:** You can use other bbox/pose files from hr16 (e.g. yolo-nas-fp16) as listed in the README; place them under the same repo subpaths under `ckpts`.
- Create the directory for each file before saving (e.g. `mkdir -p ".../ckpts/lllyasviel/Annotators"` then put the file there).
