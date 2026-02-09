# ComfyUI local server – API reference

Quick reference for the HTTP API of the ComfyUI server run with `make comfyui` (default: **http://127.0.0.1:8188**).  
ComfyUI does not ship Swagger/OpenAPI for this server; this doc is a hand-maintained summary.

Official docs: **https://docs.comfy.org**

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| GET | `/ws` | WebSocket (e.g. for execution progress) |
| GET | `/docs` | Embedded static docs (general ComfyUI docs) |
| GET | `/object_info` | All node definitions (for building API-format prompts) |
| GET | `/object_info/{node_class}` | Info for a single node class |
| GET | `/prompt` | Current queue state (running + pending) |
| **POST** | **`/prompt`** | **Queue a workflow (trigger execution)** |
| GET | `/history` | Full execution history |
| GET | `/history/{prompt_id}` | Outputs and metadata for one run |
| GET | `/queue` | Queue (running + pending) |
| POST | `/queue` | Modify queue (clear, delete items) |
| POST | `/interrupt` | Interrupt current execution |
| POST | `/free` | Free memory (e.g. unload models) |
| GET | `/view` | Get image; query params: `filename`, `subfolder`, `type` |
| GET | `/view_metadata/{folder_name}` | Metadata for a folder |
| POST | `/upload/image` | Upload image |
| POST | `/upload/mask` | Upload mask |
| GET | `/api/jobs` | Jobs list (optional query: `workflow_id`, `sort_by`, `sort_order`) |
| GET | `/api/jobs/{job_id}` | Single job details |
| GET | `/system_stats` | System stats |
| GET | `/features` | Feature flags |
| GET | `/embeddings` | Embeddings |
| GET | `/models` | Models |
| GET | `/models/{folder}` | Files in a model folder |
| GET | `/extensions` | Extensions |
| POST | `/history` | (History-related write; see server for body) |

---

## Queue a workflow (POST `/prompt`)

**Request**

- **Content-Type:** `application/json`
- **Body:** `{ "prompt": <api_format_workflow> }`

The `prompt` value must be the **API-format** workflow (from ComfyUI: **Workflow → Export (API)**). It is an object with string node IDs as keys; each value has `class_type` and `inputs` (with references like `["node_id", slot_index]`).

Optional body fields:

- `prompt_id` – UUID for this run (default: server-generated)
- `client_id` – For WebSocket association
- `extra_data` – e.g. API keys for API nodes
- `front` – if true, insert at front of queue
- `number` – queue position
- `partial_execution_targets` – for partial runs

**Response (200)**

- `prompt_id` – ID for this execution
- `number` – Queue number
- `node_errors` – Per-node validation errors (if any)

**Response (400)**

- `error` – Validation/execution error
- `node_errors` – Per-node errors

**Example (curl)**

```bash
curl -X POST http://127.0.0.1:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": '"$(cat path/to/workflow_api.json)"'}'
```

Or use the repo script:

```bash
python scripts/run_comfyui_workflow.py path/to/workflow_api.json
```

---

## Get an output image (GET `/view`)

Query parameters:

- `filename` – e.g. `ComfyUI_00001_.png`
- `subfolder` – subfolder under the output root (can be empty)
- `type` – usually `output`

Example: `http://127.0.0.1:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output`

---

## WebSocket (GET `/ws`)

Connect with `clientId` for correlation with a given run, e.g.:

`ws://127.0.0.1:8188/ws?clientId=<uuid>`

Send the same `client_id` in the POST `/prompt` body to receive execution messages (e.g. `executing`, progress) for that run.

---

*This file is maintained in this repo only; no changes are made under `external/`.*
