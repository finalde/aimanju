.PHONY: comfyui venv

# ComfyUI user dir (workflows, settings, db). Default: comfyui_workflows at repo root.
# Override: make comfyui COMFYUI_USER_DIR=/path/to/dir
COMFYUI_USER_DIR ?= $(CURDIR)/comfyui_workflows
# ComfyUI listen port. Override when 8188 is in use: make comfyui PORT=8189
PORT ?= 8188

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

comfyui:
	python external/ComfyUI/main.py --enable-manager --user-directory "$(COMFYUI_USER_DIR)" --port $(PORT)

comfyui-run:
	@test -n "$(WORKFLOW)" || (echo "Usage: make comfyui-run WORKFLOW=path/to/workflow_api.json" && exit 1)
	python scripts/run_comfyui_workflow.py "$(WORKFLOW)"
