.PHONY: comfyui venv

# ComfyUI user dir (workflows, settings, db). Default: comfyui_workflows at repo root.
# Override: make comfyui COMFYUI_USER_DIR=/path/to/dir
COMFYUI_USER_DIR ?= $(CURDIR)/comfyui_workflows

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

comfyui:
	python external/ComfyUI/main.py --enable-manager --user-directory "$(COMFYUI_USER_DIR)"

comfyui-run:
	python scripts/run_comfyui_workflow.py $(WORKFLOW)

comfyui-run:
	@test -n "$(WORKFLOW)" || (echo "Usage: make comfyui-run WORKFLOW=path/to/workflow_api.json" && exit 1)
	python scripts/run_comfyui_workflow.py "$(WORKFLOW)"
$(WORKFLOW)" || (echo "Usage: make comfyui-run WORKFLOW=path/to/workflow_api.json" && exit 1)
	python scripts/run_comfyui_workflow.py "$(WORKFLOW)"
