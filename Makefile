.PHONY: comfyui venv

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

comfyui:
	python external/ComfyUI/main.py --enable-manager
