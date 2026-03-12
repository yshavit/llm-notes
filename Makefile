VENV := venv
MYST := $(abspath $(VENV)/bin/myst)
FEEDBACK_FORM_URL ?=

.PHONY: setup build serve start clean

build: $(VENV)/bin/pip
	cd book && \
		MYST=$(MYST) ../strict-myst build --strict --html && \
		mkdir -p _build/html/build/_assets/llm-book && \
		cp static/* _build/html/build/_assets/llm-book/ && \
		cd _build/html && \
		find . -type f -exec sed -i 's|FEEDBACK_FORM_URL|$(FEEDBACK_FORM_URL)|g' {} +

setup: $(VENV)/bin/pip

start: $(VENV)/bin/pip
	cd book && $(MYST) start

serve: build
	$(VENV)/bin/python3 -m http.server -d book/_build/html 8000

clean: $(VENV)/bin/pip
	cd book && $(MYST) clean

$(VENV)/bin/pip:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
