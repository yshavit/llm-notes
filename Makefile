VENV ?= venv
FEEDBACK_FORM_URL ?=
MYST_START_ARGS ?=
MYST_BUILD_ARGS ?=

# Platform-specific virtualenv layout
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
    EXE_SUFFIX := .exe
    PYTHON ?= python
else
    VENV_BIN := $(VENV)/bin
    EXE_SUFFIX :=
    PYTHON ?= python3
endif

VENV_PYTHON := $(VENV_BIN)/python$(EXE_SUFFIX)
VENV_PIP := $(VENV_BIN)/pip$(EXE_SUFFIX)
MYST := $(abspath $(VENV_BIN)/myst$(EXE_SUFFIX))

# Use Bash for recipes (particularly important on Windows)
ifeq ($(OS),Windows_NT)
    SHELL := bash
else
    SHELL := /bin/sh
endif

.PHONY: setup images build serve start clean

build: $(VENV_PYTHON) images
	cd book && \
		MYST="$(MYST)" ../strict-myst build --strict --html $(MYST_BUILD_ARGS) && \
		mkdir -p _build/html/build/_assets/llm-book && \
		cp static/* _build/html/build/_assets/llm-book/ && \
		cd _build/html && \
		find . -type f -exec sed -i 's|FEEDBACK_FORM_URL|$(FEEDBACK_FORM_URL)|g' {} +

setup: $(VENV_PYTHON)

images: 04-Training/xy-plot-light.svg 04-Training/xy-plot-dark.svg

04-Training/xy-plot-light.svg: $(VENV_PYTHON)
	$(VENV_PYTHON) scripts/generate-plot.py light

04-Training/xy-plot-dark.svg: $(VENV_PYTHON)
	$(VENV_PYTHON) scripts/generate-plot.py dark

start: $(VENV_PYTHON) images
	cd book && $(MYST) start $(MYST_START_ARGS)

serve: build
	$(VENV_PYTHON) -m http.server -d book/_build/html 8000

clean: $(VENV_PYTHON)
	cd book && $(MYST) clean && rm -f 04-Training/xy-plot-light.svg 04-Training/xy-plot-dark.svg

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install -r requirements.txt
