VENV := venv
MYST := $(abspath $(VENV)/bin/myst)

.PHONY: setup build serve start clean

build: $(VENV)/bin/pip
	cd book && MYST=$(MYST) ../strict-myst build --strict --html

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
