VENV := venv
MYST := $(abspath $(VENV)/bin/myst)

.PHONY: start build clean setup

build: $(VENV)/bin/pip
	cd book && MYST=$(MYST) ../strict-myst build --strict --html

setup: $(VENV)/bin/pip

start: $(VENV)/bin/pip
	cd book && $(MYST) start

clean: $(VENV)/bin/pip
	cd book && $(MYST) clean

$(VENV)/bin/pip:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
