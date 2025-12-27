VENV := venv
MYST := $(abspath $(VENV)/bin/myst)

.PHONY: start build clean setup

$(VENV)/bin/pip:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

setup: $(VENV)/bin/pip

start: $(VENV)/bin/pip
	cd book && $(MYST) start

build: $(VENV)/bin/pip
	cd book && MYST=$(MYST) ../strict-myst build --strict --html

clean: $(VENV)/bin/pip
	cd book && $(MYST) clean
