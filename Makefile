SHELL := /bin/bash

.PHONY: start build clean

start:
	cd book && source ../venv/bin/activate && myst start

build:
	cd book && source ../venv/bin/activate && myst build --strict --html

clean:
	cd book && source ../venv/bin/activate && myst clean
