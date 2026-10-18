# What is this?

A 0-to-60 book for myself on how LLMs work.

See it at <https://yshavit.github.io/llm-notes/>

[![TODOs][TODO-badge]][TODO-search]

[TODO-badge]: https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/yshavit/llm-notes/automation/badge-data/todo-count.json&query=$.todos&label=Remaining%20TODOs&color=orange

[TODO-search]: https://github.com/search?q=todo+repo%3Ayshavit%2Fllm-notes+%28path%3A*.md+path%3Abook%29+OR+path%3Asimpllm%29&type=code&ref=advsearch

## Editing on Windows

Requires Docker Desktop. Instructions assume Git Bash.

- Build the Docker image:

  ```bash
  docker build -f Dockerfile.llm-notes -t llm-notes .
  ```

- Start the live server:

  ```bash
  docker run --rm -it \
    --mount "type=bind,source=$(pwd),target=/workspace" \
    --mount "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock" \
    --mount "type=volume,source=llm-notes-drawio,target=/workspace/book/_plugin/drawio" \
    -p 3000:3000 \
    llm-notes
  ```

- Open <http://localhost:3000>.

or,

- to run a one-shot build:
  ```bash
  docker run --rm -it \
    --mount "type=bind,source=$(pwd),target=/workspace" \
    --mount "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock" \
    --mount "type=volume,source=llm-notes-drawio,target=/workspace/book/_plugin/drawio" \
    llm-notes \
    make VENV=/opt/venv build
  ```

Note:

- The Docker socket mount is required for Drawio rendering.
- Do not mount the Windows `venv`; the container uses `/opt/venv`.

## License

This book is licensed under CC BY 4.0.
See the LICENSE file for full terms.
