# Agent image for the upstream side of the Harbor parity runs.
# Same base as the Harbor task image (python:3.12-slim, WORKDIR /app), plus Claude Code.
FROM python:3.12-slim
ARG CLAUDE_CODE_VERSION=latest
RUN apt-get update && apt-get install -y --no-install-recommends curl bash ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s "${CLAUDE_CODE_VERSION}"
ENV PATH="/root/.local/bin:${PATH}"
RUN claude --version
WORKDIR /app
