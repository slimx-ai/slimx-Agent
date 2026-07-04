# The standalone SlimX-Agent service: the agent engine loop in its own container.
# No database, no provider credentials, no host code — it talks only to the host's
# internal agent-host callback API (SLIMX_AGENT_HOST_URL + SLIMX_AGENT_INTERNAL_TOKEN).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY slimx_agent ./slimx_agent
RUN pip install --no-cache-dir ".[service]"

RUN useradd --create-home --uid 10001 slimx
USER slimx

EXPOSE 8090

CMD ["uvicorn", "slimx_agent.service:app", "--host", "0.0.0.0", "--port", "8090"]
