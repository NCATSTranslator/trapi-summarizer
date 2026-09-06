# Vertex-backed summary server. 
# Configuration comes from configurations/<APP_ENVIRONMENT>.json. Mount secrets
# at /app/configurations/secrets at runtime.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENVIRONMENT=production

# Matches document_root in the committed configuration files.
WORKDIR /app

# Install dependencies before copying source, so source edits reuse this layer.
COPY requirements.txt dependencies/graphwerk-0.0.0-py3-none-any.whl ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir graphwerk-0.0.0-py3-none-any.whl \
 && rm graphwerk-0.0.0-py3-none-any.whl

# Context is scoped by .dockerignore.
COPY . ./

EXPOSE 8000

# Select an environment with -e APP_ENVIRONMENT=dev, or append arguments:
#   docker run ... translator-summarizer --config configurations/dev.json
# For a shell in the container: docker run --entrypoint sh -it ...
ENTRYPOINT ["python", "vertex_ui_server.py"]
