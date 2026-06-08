# Railway deployment

Three services in one Railway project, all from this single repo:

| Service     | Source                              | Notes                                  |
|-------------|-------------------------------------|----------------------------------------|
| `qdrant`    | Docker image `qdrant/qdrant:latest` | Vector DB + a persistent Volume        |
| `rag-api`   | `docker/railway/Dockerfile.api`     | FastAPI, healthcheck `/health`         |
| `streamlit` | `docker/railway/Dockerfile.streamlit` | Dark-mode chat UI (public domain)    |

Local development is unchanged and lives separately — `docker-compose.yml`
(root) + `docker/local/Dockerfile.qdrant`. Railway never uses those files.

## 1. Qdrant service

Add a service → **Deploy from Docker Image** → `qdrant/qdrant:latest`.
- Attach a **Volume** mounted at `/qdrant/storage` (data persistence).
- It exposes port `6333` (REST) on the private network automatically.
- No public domain needed — only `rag-api` talks to it, over the private net.

## 2. rag-api service

Add a service → **Deploy from this repo**. In Settings:
- **Build → Dockerfile Path:** `docker/railway/Dockerfile.api`
  (or copy `railway.api.json` to the repo root as `railway.json`).
- **Networking:** generate a public domain if you want the API reachable
  from the browser; otherwise keep it private.

Variables:
```
QDRANT_HOST=http://${{qdrant.RAILWAY_PRIVATE_DOMAIN}}:6333
QDRANT_COLLECTION=book
OLLAMA_HOST=https://<your-ollama-endpoint>
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest
OLLAMA_GENERATION_MODEL=gemma4:31b-cloud
# optional observability
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<langsmith key>
LANGSMITH_PROJECT=rag-tut
```
`PORT` is injected by Railway — the container binds to it automatically.

## 3. streamlit service

Add another service → **Deploy from this repo**. In Settings:
- **Build → Dockerfile Path:** `docker/railway/Dockerfile.streamlit`.
- **Networking:** generate a public domain (this is the user-facing UI).

Variables:
```
RAG_API_URL=http://${{rag-api.RAILWAY_PRIVATE_DOMAIN}}:8000
```
(Use `rag-api`'s public URL instead if you prefer the browser/UI to hit the
public API directly.)

## Why this mirrors local

In both environments the app is config-driven via `QDRANT_HOST` / `RAG_API_URL`:
- **Local:** `docker-compose.yml` wires `qdrant`, `rag-api`, `streamlit` together
  on the compose network (`QDRANT_HOST=http://qdrant:6333`).
- **Railway:** the same services connect over Railway's private network
  (`QDRANT_HOST=http://${{qdrant.RAILWAY_PRIVATE_DOMAIN}}:6333`).

No code changes between environments — only environment variables differ.
