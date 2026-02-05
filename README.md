# PDF Chat (Next.js + Python + LangChain)

This is a full-stack starter for chatting with PDFs using:
- **Next.js** for the UI
- **FastAPI** for the Python backend
- **LangChain + OpenAI** for embeddings and LLM responses
- **FAISS** for vector search
- **3-agent pipeline**: analyst → answerer → checker
- **Responses API streaming** for live reasoning + answer updates

## Project structure
- `frontend/` — Next.js app
- `backend/` — FastAPI app

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY in .env
uvicorn app:app --reload --port 8000
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

By default, the frontend calls `http://localhost:8000`. If you need a different backend URL, set:

```bash
export NEXT_PUBLIC_API_URL="http://localhost:8000"
```

## API endpoints
- `POST /ingest` — upload a PDF (`multipart/form-data` with `file`)
- `POST /chat` — non-streaming answer (`{ "question": "...", "temperature": 0.2, "top_k": 4 }`)
- `POST /chat/stream` — streaming answer + reasoning deltas (SSE)
- `POST /plan` — generate dynamic progress steps for a question
- `GET /health` — health + model info

### Streaming example (SSE)

```bash
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key risks in this report.",
    "top_k": 4,
    "answer_style": "precise",
    "reasoning_effort": "medium"
  }'
```

## Notes
- The FAISS index is in-memory for now; extend by saving to disk in `backend/storage/`.
- Default model is `gpt-5.1`. If unavailable on your account, set `OPENAI_MODEL` to a supported model.
- Embeddings use `text-embedding-3-large` by default.
- Reasoning deltas are model-dependent; if you don’t see them, try a reasoning-capable model and adjust `reasoning_effort`.

## Next steps to extend
- Persist FAISS index per document and support multiple PDFs
- Add conversation memory
- Add auth + user workspaces
- Add streaming responses (already wired via `/chat/stream`)
