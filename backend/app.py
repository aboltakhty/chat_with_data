import json
import os
import time
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain.prompts import ChatPromptTemplate
except ModuleNotFoundError:
    from langchain_core.prompts import ChatPromptTemplate

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

app = FastAPI(title="PDF Chat Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
openai_client = OpenAI()

VECTOR_STORE = None
INDEX_METADATA = {}


class ChatRequest(BaseModel):
    question: str
    temperature: float = 0.2
    top_k: int = 4
    model: Optional[str] = None
    answer_style: str = "precise"
    reasoning_effort: str = "medium"


class PlanRequest(BaseModel):
    question: str
    model: Optional[str] = None


def build_vector_store(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, embeddings)
    return store, chunks


def analyze_question(question: str, llm: ChatOpenAI) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an analyst agent. Break the user question into key aspects, "
                "identify ambiguous terms, and propose sub-questions. Use 3-6 bullet "
                "points max. Keep it concise."
            ),
            ("human", "Question: {question}")
        ]
    )
    chain = prompt | llm
    return chain.invoke({"question": question}).content


def answer_question(question: str, analysis: str, context: str, llm: ChatOpenAI, style: str) -> str:
    style_map = {
        "precise": "Answer in 3-6 crisp bullet points. No extra context.",
        "balanced": "Answer in a short paragraph followed by 3-5 bullet points.",
        "detailed": "Answer in structured sections with brief headings, but stay grounded in the context."
    }
    style_instruction = style_map.get(style.lower(), style_map["precise"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an answer agent. Use the provided context to answer the question. "
                "Cite specific facts from the context. If the context is insufficient, "
                "say what is missing and answer with best-effort. " + style_instruction
            ),
            ("human", "Context:\n{context}\n\nAnalysis:\n{analysis}\n\nQuestion: {question}")
        ]
    )
    chain = prompt | llm
    return chain.invoke({"context": context, "analysis": analysis, "question": question}).content


def check_answer(question: str, analysis: str, context: str, answer: str, llm: ChatOpenAI) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a checker agent. Verify whether the answer addresses all aspects "
                "from the analysis and aligns with the context. List missing points or "
                "possible corrections. End with a verdict: PASS or NEEDS_REVISION."
            ),
            (
                "human",
                "Question: {question}\n\nAnalysis:\n{analysis}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
            )
        ]
    )
    chain = prompt | llm
    return chain.invoke({
        "question": question,
        "analysis": analysis,
        "context": context,
        "answer": answer
    }).content


def build_answer_prompt(context: str, question: str, style: str) -> tuple[str, str]:
    style_map = {
        "precise": "Answer in 3-6 crisp bullet points. No extra context.",
        "balanced": "Answer in a short paragraph followed by 3-5 bullet points.",
        "detailed": "Answer in structured sections with brief headings, but stay grounded in the context."
    }
    style_instruction = style_map.get(style.lower(), style_map["precise"])
    system_prompt = (
        "You are an answer agent. Use only the provided context to answer. "
        "Cite specific facts from the context. If context is insufficient, "
        "say what is missing and answer with best-effort. "
        + style_instruction
    )
    user_prompt = f"Context:\\n{context}\\n\\nQuestion: {question}"
    return system_prompt, user_prompt


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\\ndata: {json.dumps(payload)}\\n\\n"


def build_progress_steps(question: str, llm: ChatOpenAI) -> List[str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a UX agent. Generate 4-6 short, sequential status updates that "
                "describe what the system is doing to answer the user's question. "
                "Make them feel real and specific to the question. Output each step on its own line."
            ),
            ("human", "Question: {question}")
        ]
    )
    chain = prompt | llm
    raw = chain.invoke({"question": question}).content
    steps = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return steps[:6] if steps else []


def get_context(question: str, k: int = 4) -> str:
    if VECTOR_STORE is None:
        raise HTTPException(status_code=400, detail="No PDF loaded. Upload a PDF first.")
    docs = VECTOR_STORE.similarity_search(question, k=k)
    context_blocks = []
    for idx, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        context_blocks.append(f"[{idx}] (page {page})\n{doc.page_content}")
    return "\n\n".join(context_blocks)


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(STORAGE_DIR, exist_ok=True)
    timestamp = int(time.time())
    file_path = os.path.join(STORAGE_DIR, f"{timestamp}_{file.filename}")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    global VECTOR_STORE, INDEX_METADATA
    VECTOR_STORE, chunks = build_vector_store(file_path)
    INDEX_METADATA = {
        "filename": file.filename,
        "chunks": len(chunks),
        "file_path": file_path
    }

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks": len(chunks)
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model = request.model or OPENAI_MODEL
    temperature = max(0.0, min(request.temperature, 1.2))
    top_k = max(1, min(request.top_k, 8))
    answer_style = request.answer_style or "precise"

    llm = ChatOpenAI(model=model, temperature=temperature)
    context = get_context(question, k=top_k)
    analysis = analyze_question(question, llm)
    answer = answer_question(question, analysis, context, llm, answer_style)
    checker = check_answer(question, analysis, context, answer, llm)

    return {
        "answer": answer,
        "analysis": analysis,
        "checker": checker,
        "source": INDEX_METADATA
    }


@app.post("/plan")
async def plan(request: PlanRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    model = request.model or OPENAI_MODEL
    llm = ChatOpenAI(model=model, temperature=0.4)
    steps = build_progress_steps(question, llm)
    return {"steps": steps}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model = request.model or OPENAI_MODEL
    top_k = max(1, min(request.top_k, 8))
    answer_style = request.answer_style or "precise"
    reasoning_effort = request.reasoning_effort or "medium"

    context = get_context(question, k=top_k)
    system_prompt, user_prompt = build_answer_prompt(context, question, answer_style)

    def event_generator():
        try:
            stream = openai_client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                reasoning={"effort": reasoning_effort}
            )
            for event in stream:
                event_type = getattr(event, "type", None)
                if not event_type:
                    continue
                if event_type in (
                    "response.output_text.delta",
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta"
                ):
                    delta = getattr(event, "delta", "")
                    yield sse(event_type, {"delta": delta})
                elif event_type in (
                    "response.in_progress",
                    "response.completed",
                    "response.created"
                ):
                    yield sse(event_type, {"status": event_type})
                elif event_type == "error":
                    message = getattr(event, "message", "Unknown error")
                    yield sse("error", {"message": message})
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
        finally:
            yield sse("response.completed", {"status": "response.completed"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "has_index": VECTOR_STORE is not None
    }

# uvicorn app:app --reload --port 8000
