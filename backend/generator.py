"""
generator.py - LLM generation with smart fallback gate
Uses openai/gpt-oss-120b
"""
import os
from groq import Groq
from dotenv import load_dotenv
from retriever import retrieve

load_dotenv()

_client          = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL            = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
SCORE_THRESHOLD  = 0.0    # CrossEncoder scores; tune after testing with your docs
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are StudyRAG, a helpful AI study assistant.

Rules:
- When answering from documents, cite sources inline as [Doc: filename, Page: N].
- When answering from general knowledge, start with: "This isn't covered in your uploaded documents, but from general knowledge:"
- Be concise and structured. Use bullet points for lists of facts.
- If a question is ambiguous, answer the most likely interpretation.
- Never make up citations or page numbers."""

def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        parts.append(f"[{i+1}] {c['source']}, Page {c['page']}:\n{c['text']}")
    return "\n\n---\n\n".join(parts)
MAX_TOKENS = 4096
EXTRA = {"reasoning_effort": "low"} if MODEL.startswith("openai/gpt-oss") else {}

def _prepare(query: str, chat_history: list[dict], source_filter: str | None = None):
    chunks     = retrieve(query, source_filter=source_filter)
    best_score = max((c["score"] for c in chunks), default=-999)
    use_docs   = bool(chunks) and best_score > SCORE_THRESHOLD

    if use_docs:
        user_content = (
            "Answer the question using ONLY the document excerpts below. "
            "Cite each fact as [Doc: filename, Page: N]. "
            "If the excerpts don't contain enough information, say so.\n\n"
            f"Document excerpts:\n{_format_context(chunks)}\n\n"
            f"Question: {query}"
        )
    else:
        user_content = (
            "The user's uploaded documents don't appear to cover this topic. "
            "Answer from your general knowledge and clearly say so at the start.\n\n"
            f"Question: {query}"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in chat_history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_content})
    return messages, (chunks if use_docs else []), use_docs


def answer(query: str, chat_history: list[dict], source_filter: str | None = None) -> dict:
    messages, sources, use_docs = _prepare(query, chat_history, source_filter)
    resp = _client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2,
        max_tokens=MAX_TOKENS, **EXTRA,
    )
    return {"answer": resp.choices[0].message.content,
            "sources": sources, "from_docs": use_docs}


def answer_stream(query: str, chat_history: list[dict], source_filter: str | None = None):
    messages, sources, use_docs = _prepare(query, chat_history, source_filter)
    yield {"type": "sources", "data": sources}
    yield {"type": "meta", "data": {"from_docs": use_docs}}

    stream = _client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2,
        max_tokens=MAX_TOKENS, stream=True, **EXTRA,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            yield {"type": "token", "data": token}