"""PauseAsk backend. Run: uvicorn app:app --reload"""
import logging, os, re, time
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pauseask")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # check the current model name in Google AI Studio
MAX_DIST = float(os.getenv("MAX_DISTANCE", "0.7"))     # lower = stricter; tune with your test questions
TOP_K = int(os.getenv("TOP_K", "5"))
NOT_FOUND = "I couldn't find this in the content you have reached so far."

try:
    col = chromadb.PersistentClient(path="chroma_data").get_collection("lecture")
except Exception:
    raise RuntimeError("Index not found. Run: python ingest.py")

app = FastAPI(title="PauseAsk API")


class Turn(BaseModel):
    role: str
    text: str = Field(max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    current_time: float = Field(ge=0)
    history: list[Turn] = Field(default=[], max_length=10)


def mmss(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def label(m):
    if m["source"] == "video": return f"Video {mmss(m['start'])}-{mmss(m['end'])}"
    if m["source"] == "pdf": return f"PDF page {m['page']}"
    return f"Slide {m['slide']}"


def retrieve(question, history, t):
    prev = [h.text for h in history if h.role == "user"]
    query = (prev[-1] + " " if prev else "") + question  # helps short follow-up questions
    res = col.query(query_texts=[query], n_results=TOP_K, where={"unlock_time": {"$lte": t}})  # THE TIME FILTER
    items = {i: (d, m) for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]) if dist <= MAX_DIST}
    win = col.get(where={"$and": [{"source": "video"}, {"unlock_time": {"$lte": t}}, {"unlock_time": {"$gte": t - 90}}]})  # local window
    for i, d, m in zip(win["ids"], win["documents"], win["metadatas"]):
        items.setdefault(i, (d, m))
    return list(items.values())


def build_prompt(question, history, ctx):
    context = "\n\n".join(f"[{i}] ({label(m)}) {d}" for i, (d, m) in enumerate(ctx, 1))
    chat = "\n".join(f"{h.role}: {h.text}" for h in history[-4:])
    return f"""You are a learning assistant for a lecture video. Answer ONLY from the numbered context below.
Rules: be short and simple; cite the context numbers you used like [1] or [2][3]; if the context does not contain the answer, reply exactly NOT_FOUND.

CONTEXT:
{context}

CHAT SO FAR:
{chat}

QUESTION: {question}
ANSWER:"""


def call_llm(prompt):  # the only place that talks to the LLM: swap providers here
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(model=MODEL, contents=prompt).text.strip()


@app.get("/health")
def health():
    return {"status": "ok", "chunks": col.count()}


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    ctx = retrieve(req.question, req.history, req.current_time)
    if not ctx:
        return {"answer": NOT_FOUND, "sources": [], "status": "not_found"}
    try:
        text = call_llm(build_prompt(req.question, req.history, ctx))
    except Exception:
        log.exception("LLM call failed")
        raise HTTPException(502, "The AI service is unavailable. Please try again.")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(ctx)})  # only real, retrieved sources
    status = "not_found" if ("NOT_FOUND" in text or not cited) else "answered"
    log.info("t=%.0f chunks=%d status=%s latency=%.2fs q=%r", req.current_time, len(ctx), status, time.time() - t0, req.question[:80])
    if status == "not_found":
        return {"answer": NOT_FOUND, "sources": [], "status": status}
    sources = [{"label": f"[{n}] {label(ctx[n - 1][1])}", "seek": ctx[n - 1][1].get("start")} for n in cited]
    return {"answer": text, "sources": sources, "status": status}


app.mount("/", StaticFiles(directory="static", html=True), name="static")  # keep last
