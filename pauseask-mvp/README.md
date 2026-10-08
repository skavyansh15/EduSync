# PauseAsk (MVP)

Pause a lecture, ask a question, get an answer from **only the content up to that moment**, with sources.
Python (FastAPI + ChromaDB + Gemini free tier) backend, plain HTML/CSS/JS page.

## Run it locally
1. Install Python 3.10+. In this folder: `python -m venv venv`, activate it, then `pip install -r requirements.txt`
2. Get a free API key in Google AI Studio. Copy `.env.example` to `.env` and paste the key. Check the model name in AI Studio.
3. `python ingest.py` builds the index from `data/` (a sample lecture is included; the first run downloads a small embedding model).
4. `uvicorn app:app --reload`, then open http://127.0.0.1:8000 (API docs at /docs, health check at /health).

## Try these (sample lecture, type the time in the page if you have no video)
- Time 100, "What is overfitting?" -> answered, with sources.
- Time 100, "What is L1 regularization?" -> not found (taught later, so it is locked).
- Time 200, "What is L1 regularization?" -> answered.
- Time 200, "Who invented the telephone?" -> not found.
- Ask a follow-up such as "Can you give a simpler example?" -> uses recent chat.

## Use your own content
Put these in `data/`: `lecture.srt` (subtitles), `lecture.pdf`, `lecture.pptx`; put your video at `static/lecture.mp4` (keep it small).
Edit `data/unlock_map.json`: for every PDF page and PPT slide, the video second where it is first taught.
**Pages/slides missing from this file stay locked** (never shown). Run `python ingest.py` again after any change.

## How it works
`ingest.py`: subtitles -> ~20-30 s chunks (PDF page / PPT slide each one chunk) + metadata (`unlock_time`) -> ChromaDB.
`app.py` `/ask`: filter `unlock_time <= current_time` -> top matches + last ~90 s of transcript -> LLM answers only from that context
-> we keep only sources actually cited and retrieved -> otherwise "not found". Logs every request.

## Deploy (Render free tier, as far as we know: check current terms)
Build command: `pip install -r requirements.txt && python ingest.py` | Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
Environment variables: `GEMINI_API_KEY` (never commit it), optionally `GEMINI_MODEL`, `MAX_DISTANCE`.

## To tune / known limits
- `MAX_DISTANCE` (default 0.7) decides what counts as relevant; tune it on your ~20 test questions.
- `UNLOCK_AT=end` is strict (a chunk unlocks when it finishes); `start` gives slightly fresher context but may peek up to one chunk ahead.
- No login, no database for chat (the browser keeps the last few turns), no rate limiting yet.
