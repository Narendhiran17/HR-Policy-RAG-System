# Novatech HR Assistant — RAG-Powered Policy Q&A

A retrieval-augmented generation system that answers employee questions about HR policy, grounded in a set of company policy documents, with source citations and no hallucinated answers.

## Why this project (not just "chat with a PDF")

Most beginner RAG demos are a single PDF and a happy-path question. This project's document set (`data/`) is deliberately built to exercise the retrieval problems a real production system runs into:

1. **Version conflicts** — `08_leave_policy_v2.0_2025.md` is a stale leave policy with the same title as the current one (`01_leave_policy.md`) but different numbers. The system retrieves both and asks the LLM to explicitly flag the discrepancy and identify the current version by effective date, rather than silently picking one.
2. **Paraphrase robustness** — `07_hr_faq.md` restates policy content in casual language, testing whether retrieval still finds it when the query wording doesn't match the formal policy's wording.
3. **Correct abstention** — the system must say "I don't know" for questions no document answers (e.g. stock options), not hallucinate a plausible-sounding policy.

## Architecture

```
data/*.md  →  chunking.py  →  embedding.py  →  ChromaDB  →  rag.py  →  main.py (FastAPI)
(HR docs)   (section-aware)  (TF-IDF+SVD)    (vector store) (retrieve  (API layer)
                                                              + generate)
```

- **Chunking** (`app/chunking.py`): splits each doc along its `##`/`###` headers rather than fixed character windows, so each chunk is a semantically complete policy section. Metadata (title, version, effective date, section) travels with every chunk.
- **Embedding** (`app/embedding.py`): TF-IDF + Truncated SVD (a form of LSA), fit on the document corpus. **This is a deliberate offline substitute** — see "Known limitations" below for why and what to swap in for production use.
- **Vector store**: ChromaDB, persisted locally to `chroma_store/`.
- **Retrieval + generation** (`app/rag.py`): retrieves top-k chunks, filters by a relevance threshold (so weak matches don't get passed to the LLM), and calls Claude with a system prompt that enforces grounded, citation-backed, abstention-capable answers.
- **API** (`app/main.py`): FastAPI, one `/query` endpoint, `/health` for readiness checks.

## Setup

```bash
cd hr_rag_assistant
pip install -r requirements.txt

# Build the vector index (run once, and again after editing data/*.md)
python app/ingest.py

# Set your Claude API key
export ANTHROPIC_API_KEY=your_key_here

# Start the API
uvicorn app.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many casual leave days do I get?"}'
```

Or query directly from the command line without the API:
```bash
python app/rag.py "How many casual leave days do I get?"
```

## Evaluation

```bash
python eval/run_eval.py
```

Runs 8 hand-labeled test questions (`eval/qa_test_set.json`) against the retrieval layer and reports hit rate — i.e., did the expected source document show up in the top-5 retrieved chunks? Includes the version-conflict case and a genuine no-answer case. Currently: **8/8 (100%)**, with `RELEVANCE_THRESHOLD` empirically tuned against this set (see the comment in `app/rag.py`).

This checks retrieval only, not generation quality. For end-to-end answer quality (faithfulness, answer relevancy, context precision), the natural next step is [RAGAS](https://github.com/explodinggradients/ragas) — it needs LLM calls to judge answers, so it's kept as a documented next step rather than baked in here.

## Known limitations (be upfront about these — they're good interview material)

1. **TF-IDF instead of dense embeddings.** ChromaDB's default embedding function downloads a transformer model from an external host; that download isn't guaranteed in every environment (locked-down CI, offline dev boxes). TF-IDF+SVD needs no download and is fully reproducible, but it matches on shared vocabulary rather than deep semantic meaning — it will do worse than a transformer embedding on queries that paraphrase heavily with little word overlap. **Upgrade path:** swap `embedding.py`'s `TfidfEmbeddingFunction` for `chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")` — nothing else in the pipeline needs to change.
2. **Small corpus.** 8 documents / ~72 chunks. The relevance threshold was tuned against this specific corpus's similarity distribution; it would need re-tuning on a larger, more diverse document set.
3. **No conversation memory.** Each query is independent — no multi-turn follow-up handling yet.
4. **No re-ranking model.** Retrieval currently ranks purely by embedding similarity plus a same-title freshness tiebreak; a dedicated cross-encoder re-ranker would likely improve precision further.

## Project structure
```
hr_rag_assistant/
├── data/                    # Source HR policy documents (8 .md files)
├── app/
│   ├── chunking.py          # Header-aware document chunking
│   ├── embedding.py         # TF-IDF+SVD embedding function (offline)
│   ├── ingest.py            # Builds the ChromaDB index
│   ├── rag.py                # Retrieval + grounded generation
│   └── main.py               # FastAPI app
├── eval/
│   ├── qa_test_set.json     # Hand-labeled retrieval test cases
│   └── run_eval.py          # Retrieval evaluation harness
├── chroma_store/            # Persisted vector index (generated by ingest.py)
└── requirements.txt
```
