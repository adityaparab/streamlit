"""LLM-as-a-judge evaluation of the retriever in `extract.py`.

Runs retrieval for one or more queries, then asks an Ollama LLM to score the
relevance of each retrieved chunk on a 1-5 scale. Aggregates per-query and
overall metrics so the quality of the retriever can be assessed.
"""

import argparse
import json
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from extract import OLLAMA_GENERATION_MODEL, OLLAMA_HOST, search

load_dotenv()

# Reuse the generation model from `.env` as the judge.
JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", OLLAMA_GENERATION_MODEL)

JUDGE_INSTRUCTIONS = (
    "You are an impartial evaluator of a document retrieval system. "
    "Given a user QUERY and a retrieved CHUNK, judge how relevant the chunk is "
    "to answering the query.\n\n"
    "Score on this scale:\n"
    "  1 = irrelevant\n"
    "  2 = mostly irrelevant\n"
    "  3 = partially relevant\n"
    "  4 = relevant\n"
    "  5 = highly relevant / directly answers the query\n\n"
    "Respond with ONLY a JSON object: "
    '{"score": <int 1-5>, "reason": "<one sentence>"}'
)


def _judge_chunk(llm: ChatOllama, query: str, chunk: str) -> dict:
    """Ask the judge LLM to score a single retrieved chunk."""
    prompt = (
        f"{JUDGE_INSTRUCTIONS}\n\nQUERY: {query}\n\nCHUNK:\n{chunk}\n\nJSON:"
    )
    raw = llm.invoke(prompt).content.strip()

    # Be forgiving about models that wrap JSON in prose or code fences.
    start, end = raw.find("{"), raw.rfind("}")
    try:
        parsed = json.loads(raw[start : end + 1])
        score = int(parsed.get("score", 0))
        reason = str(parsed.get("reason", "")).strip()
    except (ValueError, json.JSONDecodeError):
        score, reason = 0, f"unparseable judge output: {raw[:120]!r}"

    return {"score": score, "reason": reason}


def evaluate_query(llm: ChatOllama, query: str, k: int = 4) -> dict:
    """Retrieve for `query` and have the judge score each retrieved chunk."""
    results = search(query, k=k)

    judged = []
    for doc, retrieval_score in results:
        verdict = _judge_chunk(llm, query, doc.page_content)
        judged.append(
            {
                "source": doc.metadata.get("source", "?"),
                "page": doc.metadata.get("page", "?"),
                "chunk_index": doc.metadata.get("chunk_index", "?"),
                "retrieval_score": round(float(retrieval_score), 4),
                "relevance": verdict["score"],
                "reason": verdict["reason"],
            }
        )

    scores = [j["relevance"] for j in judged if j["relevance"] > 0]
    avg = sum(scores) / len(scores) if scores else 0.0
    # Precision@k: fraction of retrieved chunks judged relevant (>= 4).
    precision = sum(1 for s in scores if s >= 4) / len(judged) if judged else 0.0

    return {
        "query": query,
        "avg_relevance": round(avg, 3),
        "precision_at_k": round(precision, 3),
        "chunks": judged,
    }


def judge(queries: list[str], k: int = 4) -> dict:
    """Evaluate retriever quality across `queries` using an LLM judge."""
    llm = ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_HOST, temperature=0, streaming=True)

    per_query = []
    for query in queries:
        report = evaluate_query(llm, query, k=k)
        per_query.append(report)

        print(f"\nQuery: {report['query']!r}")
        print(
            f"  avg_relevance={report['avg_relevance']}  "
            f"precision@{k}={report['precision_at_k']}"
        )
        for c in report["chunks"]:
            print(
                f"    [rel={c['relevance']}] {c['source']} "
                f"p{c['page']} c{c['chunk_index']}: {c['reason']}"
            )

    overall_avg = (
        sum(r["avg_relevance"] for r in per_query) / len(per_query)
        if per_query
        else 0.0
    )
    overall_prec = (
        sum(r["precision_at_k"] for r in per_query) / len(per_query)
        if per_query
        else 0.0
    )

    print("\n" + "=" * 60)
    print(
        f"OVERALL  avg_relevance={round(overall_avg, 3)}  "
        f"precision@{k}={round(overall_prec, 3)}  (judge={JUDGE_MODEL})"
    )

    return {
        "judge_model": JUDGE_MODEL,
        "k": k,
        "overall_avg_relevance": round(overall_avg, 3),
        "overall_precision_at_k": round(overall_prec, 3),
        "queries": per_query,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge evaluation of the retriever."
    )
    parser.add_argument("queries", nargs="+", help="One or more queries to evaluate.")
    parser.add_argument(
        "-k", type=int, default=4, help="Number of chunks to retrieve per query."
    )
    args = parser.parse_args()
    judge(args.queries, k=args.k)
