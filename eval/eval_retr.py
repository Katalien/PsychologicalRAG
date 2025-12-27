import json
from collections import defaultdict
from typing import List

import numpy as np


def reciprocal_rank(categories: List[str], allowed_topics: List[str]) -> float:
    """
    Reciprocal Rank с учётом allowed_topics
    """
    for i, cat in enumerate(categories):
        if cat in allowed_topics:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(categories: List[str], allowed_topics: List[str]) -> int:
    """
    Hit@k с учётом allowed_topics
    """
    return int(any(cat in allowed_topics for cat in categories))


def precision_at_k(categories: List[str], allowed_topics: List[str], k: int) -> float:
    """
    Precision@k с учётом allowed_topics
    """
    if k == 0:
        return 0.0
    return sum(cat in allowed_topics for cat in categories) / k


def evaluate_retrieval(bot, benchmark_path: str, k: int):
    # Загружаем benchmark
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    questions = bench["questions"]

    hits = []
    precisions = []
    rrs = []

    per_theme_stats = defaultdict(list)
    detailed_results = []

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expected_theme = item["expected_theme"]
        allowed_topics = item["allowed_topics"]

        docs = bot.db.similarity_search(question, k=k)

        retrieved_categories = [
            doc.metadata.get("category") for doc in docs
        ]

        hit = hit_at_k(retrieved_categories, allowed_topics)
        precision = precision_at_k(retrieved_categories, allowed_topics, k)
        rr = reciprocal_rank(retrieved_categories, allowed_topics)

        hits.append(hit)
        precisions.append(precision)
        rrs.append(rr)

        per_theme_stats[expected_theme].append({
            "hit": hit,
            "precision": precision,
            "rr": rr
        })

        detailed_results.append({
            "id": qid,
            "question": question,
            "expected_theme": expected_theme,
            "allowed_topics": allowed_topics,
            "retrieved_categories": retrieved_categories,
            "hit@k": hit,
            "precision@k": precision,
            "rr": rr
        })

    results = {
        "overall": {
            f"Hit@{k}": float(np.mean(hits)),
            f"Precision@{k}": float(np.mean(precisions)),
            f"MRR@{k}": float(np.mean(rrs))
        },
        "per_theme": {}
    }

    for theme, values in per_theme_stats.items():
        results["per_theme"][theme] = {
            f"Hit@{k}": float(np.mean([v["hit"] for v in values])),
            f"Precision@{k}": float(np.mean([v["precision"] for v in values])),
            f"MRR@{k}": float(np.mean([v["rr"] for v in values]))
        }

    return results, detailed_results
