# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock article database for the deep_research_team MAS."""

import json

_ARTICLES = {
    "ART-001": {
        "topic": "transformer_attention",
        "title": "Efficient Attention Patterns in Long-Context Transformers",
        "authors": ["L. Marquez", "J. Patel"],
        "year": 2023,
        "venue": "NeurIPS",
        "content": "We study sparse attention masks that reduce compute by 47% while preserving perplexity within 0.3 nats.",
        "findings": "Sparse local-global mixed attention reduces FLOPs by 47% with negligible quality loss.",
        "citations": 184,
        "key_metric": "47%",
    },
    "ART-002": {
        "topic": "diffusion_models",
        "title": "Latent Consistency Models for Real-Time Image Synthesis",
        "authors": ["A. Becker", "C. Yamamoto"],
        "year": 2024,
        "venue": "ICLR",
        "content": "Distillation of diffusion models into 4-step samplers reaching FID 6.2 on ImageNet 256.",
        "findings": "Consistency distillation enables 4-step generation at FID 6.2 with no adversarial loss.",
        "citations": 91,
        "key_metric": "6.2",
    },
    "ART-003": {
        "topic": "rl_from_human_feedback",
        "title": "Direct Preference Optimization without a Reward Model",
        "authors": ["S. Ahn", "M. Chowdhury", "K. Estevez"],
        "year": 2023,
        "venue": "NeurIPS",
        "content": "DPO replaces RLHF reward modeling with a closed-form preference loss, reaching parity on MT-Bench.",
        "findings": "DPO achieves MT-Bench score 7.41, matching PPO-RLHF without a separate reward model.",
        "citations": 612,
        "key_metric": "7.41",
    },
    "ART-004": {
        "topic": "graph_neural_networks",
        "title": "Heterogeneous Message Passing for Citation Networks",
        "authors": ["O. Kowalski"],
        "year": 2022,
        "venue": "KDD",
        "content": "Edge-typed message passing improves node classification accuracy by 3.8 points on OGB-MAG.",
        "findings": "Heterogeneous edge typing yields 3.8-point F1 improvement on OGB-MAG citation prediction.",
        "citations": 245,
        "key_metric": "3.8",
    },
    "ART-005": {
        "topic": "transformer_attention",
        "title": "FlashAttention-2: Faster Attention with Better Parallelism",
        "authors": ["T. Dao"],
        "year": 2023,
        "venue": "arXiv",
        "content": "Improved tiling and warp-specialization deliver 2.0x speedup over FlashAttention v1.",
        "findings": "FlashAttention-2 doubles end-to-end throughput vs v1 on A100 for sequence length 8192.",
        "citations": 503,
        "key_metric": "2.0x",
    },
}

_TOPIC_SUMMARIES = {
    "transformer_attention": "Recent work on attention focuses on reducing the O(n^2) cost via sparsity, low-rank, or kernel approximations.",
    "diffusion_models": "Diffusion models dominate image generation; recent advances focus on sampler distillation for real-time use.",
    "rl_from_human_feedback": "RLHF aligns LLMs with preferences; DPO recently removed the need for a separate reward model.",
    "graph_neural_networks": "GNNs propagate information along edges; heterogeneous extensions handle multiple edge types.",
}


def search_articles(query: str) -> str:
    """Return article IDs whose topic matches the query (substring match)."""
    q = query.lower().replace(" ", "_")
    matches = [aid for aid, a in _ARTICLES.items() if q in a["topic"]]
    return json.dumps({"query": query, "article_ids": matches})


def get_article_topic(article_id: str) -> str:
    """Return the topic of an article."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({"article_id": article_id, "topic": a["topic"]})


def get_article_content(article_id: str) -> str:
    """Return the article body text."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({"article_id": article_id, "content": a["content"]})


def get_article_metadata(article_id: str) -> str:
    """Return article title, authors, year, and venue."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({
        "article_id": article_id,
        "title": a["title"],
        "authors": a["authors"],
        "year": a["year"],
        "venue": a["venue"],
    })


def get_article_findings(article_id: str) -> str:
    """Return the key findings of an article."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({"article_id": article_id, "findings": a["findings"]})


def get_article_citations(article_id: str) -> str:
    """Return the citation count of an article."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({"article_id": article_id, "citations": a["citations"]})


def get_topic_summary(topic: str) -> str:
    """Return a short background summary for a research topic."""
    s = _TOPIC_SUMMARIES.get(topic)
    if not s:
        return json.dumps({"error": f"unknown topic {topic}"})
    return json.dumps({"topic": topic, "summary": s})


def get_key_metric(article_id: str) -> str:
    """Return the headline quantitative metric reported in an article."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({"article_id": article_id, "key_metric": a["key_metric"]})
