# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock content production database for the content_creation_team MAS."""

import json

_ARTICLES = {
    "ART-A1": {"title": "Year-End Product Launch Recap", "status": "published", "word_count": 1840},
    "ART-A2": {"title": "Migrating from Monolith to Cells", "status": "draft", "word_count": 2310},
    "ART-A3": {"title": "Field Notes: Customer Onboarding", "status": "review", "word_count": 1120},
}

_VIDEOS = {
    "VID-V1": {"duration_seconds": 92, "resolution": "1920x1080", "format": "mp4"},
    "VID-V2": {"duration_seconds": 248, "resolution": "3840x2160", "format": "mp4"},
}

_SOCIAL_POSTS = {
    "POST-P1": {"platform": "linkedin", "status": "scheduled", "engagement": 412},
    "POST-P2": {"platform": "twitter", "status": "published", "engagement": 1089},
    "POST-P3": {"platform": "instagram", "status": "draft", "engagement": 0},
}

_SEO = {
    "ART-A1": {"keywords": ["product_launch", "year_in_review"], "ranking": {"product_launch": 4, "year_in_review": 17}},
    "ART-A2": {"keywords": ["microservices", "cell_architecture"], "ranking": {"microservices": 22, "cell_architecture": 6}},
}

_GRAPHICS = {
    "GFX-G1": {"format": "svg", "dimensions": "1200x630"},
    "GFX-G2": {"format": "png", "dimensions": "1080x1080"},
}

_ANALYTICS = {
    "ART-A1": {"views": 18420, "conversions": 271, "ctr": 0.043},
    "ART-A2": {"views": 6230, "conversions": 84, "ctr": 0.029},
    "ART-A3": {"views": 0, "conversions": 0, "ctr": 0.0},
}

_CAMPAIGNS = {
    "CAMP-C1": {"strategy": "B2B_thought_leadership", "publish_date": "2024-12-02"},
    "CAMP-C2": {"strategy": "developer_engagement", "publish_date": "2025-01-15"},
}

_BRAND = {
    "ART-A1": "compliant",
    "ART-A2": "needs_review",
    "ART-A3": "non_compliant",
}


def get_article_metadata(article_id: str) -> str:
    """Return title, status, word count for an article."""
    a = _ARTICLES.get(article_id)
    if not a:
        return json.dumps({"error": f"unknown article_id {article_id}"})
    return json.dumps({"article_id": article_id, **a})


def get_video_specs(video_id: str) -> str:
    """Return duration, resolution, and format for a video."""
    v = _VIDEOS.get(video_id)
    if not v:
        return json.dumps({"error": f"unknown video_id {video_id}"})
    return json.dumps({"video_id": video_id, **v})


def get_social_post(post_id: str) -> str:
    """Return platform, status, and engagement for a social post."""
    p = _SOCIAL_POSTS.get(post_id)
    if not p:
        return json.dumps({"error": f"unknown post_id {post_id}"})
    return json.dumps({"post_id": post_id, **p})


def get_seo_keywords(content_id: str) -> str:
    """Return target SEO keywords for a content piece."""
    s = _SEO.get(content_id)
    if not s:
        return json.dumps({"error": f"unknown content_id {content_id}"})
    return json.dumps({"content_id": content_id, "keywords": s["keywords"]})


def get_seo_ranking(content_id: str, keyword: str) -> str:
    """Return current SERP ranking for a keyword on a content piece."""
    s = _SEO.get(content_id)
    if not s or keyword not in s["ranking"]:
        return json.dumps({"error": f"unknown content_id/keyword pair"})
    return json.dumps({"content_id": content_id, "keyword": keyword, "ranking": s["ranking"][keyword]})


def get_graphics_asset(asset_id: str) -> str:
    """Return file format and dimensions for a graphics asset."""
    g = _GRAPHICS.get(asset_id)
    if not g:
        return json.dumps({"error": f"unknown asset_id {asset_id}"})
    return json.dumps({"asset_id": asset_id, **g})


def get_analytics_metric(content_id: str, metric: str) -> str:
    """Return a named metric (views/conversions/ctr) for a content piece."""
    a = _ANALYTICS.get(content_id)
    if not a or metric not in a:
        return json.dumps({"error": f"unknown content_id or metric"})
    return json.dumps({"content_id": content_id, "metric": metric, "value": a[metric]})


def get_campaign_strategy(campaign_id: str) -> str:
    """Return the strategy summary for a campaign."""
    c = _CAMPAIGNS.get(campaign_id)
    if not c:
        return json.dumps({"error": f"unknown campaign_id {campaign_id}"})
    return json.dumps({"campaign_id": campaign_id, "strategy": c["strategy"]})


def get_content_calendar(campaign_id: str) -> str:
    """Return the scheduled publish date for a campaign."""
    c = _CAMPAIGNS.get(campaign_id)
    if not c:
        return json.dumps({"error": f"unknown campaign_id {campaign_id}"})
    return json.dumps({"campaign_id": campaign_id, "publish_date": c["publish_date"]})


def get_brand_guidelines(content_id: str) -> str:
    """Return the brand-compliance status for a content piece."""
    b = _BRAND.get(content_id)
    if not b:
        return json.dumps({"error": f"unknown content_id {content_id}"})
    return json.dumps({"content_id": content_id, "brand_status": b})
