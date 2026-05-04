from __future__ import annotations

from urllib.parse import urlparse


def extract_domain(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def score_source(url: str, source_type: str | None = None) -> float:
    domain = extract_domain(url) or ""
    st = (source_type or "").strip().lower()

    if domain.endswith((".gov", ".edu")):
        return 0.95

    if st in {"official", "company_website", "company"}:
        return 0.85

    if st in {"review", "marketplace"}:
        return 0.80

    if st in {"news", "industry_report", "report"}:
        return 0.75

    if st in {"forum", "social"}:
        return 0.65

    if st in {"blog"}:
        return 0.55

    return 0.45

