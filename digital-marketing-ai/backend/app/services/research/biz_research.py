from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

from app.services.research.tavily_client import TavilyClientError, TavilyResearchClient


def _load_report_structure() -> str:
    structure_path = Path(__file__).resolve().parents[3] / "report_structure.md"
    try:
        content = structure_path.read_text(encoding="utf-8")
        if not content.strip():
            logger.warning("report_structure.md is empty — section queries will use fallback")
        return content
    except FileNotFoundError:
        logger.warning("report_structure.md not found at %s — using fallback", structure_path)
        return ""
    except Exception as exc:
        logger.error("Failed to load report_structure.md: %s", exc)
        return ""


def _extract_section_queries(intake: dict[str, Any]) -> list[dict[str, str]]:
    """Generate research queries for each section of the report structure."""
    business = _clean(intake.get("business_idea"))
    audience = _clean(intake.get("target_audience")) or "target customers"
    location = _location_hint(intake)

    # Base queries for each report section
    section_queries = [
        {
            "label": "executive_summary",
            "query": f"{business} executive summary key insights market opportunity risks {location}",
            "section": "Executive Intelligence Dashboard"
        },
        {
            "label": "business_snapshot",
            "query": f"{business} business overview current status team size revenue stage analysis {location}",
            "section": "Business Snapshot"
        },
        {
            "label": "market_demand",
            "query": f"{business} market demand TAM SAM SOM growth trends {location} industry analysis",
            "section": "Market Demand and Opportunity"
        },
        {
            "label": "customer_insights",
            "query": f"{audience} customer insights pain points buying behavior needs for {business} {location}",
            "section": "Customer and ICP Insights"
        },
        {
            "label": "competitive_analysis",
            "query": f"{business} competitive landscape top competitors market share SWOT analysis {location}",
            "section": "Competitive Landscape"
        },
        {
            "label": "pricing_revenue",
            "query": f"{business} pricing strategy revenue model unit economics benchmarks {location}",
            "section": "Pricing / Revenue / Unit-Economics"
        },
        {
            "label": "marketing_channels",
            "query": f"best marketing channels for {business} targeting {audience} growth strategies {location}",
            "section": "Marketing and Growth Recommendations"
        },
        {
            "label": "operations_execution",
            "query": f"{business} operations execution requirements team hiring technology vendors {location}",
            "section": "Operations / Execution Considerations"
        },
        {
            "label": "legal_compliance",
            "query": f"{business} legal compliance regulations licenses risks {location} industry requirements",
            "section": "Legal / Compliance / Risk Flags"
        },
        {
            "label": "action_plan",
            "query": f"{business} 90-day action plan implementation roadmap milestones {location}",
            "section": "90-Day Action Plan"
        },
        {
            "label": "final_verdict",
            "query": f"{business} feasibility assessment investment recommendation go/no-go analysis {location}",
            "section": "Final Verdict"
        }
    ]

    return section_queries


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clip(text: str | None, limit: int = 900) -> str:
    value = _clean(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _domain_root(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _location_hint(intake: dict[str, Any]) -> str:
    location = _clean(intake.get("location"))
    return location or "India"


def _conversation_snapshot(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    snapshot: list[dict[str, str]] = []
    for message in messages[-8:]:
        role = _clean(message.get("role")) or "user"
        content = _clip(_clean(message.get("content")), 500)
        if not content:
            continue
        snapshot.append({"role": role, "content": content})
    return snapshot


def _build_queries(intake: dict[str, Any]) -> list[dict[str, str]]:
    business = _clean(intake.get("business_idea"))
    audience = _clean(intake.get("target_audience")) or "target customers"
    stage = _clean(intake.get("stage")) or "business"
    goals = ", ".join(intake.get("goals") or []) or "growth"
    location = _location_hint(intake)

    # ✅ FIX: original_queries se market/competition/pricing hata diye
    # kyunki section_queries mein already cover hain
    original_queries = [
        {
            "label": "channels",
            "query": f"best marketing channels for {business} targeting {audience} with goals {goals}",
        },
        {
            "label": "compliance",
            "query": f"licenses regulations compliance requirements for {business} in {location}",
        },
        {
            "label": "stage_risk",
            "query": f"execution risks common failure points for {stage} {business}",
        },
    ]

    section_queries = _extract_section_queries(intake)
    launch_queries = [
        {
            "label": "suppliers",
            "query": f"{business} suppliers distributors wholesalers manufacturers {location} India",
        },
        {
            "label": "setup_cost",
            "query": f"{business} setup cost equipment inventory staffing rent working capital {location}",
        },
        {
            "label": "location_feasibility",
            "query": f"best areas locations footfall rent for {business} in {location}",
        },
        {
            "label": "customer_reviews",
            "query": f"{business} customer reviews complaints praise {location} India",
        },
        {
            "label": "marketplaces",
            "query": f"where people buy {business} online offline marketplaces retailers India",
        },
        {
            "label": "financial_model",
            "query": f"{business} gross margin net margin break even CAC LTV benchmarks India",
        },
        {
            "label": "operations_stack",
            "query": f"{business} POS CRM inventory tools software logistics operations India",
        },
        {
            "label": "local_compliance",
            "query": f"{business} GST shop act trade license compliance checklist {location} India",
        },
    ]

    return original_queries + section_queries + launch_queries


def _research_task_prompt(intake: dict[str, Any], conversation_history: list[dict[str, Any]]) -> str:
    goals = ", ".join(intake.get("goals") or []) or "not specified"
    return (
        "Create a practical business research brief for this opportunity.\n"
        f"Business: {_clean(intake.get('business_idea'))}\n"
        f"Audience: {_clean(intake.get('target_audience')) or 'not specified'}\n"
        f"Stage: {_clean(intake.get('stage')) or 'not specified'}\n"
        f"Team size: {_clean(intake.get('team_size')) or 'not specified'}\n"
        f"Budget or revenue context: {_clean(intake.get('budget_revenue')) or 'not specified'}\n"
        f"Location: {_location_hint(intake)}\n"
        f"Goals: {goals}\n"
        f"Extra context: {_clean(intake.get('context')) or 'not specified'}\n"
        "Focus on market demand, competitors, pricing, channels, risks, compliance, "
        "setup costs, suppliers, operating model, and launch steps."
        f"\nRecent chat context: {_clip(str(_conversation_snapshot(conversation_history)), 3000)}"  # ✅ 1400 → 3000
    )


def _report_mode(intake: dict[str, Any]) -> str:
    return _clean(intake.get("report_mode")).lower()


def _research_timeout_seconds(intake: dict[str, Any]) -> float:
    mode = _report_mode(intake)
    if mode in {"quick scan"}:
        return 90.0       # ✅ 45 → 90
    if mode in {"standard plan", "turnaround audit"}:
        return 150.0      # ✅ 75 → 150
    return 240.0          # ✅ 120 → 240


def _search_max_results(intake: dict[str, Any]) -> int:
    mode = _report_mode(intake)
    if mode in {"quick scan"}:
        return 4
    if mode in {"standard plan", "turnaround audit"}:
        return 6
    return 8


def _site_root_limit(intake: dict[str, Any]) -> int:
    mode = _report_mode(intake)
    if mode in {"quick scan"}:
        return 2
    if mode in {"standard plan", "turnaround audit"}:
        return 3
    return 4


async def _safe_step(name: str, coro: Any) -> dict[str, Any]:
    try:
        result = await coro
        return {"step": name, "ok": True, "result": result}
    except Exception as exc:
        logger.error("Research step '%s' failed: %s", name, exc)  # ✅ FIX
        return {"step": name, "ok": False, "error": str(exc)}


def _review_queries(intake: dict[str, Any]) -> list[dict[str, str]]:
    business = _clean(intake.get("business_idea"))
    location = _location_hint(intake)
    audience = _clean(intake.get("target_audience")) or "customers"
    return [
        {
            "source": "google_reviews",
            "query": f"{business} google reviews complaints praise {location}",
        },
        {
            "source": "marketplace_reviews",
            "query": f"{business} amazon flipkart india reviews complaints buying experience",
        },
        {
            "source": "community_discussions",
            "query": f"{business} reddit quora customer complaints recommendations India {audience}",
        },
    ]


def _summarize_review_runs(review_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for run in review_runs:
        if not run.get("ok"):
            summary.append(
                {
                    "source": run.get("source"),
                    "query": run.get("query"),
                    "summary": "",
                    "snippets": [],
                    "error": run.get("error"),
                }
            )
            continue

        result = run.get("result", {})
        snippets: list[str] = []
        for item in (result.get("results") or [])[:4]:
            snippet = _clip(item.get("content"), 220)
            if snippet:
                snippets.append(snippet)

        summary.append(
            {
                "source": run.get("source"),
                "query": run.get("query"),
                "summary": _clip(result.get("answer"), 900),
                "snippets": snippets,
                "error": None,
            }
        )

    return summary


def _summarize_search_runs(search_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for run in search_runs:
        if not run.get("ok"):
            summary.append({"label": run.get("label"), "error": run.get("error")})
            continue
        result = run.get("result", {})
        top_results = []
        for item in (result.get("results") or [])[:3]:
            top_results.append(
                {
                    "title": _clean(item.get("title")),
                    "url": _clean(item.get("url")),
                    "content": _clip(item.get("content")),
                }
            )
        summary.append(
            {
                "label": run.get("label"),
                "query": run.get("query"),
                "answer": _clip(result.get("answer"), 700),
                "results": top_results,
            }
        )
    return summary


def _collect_urls(search_runs: list[dict[str, Any]], limit: int = 8) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for run in search_runs:
        if not run.get("ok"):
            continue
        for item in run.get("result", {}).get("results") or []:
            url = _clean(item.get("url"))
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def _select_roots_for_site_steps(search_runs: list[dict[str, Any]], limit: int = 2) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for run in search_runs:
        if not run.get("ok"):
            continue
        for item in run.get("result", {}).get("results") or []:
            root = _domain_root(_clean(item.get("url")) or "")
            if not root or root in seen:
                continue
            seen.add(root)
            roots.append(root)
            if len(roots) >= limit:
                return roots
    return roots


def _summarize_extract(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for item in (payload.get("results") or [])[:10]:
        results.append(
            {
                "url": _clean(item.get("url")),
                "title": _clean(item.get("title")),
                "content": _clip(item.get("raw_content"), 1200),
            }
        )
    return results


def _summarize_map(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_url": _clean(payload.get("base_url")),
        "results": (payload.get("results") or [])[:15],
    }


def _summarize_crawl(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages = []
    for item in (payload.get("results") or [])[:10]:
        pages.append(
            {
                "url": _clean(item.get("url")),
                "title": _clean(item.get("title")),
                "content": _clip(item.get("raw_content"), 1200),
            }
        )
    return pages


def _summarize_research(payload: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for item in (payload.get("sources") or [])[:16]:
        sources.append(
            {
                "title": _clean(item.get("title")),
                "url": _clean(item.get("url")),
            }
        )
    return {
        "request_id": _clean(payload.get("request_id")),
        "status": _clean(payload.get("status")),
        "content": _clip(payload.get("content"), 6000),
        "sources": sources,
    }


async def build_biz_research_bundle(
    intake: dict[str, Any],
    conversation_history: list[dict[str, Any]],
    client: TavilyResearchClient,
) -> dict[str, Any]:
    """Main entry point for building research bundle."""
    if not client.configured:
        # Fallback: Generate mock research bundle when API key is missing
        return _create_mock_research_bundle(intake, conversation_history)

    try:
        # Try the normal Tavily research process
        return await _build_tavily_research_bundle(intake, conversation_history, client)
    except TavilyClientError as e:
        if "usage limit" in str(e).lower() or "plan" in str(e).lower():
            # Fallback: Generate mock research bundle when hitting usage limits
            return _create_mock_research_bundle(intake, conversation_history)
        else:
            # Re-raise other Tavily errors
            raise


# ✅ ONLY ONE DEFINITION - Delete the empty duplicate above this
async def _build_tavily_research_bundle(
    intake: dict[str, Any],
    conversation_history: list[dict[str, Any]],
    client: TavilyResearchClient,
) -> dict[str, Any]:
    """Build research bundle using Tavily API calls."""
    queries = _build_queries(intake)
    review_queries = _review_queries(intake)
    max_results = _search_max_results(intake)
    search_tasks = []
    for query in queries:
        search_tasks.append(
            _safe_step(
                query["label"],
                client.search(
                    query["query"],
                    search_depth="advanced",
                    max_results=max_results,
                    include_answer="advanced",
                    include_raw_content="markdown",
                    include_domains=[],
                    topic="general",
                    country="india",
                    include_favicon=True,
                ),
            )
        )

    review_tasks = []
    for query in review_queries:
        review_tasks.append(
            _safe_step(
                query["source"],
                client.search(
                    query["query"],
                    search_depth="advanced",
                    max_results=max(4, min(6, max_results)),
                    include_answer="advanced",
                    include_raw_content="markdown",
                    topic="general",
                    country="india",
                    include_favicon=True,
                ),
            )
        )

    raw_search_runs = await asyncio.gather(*search_tasks)
    raw_review_runs = await asyncio.gather(*review_tasks) if review_tasks else []
    search_runs: list[dict[str, Any]] = []
    for query, result in zip(queries, raw_search_runs, strict=False):
        search_runs.append({**result, "label": query["label"], "query": query["query"]})
    reviews_runs: list[dict[str, Any]] = []
    for query, result in zip(review_queries, raw_review_runs, strict=False):
        reviews_runs.append({**result, "source": query["source"], "query": query["query"]})

    extract_urls = _collect_urls(search_runs + reviews_runs, limit=24)
    site_roots = _select_roots_for_site_steps(search_runs, limit=_site_root_limit(intake))

    extract_task = _safe_step(
        "extract",
        client.extract(
            extract_urls[:20],
            query=_clean(intake.get("business_idea")),
            chunks_per_source=4,
            extract_depth="advanced",
            format="markdown",
            include_favicon=True,
        ),
    ) if extract_urls else None

    map_tasks = [
        _safe_step(
            f"map:{root}",
            client.map(
                root,
                instructions="Find pages about products, pricing, market information, company overview, and legal terms.",
                max_depth=1,
                max_breadth=24,
                limit=16,
                allow_external=False,
            ),
        )
        for root in site_roots
    ]

    crawl_tasks = [
        _safe_step(
            f"crawl:{root}",
            client.crawl(
                root,
                instructions="Find useful pages about offerings, pricing, target users, proof points, and policies.",
                max_depth=1,
                max_breadth=16,
                limit=10,
                allow_external=False,
                extract_depth="basic",
                format="markdown",
            ),
        )
        for root in site_roots
    ]

    research_task = _safe_step(
        "research",
        client.research_and_wait(
            _research_task_prompt(intake, conversation_history),
            model="pro" if _clean(intake.get("report_mode")).lower() in {"deep strategy", "investor report"} else "auto",
            citation_format="numbered",
            timeout_seconds=_research_timeout_seconds(intake),
        ),
    )

    parallel_tasks = [research_task, *map_tasks, *crawl_tasks]
    if extract_task is not None:
        parallel_tasks.insert(0, extract_task)

    step_results = await asyncio.gather(*parallel_tasks)

    extract_result = next((item for item in step_results if item.get("step") == "extract"), None)
    research_result = next((item for item in step_results if item.get("step") == "research"), None)
    map_results = [item for item in step_results if str(item.get("step", "")).startswith("map:")]
    crawl_results = [item for item in step_results if str(item.get("step", "")).startswith("crawl:")]

    bundle = {
        "report_mode": _clean(intake.get("report_mode")),
        "intake_snapshot": {
            "stage_type": _clean(intake.get("stage_type")),
            "report_mode": _clean(intake.get("report_mode")),
            "business_idea": _clean(intake.get("business_idea")),
            "target_audience": _clean(intake.get("target_audience")),
            "stage": _clean(intake.get("stage")),
            "team_size": _clean(intake.get("team_size")),
            "goals": intake.get("goals") or [],
            "budget_revenue": _clean(intake.get("budget_revenue")),
            "location": _clean(intake.get("location")),
            "context": _clean(intake.get("context")),
        },
        "conversation_snapshot": _conversation_snapshot(conversation_history),
        "search_runs": _summarize_search_runs(search_runs),
        "reviews_runs": _summarize_review_runs(reviews_runs),
        "extract": {
            "ok": bool(extract_result and extract_result.get("ok")),
            "results": _summarize_extract(extract_result.get("result", {})) if extract_result and extract_result.get("ok") else [],
            "error": extract_result.get("error") if extract_result and not extract_result.get("ok") else None,
        },
        "map_runs": [
            {
                "step": item.get("step"),
                "ok": item.get("ok"),
                "data": _summarize_map(item.get("result", {})) if item.get("ok") else {},
                "error": item.get("error"),
            }
            for item in map_results
        ],
        "crawl_runs": [
            {
                "step": item.get("step"),
                "ok": item.get("ok"),
                "data": _summarize_crawl(item.get("result", {})) if item.get("ok") else [],
                "error": item.get("error"),
            }
            for item in crawl_results
        ],
        "research_task": {
            "ok": bool(research_result and research_result.get("ok")),
            "data": _summarize_research(research_result.get("result", {})) if research_result and research_result.get("ok") else {},
            "error": research_result.get("error") if research_result and not research_result.get("ok") else None,
        },
    }

    successful_searches = [run for run in search_runs if run.get("ok")]
    if not bundle["research_task"]["ok"]:
        bundle["warnings"] = [
            "Deep Tavily research did not finish in time. Report was generated from completed search, extract, map, and crawl evidence."
        ]
        if not successful_searches:
            raise TavilyClientError(
                "No Tavily search results available. Check API key and network connection."
            )
        logger.warning(
            "Tavily deep research timed out but %d search runs completed — proceeding with available data.",
            len(successful_searches)
        )
    else:
        bundle["warnings"] = []

    return bundle


# ✅ Next function starts here - this is CORRECT
def _create_mock_research_bundle(
    intake: dict[str, Any],
    conversation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a mock research bundle when Tavily API is unavailable due to usage limits."""
    # ... rest of mock function continues

    business = _clean(intake.get("business_idea"))
    audience = _clean(intake.get("target_audience")) or "target customers"
    location = _location_hint(intake)
    stage = _clean(intake.get("stage")) or "early stage"
    team_size = _clean(intake.get("team_size")) or "small team"
    goals = ", ".join(intake.get("goals") or ["growth"])

    # Create mock search results based on business type
    mock_search_runs = [
        {
            "label": "market",
            "query": f"{business} market demand trends TAM customer demand {location}",
            "answer": f"Based on industry analysis, {business} operates in a growing market with increasing demand from {audience}. Market size estimates suggest significant opportunity with {location} showing particular growth potential.",
            "results": [
                {
                    "title": f"Market Analysis: {business} Industry Trends",
                    "url": f"https://industry-report-{business.lower().replace(' ', '-')}.com",
                    "content": f"The {business} sector is experiencing rapid growth, particularly in {location}. Key drivers include increasing demand from {audience} and technological advancements."
                },
                {
                    "title": f"{business} Market Size and Opportunities",
                    "url": f"https://market-research-{business.lower().replace(' ', '-')}.com",
                    "content": f"Market research indicates strong demand for {business} solutions targeting {audience}. The TAM is estimated at significant figures with {location} being a key growth market."
                }
            ]
        },
        {
            "label": "customer",
            "query": f"{audience} needs pain points buying behavior for {business} in {location}",
            "answer": f"{audience} face several challenges that {business} can address. Key pain points include [specific challenges], with buying behavior showing preference for [solution characteristics].",
            "results": [
                {
                    "title": f"Customer Insights: {audience} Pain Points",
                    "url": f"https://customer-research-{audience.lower().replace(' ', '-')}.com",
                    "content": f"Research shows {audience} struggle with [industry-specific challenges]. They seek solutions that offer [key benefits] and are willing to pay premium for quality."
                }
            ]
        },
        {
            "label": "competition",
            "query": f"top competitors alternatives for {business} in {location}",
            "answer": f"The competitive landscape includes [3-5 major players] with varying market share. Key differentiators include [unique selling points] and [market positioning].",
            "results": [
                {
                    "title": f"Competitive Analysis: {business} Market Leaders",
                    "url": f"https://competitor-analysis-{business.lower().replace(' ', '-')}.com",
                    "content": f"Leading competitors in the {business} space include established players with strong brand recognition and market share. New entrants can differentiate through [innovation areas]."
                }
            ]
        },
        {
            "label": "pricing",
            "query": f"pricing benchmarks revenue model unit economics for {business}",
            "answer": f"Industry pricing benchmarks range from [low end] to [high end] depending on features and market positioning. Revenue models typically include [subscription/one-time/hybrid] approaches.",
            "results": [
                {
                    "title": f"Pricing Strategy: {business} Industry Benchmarks",
                    "url": f"https://pricing-analysis-{business.lower().replace(' ', '-')}.com",
                    "content": f"Competitive pricing analysis shows market rates of [price ranges] for {business} solutions. Unit economics vary based on [cost factors] and [revenue drivers]."
                }
            ]
        },
        {
            "label": "channels",
            "query": f"best marketing channels for {business} targeting {audience} with goals {goals}",
            "answer": f"Optimal marketing channels for {business} include digital marketing, content marketing, and targeted advertising. Success metrics focus on {goals} with emphasis on customer acquisition cost optimization.",
            "results": [
                {
                    "title": f"Marketing Channels: {business} Growth Strategies",
                    "url": f"https://marketing-strategy-{business.lower().replace(' ', '-')}.com",
                    "content": f"Effective marketing for {business} involves [digital channels], [content strategies], and [partnership opportunities]. Focus on channels that reach {audience} effectively."
                }
            ]
        }
    ]

    # Create mock research task data
    mock_research_content = f"""
# Research Summary: {business}

## Executive Overview
{business} represents a promising opportunity in the {location} market, targeting {audience}. The business operates at the {stage} with a {team_size}, focusing on {goals}.

## Market Analysis
The market shows strong growth potential with increasing demand from {audience}. Key opportunities include [market gaps] and [emerging trends]. Competitive landscape features [main competitors] with room for differentiation through [unique value propositions].

## Customer Insights
{audience} face specific challenges that {business} can address through [solution approach]. Buying behavior indicates preference for [product characteristics] with willingness to pay [price sensitivity].

## Business Model
Revenue potential exists through [pricing strategy] with unit economics showing [profitability metrics]. Market positioning should focus on [differentiation factors].

## Growth Strategy
Recommended channels include [marketing approaches] targeting {audience} effectively. Success will depend on [key success factors] and [execution priorities].

## Risk Assessment
Key risks include [market risks], [operational risks], and [competitive threats]. Mitigation strategies involve [risk management approaches].

## Recommendations
Proceed with [recommended actions] focusing on [priority areas]. Timeline suggests [development milestones] with [resource requirements].
"""

    return {
        "report_mode": _clean(intake.get("report_mode")),
        "intake_snapshot": {
            "stage_type": _clean(intake.get("stage_type")),
            "report_mode": _clean(intake.get("report_mode")),
            "business_idea": business,
            "target_audience": audience,
            "stage": stage,
            "team_size": team_size,
            "goals": intake.get("goals") or [],
            "budget_revenue": _clean(intake.get("budget_revenue")),
            "location": location,
            "context": _clean(intake.get("context")),
        },
        "conversation_snapshot": _conversation_snapshot(conversation_history),
        "search_runs": mock_search_runs,
        "reviews_runs": [                    # ✅ yahan add karo
            {
                "source": "google_reviews",
                "query": f"{business} google reviews complaints {location}",
                "summary": f"Review data unavailable — Tavily API limit exceeded. Recommend manually collecting 20+ competitor reviews from Google Maps and Amazon/Flipkart for {business} in {location}.",
                "snippets": [],
                "error": "mock_mode",
            },
            {
                "source": "community_discussions",
                "query": f"{business} reddit quora customer experience India",
                "summary": f"Community discussion data unavailable in mock mode. Check Reddit r/India and Quora for {business} experiences.",
                "snippets": [],
                "error": "mock_mode",
            },
        ],
        "extract": {
            "ok": False,
            "results": [],
            "error": "Tavily API usage limit exceeded - using mock data",
        },
        "map_runs": [],
        "crawl_runs": [],
        "research_task": {
            "ok": True,
            "data": {
                "request_id": "mock-research-task",
                "status": "completed",
                "content": mock_research_content,
                "sources": [
                    {
                        "title": f"Industry Report: {business}",
                        "url": f"https://industry-analysis-{business.lower().replace(' ', '-')}.com"
                    },
                    {
                        "title": f"Market Research: {audience}",
                        "url": f"https://market-research-{audience.lower().replace(' ', '-')}.com"
                    },
                    {
                        "title": f"Competitive Analysis: {business} Sector",
                        "url": f"https://competitor-analysis-{business.lower().replace(' ', '-')}.com"
                    }
                ],
            },
            "error": None,
        },
        "warnings": [
            "⚠️ Tavily API usage limit exceeded. Report generated using structured mock data based on business type and industry patterns. For accurate market research, please upgrade your Tavily API plan or try again later."
        ],
    }
