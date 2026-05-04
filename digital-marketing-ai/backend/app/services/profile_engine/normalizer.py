"""
Normalizer — converts raw LLM string outputs into typed enums.

Pure functions with no side effects.  Called by extractor, classifier,
readiness, and need_routing after every LLM response to ensure
all downstream services receive valid enum values.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import (
    IndustryType,
    NeedState,
    PersonaType,
    ReadinessLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation maps
# ---------------------------------------------------------------------------

_PERSONA_MAP: dict[str, PersonaType] = {
    "founder": PersonaType.FOUNDER,
    "co-founder": PersonaType.FOUNDER,
    "co founder": PersonaType.FOUNDER,
    "ceo": PersonaType.FOUNDER,
    "owner": PersonaType.FOUNDER,
    "business owner": PersonaType.FOUNDER,
    "entrepreneur": PersonaType.FOUNDER,
    "startup founder": PersonaType.FOUNDER,
    "marketer": PersonaType.MARKETER,
    "marketing": PersonaType.MARKETER,
    "marketing manager": PersonaType.MARKETER,
    "marketing director": PersonaType.MARKETER,
    "cmo": PersonaType.MARKETER,
    "vp marketing": PersonaType.MARKETER,
    "head of marketing": PersonaType.MARKETER,
    "sales": PersonaType.SALES_LEAD,
    "sales manager": PersonaType.SALES_LEAD,
    "sales lead": PersonaType.SALES_LEAD,
    "sales_lead": PersonaType.SALES_LEAD,
    "vp sales": PersonaType.SALES_LEAD,
    "sales director": PersonaType.SALES_LEAD,
    "head of sales": PersonaType.SALES_LEAD,
    "sales representative": PersonaType.SALES_LEAD,
    "account executive": PersonaType.SALES_LEAD,
    "business development": PersonaType.SALES_LEAD,
    "agency": PersonaType.AGENCY,
    "digital agency": PersonaType.AGENCY,
    "marketing agency": PersonaType.AGENCY,
    "consultant": PersonaType.AGENCY,
    "consulting": PersonaType.AGENCY,
    "consultancy": PersonaType.AGENCY,
    "freelancer": PersonaType.AGENCY,
    "contractor": PersonaType.AGENCY,
    "service provider": PersonaType.AGENCY,
    "enterprise": PersonaType.ENTERPRISE,
    "large company": PersonaType.ENTERPRISE,
    "large enterprise": PersonaType.ENTERPRISE,
    "corporation": PersonaType.ENTERPRISE,
    "corporate": PersonaType.ENTERPRISE,
    "big company": PersonaType.ENTERPRISE,
    "multinational": PersonaType.ENTERPRISE,
    "smb": PersonaType.SMB,
    "small business": PersonaType.SMB,
    "small and medium": PersonaType.SMB,
    "medium business": PersonaType.SMB,
    "sme": PersonaType.SMB,
    "startup": PersonaType.SMB,
    "growing company": PersonaType.SMB,
}

_INDUSTRY_MAP: dict[str, IndustryType] = {
    "saas": IndustryType.SAAS,
    "software": IndustryType.SAAS,
    "software as a service": IndustryType.SAAS,
    "software-as-a-service": IndustryType.SAAS,
    "tech": IndustryType.SAAS,
    "technology": IndustryType.SAAS,
    "technology company": IndustryType.SAAS,
    "software company": IndustryType.SAAS,
    "software platform": IndustryType.SAAS,
    "digital product": IndustryType.SAAS,
    "b2b software": IndustryType.SAAS,
    "b2c app": IndustryType.SAAS,
    "mobile app": IndustryType.SAAS,
    "ecommerce": IndustryType.ECOMMERCE,
    "e-commerce": IndustryType.ECOMMERCE,
    "online store": IndustryType.ECOMMERCE,
    "online retail": IndustryType.ECOMMERCE,
    "e-commerce business": IndustryType.ECOMMERCE,
    "retail": IndustryType.ECOMMERCE,
    "retail business": IndustryType.ECOMMERCE,
    "online shop": IndustryType.ECOMMERCE,
    "marketplace": IndustryType.ECOMMERCE,
    "seller": IndustryType.ECOMMERCE,
    "dropshipping": IndustryType.ECOMMERCE,
    "professional_services": IndustryType.PROFESSIONAL_SERVICES,
    "professional services": IndustryType.PROFESSIONAL_SERVICES,
    "professional services firm": IndustryType.PROFESSIONAL_SERVICES,
    "consulting": IndustryType.PROFESSIONAL_SERVICES,
    "consulting firm": IndustryType.PROFESSIONAL_SERVICES,
    "law": IndustryType.PROFESSIONAL_SERVICES,
    "law firm": IndustryType.PROFESSIONAL_SERVICES,
    "legal": IndustryType.PROFESSIONAL_SERVICES,
    "accounting": IndustryType.PROFESSIONAL_SERVICES,
    "accounting firm": IndustryType.PROFESSIONAL_SERVICES,
    "accountant": IndustryType.PROFESSIONAL_SERVICES,
    "tax services": IndustryType.PROFESSIONAL_SERVICES,
    "audit": IndustryType.PROFESSIONAL_SERVICES,
    "healthcare": IndustryType.HEALTHCARE,
    "health": IndustryType.HEALTHCARE,
    "health tech": IndustryType.HEALTHCARE,
    "healthcare provider": IndustryType.HEALTHCARE,
    "medical": IndustryType.HEALTHCARE,
    "medical device": IndustryType.HEALTHCARE,
    "telemedicine": IndustryType.HEALTHCARE,
    "pharmacy": IndustryType.HEALTHCARE,
    "hospital": IndustryType.HEALTHCARE,
    "clinic": IndustryType.HEALTHCARE,
    "fitness": IndustryType.HEALTHCARE,
    "wellness": IndustryType.HEALTHCARE,
    "education": IndustryType.EDUCATION,
    "edtech": IndustryType.EDUCATION,
    "education tech": IndustryType.EDUCATION,
    "e-learning": IndustryType.EDUCATION,
    "elearning": IndustryType.EDUCATION,
    "online course": IndustryType.EDUCATION,
    "training": IndustryType.EDUCATION,
    "school": IndustryType.EDUCATION,
    "university": IndustryType.EDUCATION,
    "tutoring": IndustryType.EDUCATION,
    "learning platform": IndustryType.EDUCATION,
    "real_estate": IndustryType.REAL_ESTATE,
    "real estate": IndustryType.REAL_ESTATE,
    "property": IndustryType.REAL_ESTATE,
    "property management": IndustryType.REAL_ESTATE,
    "real estate agent": IndustryType.REAL_ESTATE,
    "real estate broker": IndustryType.REAL_ESTATE,
    "real estate company": IndustryType.REAL_ESTATE,
    "proptech": IndustryType.REAL_ESTATE,
    "real estate tech": IndustryType.REAL_ESTATE,
    "housing": IndustryType.REAL_ESTATE,
    "real estate market": IndustryType.REAL_ESTATE,
    "finance": IndustryType.FINANCE,
    "fintech": IndustryType.FINANCE,
    "financial": IndustryType.FINANCE,
    "financial services": IndustryType.FINANCE,
    "banking": IndustryType.FINANCE,
    "bank": IndustryType.FINANCE,
    "insurance": IndustryType.FINANCE,
    "investment": IndustryType.FINANCE,
    "wealth management": IndustryType.FINANCE,
    "payment": IndustryType.FINANCE,
    "payment processing": IndustryType.FINANCE,
    "lending": IndustryType.FINANCE,
    "loan": IndustryType.FINANCE,
    "cryptocurrency": IndustryType.FINANCE,
    "crypto": IndustryType.FINANCE,
    "blockchain": IndustryType.FINANCE,
}

_READINESS_MAP: dict[str, ReadinessLevel] = {
    "idea_stage": ReadinessLevel.IDEA_STAGE,
    "idea stage": ReadinessLevel.IDEA_STAGE,
    "idea": ReadinessLevel.IDEA_STAGE,
    "pre-launch": ReadinessLevel.IDEA_STAGE,
    "mvp": ReadinessLevel.MVP,
    "minimum viable product": ReadinessLevel.MVP,
    "beta": ReadinessLevel.MVP,
    "early_traction": ReadinessLevel.EARLY_TRACTION,
    "early traction": ReadinessLevel.EARLY_TRACTION,
    "early stage": ReadinessLevel.EARLY_TRACTION,
    "growth": ReadinessLevel.EARLY_TRACTION,
    "scaling": ReadinessLevel.SCALING,
    "scale": ReadinessLevel.SCALING,
    "scale-up": ReadinessLevel.SCALING,
    "growing fast": ReadinessLevel.SCALING,
    "mature": ReadinessLevel.MATURE,
    "established": ReadinessLevel.MATURE,
    "enterprise": ReadinessLevel.MATURE,
    "market leader": ReadinessLevel.MATURE,
}

_NEED_MAP: dict[str, NeedState] = {
    "brand_awareness": NeedState.BRAND_AWARENESS,
    "brand awareness": NeedState.BRAND_AWARENESS,
    "awareness": NeedState.BRAND_AWARENESS,
    "visibility": NeedState.BRAND_AWARENESS,
    "lead_generation": NeedState.LEAD_GENERATION,
    "lead generation": NeedState.LEAD_GENERATION,
    "leads": NeedState.LEAD_GENERATION,
    "prospecting": NeedState.LEAD_GENERATION,
    "customer_retention": NeedState.CUSTOMER_RETENTION,
    "customer retention": NeedState.CUSTOMER_RETENTION,
    "retention": NeedState.CUSTOMER_RETENTION,
    "churn": NeedState.CUSTOMER_RETENTION,
    "revenue_growth": NeedState.REVENUE_GROWTH,
    "revenue growth": NeedState.REVENUE_GROWTH,
    "revenue": NeedState.REVENUE_GROWTH,
    "sales growth": NeedState.REVENUE_GROWTH,
    "product_launch": NeedState.PRODUCT_LAUNCH,
    "product launch": NeedState.PRODUCT_LAUNCH,
    "launch": NeedState.PRODUCT_LAUNCH,
    "go to market": NeedState.PRODUCT_LAUNCH,
    "gtm": NeedState.PRODUCT_LAUNCH,
    "market_expansion": NeedState.MARKET_EXPANSION,
    "market expansion": NeedState.MARKET_EXPANSION,
    "expansion": NeedState.MARKET_EXPANSION,
    "new market": NeedState.MARKET_EXPANSION,
    "international": NeedState.MARKET_EXPANSION,
}


# ---------------------------------------------------------------------------
# Public normalisation functions
# ---------------------------------------------------------------------------


def normalize_persona(raw: str | None) -> PersonaType:
    """Map a raw string to a PersonaType enum value."""
    if not raw:
        return PersonaType.UNKNOWN
    key = raw.strip().lower()
    result = _PERSONA_MAP.get(key)
    if result:
        return result
    # Partial match fallback
    for token, value in _PERSONA_MAP.items():
        if token in key or key in token:
            return value
    logger.debug("normalize_persona: no match for %r, defaulting UNKNOWN", raw)
    return PersonaType.UNKNOWN


def normalize_industry(raw: str | None) -> IndustryType:
    """Map a raw string to an IndustryType enum value."""
    if not raw:
        return IndustryType.OTHER
    key = raw.strip().lower()
    result = _INDUSTRY_MAP.get(key)
    if result:
        return result
    for token, value in _INDUSTRY_MAP.items():
        if token in key or key in token:
            return value
    logger.debug("normalize_industry: no match for %r, defaulting OTHER", raw)
    return IndustryType.OTHER


def normalize_readiness(raw: str | None) -> ReadinessLevel:
    """Map a raw string to a ReadinessLevel enum value."""
    if not raw:
        return ReadinessLevel.UNKNOWN
    key = raw.strip().lower()
    result = _READINESS_MAP.get(key)
    if result:
        return result
    for token, value in _READINESS_MAP.items():
        if token in key or key in token:
            return value
    logger.debug("normalize_readiness: no match for %r, defaulting UNKNOWN", raw)
    return ReadinessLevel.UNKNOWN


def normalize_need_state(raw: str | None) -> NeedState:
    """Map a raw string to a NeedState enum value."""
    if not raw:
        return NeedState.UNKNOWN
    key = raw.strip().lower()
    result = _NEED_MAP.get(key)
    if result:
        return result
    for token, value in _NEED_MAP.items():
        if token in key or key in token:
            return value
    logger.debug("normalize_need_state: no match for %r, defaulting UNKNOWN", raw)
    return NeedState.UNKNOWN


def normalize_need_list(raw_list: list[str]) -> list[NeedState]:
    """Normalize a list of raw need strings, skipping UNKNOWN results."""
    results: list[NeedState] = []
    for raw in raw_list:
        need = normalize_need_state(raw)
        if need != NeedState.UNKNOWN:
            results.append(need)
    return results
