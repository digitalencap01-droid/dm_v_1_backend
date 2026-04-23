"""
LLM Prompts — all prompt templates for the Profile Intelligence Engine.

Prompt text lives EXCLUSIVELY in this module.
No service or route file may contain raw prompt strings.

Each function returns a fully-rendered string ready to pass to LLMClient.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompts (used as the "system" turn)
# ---------------------------------------------------------------------------

SYSTEM_EXTRACTION = (
    "You are a business analyst AI. Your job is to extract structured "
    "information from a free-form business description. "
    "Respond ONLY with valid JSON matching the schema provided. "
    "Do not add explanations or markdown fences."
)

SYSTEM_CLASSIFICATION = (
    "You are a business persona and industry classifier. "
    "Respond ONLY with valid JSON matching the schema provided. "
    "Be concise and deterministic."
)

SYSTEM_READINESS = (
    "You are a startup and business readiness evaluator. "
    "Assess the maturity stage of a business based on the provided context. "
    "Respond ONLY with valid JSON matching the schema provided."
)

SYSTEM_NEED_ROUTING = (
    "You are a digital marketing strategist. "
    "Identify the primary business need and secondary needs for a business. "
    "Respond ONLY with valid JSON matching the schema provided."
)

SYSTEM_PROFILE_BUILDER = (
    "You are a senior marketing consultant building a comprehensive business profile. "
    "Synthesise all available information into actionable insights. "
    "Respond ONLY with valid JSON matching the schema provided."
)

SYSTEM_REQUIRED_QUESTION_GENERATOR = (
    "You generate a single best next REQUIRED follow-up question. "
    "Return ONLY valid JSON with the requested keys. "
    "Do not add explanations outside the JSON."
)

SYSTEM_SLOT_FILLER = (
    "You extract a concise structured value from a user's answer. "
    "Return ONLY valid JSON with the requested keys."
)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

def build_extraction_prompt(raw_input: str) -> str:
    return f"""Extract structured business information from the following description.

Business Description:
\"\"\"
{raw_input}
\"\"\"

Return a JSON object with exactly these fields:
{{
  "business_name": "<string or null>",
  "description": "<concise 1-2 sentence description>",
  "target_audience": "<who they serve, or null>",
  "product_or_service": "<what they sell, or null>",
  "revenue_model": "<how they make money, or null>",
  "current_challenges": ["<challenge 1>", "<challenge 2>"],
  "mentioned_goals": ["<goal 1>", "<goal 2>"],
  "mentioned_channels": ["<channel 1>"],
  "raw_persona_hint": "<any hint about who the operator is, or null>",
  "raw_industry_hint": "<industry hint from text, or null>",
  "raw_readiness_hint": "<stage hint from text, or null>",
  "confidence_hints": {{
    "persona": <0.0-1.0>,
    "industry": <0.0-1.0>,
    "readiness": <0.0-1.0>
  }}
}}

Rules:
- Use null for fields not present in the text.
- Keep arrays empty ([]) if nothing is mentioned.
- confidence_hints values reflect how clearly the text signals that dimension.
"""


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

def build_classification_prompt(
    description: str,
    target_audience: str | None,
    product_or_service: str | None,
    raw_persona_hint: str | None,
    raw_industry_hint: str | None,
) -> str:
    return f"""Classify the following business into a persona type and industry.

Business description: {description}
Target audience: {target_audience or 'not specified'}
Product/service: {product_or_service or 'not specified'}
Persona hint from text: {raw_persona_hint or 'none'}
Industry hint from text: {raw_industry_hint or 'none'}

Valid persona values: founder, marketer, sales_lead, agency, enterprise, smb, unknown
Valid industry values: saas, ecommerce, professional_services, healthcare, education, real_estate, finance, other

Return a JSON object:
{{
  "persona": "<one of the valid persona values>",
  "industry": "<one of the valid industry values>",
  "persona_confidence": <0.0-1.0>,
  "industry_confidence": <0.0-1.0>,
  "raw_persona": "<short phrase describing persona evidence>",
  "raw_industry": "<short phrase describing industry evidence>"
}}
"""


# ---------------------------------------------------------------------------
# Readiness assessment prompt
# ---------------------------------------------------------------------------

def build_readiness_prompt(
    description: str,
    challenges: list[str],
    goals: list[str],
    raw_readiness_hint: str | None,
    answers: dict[str, str],
) -> str:
    challenge_text = "; ".join(challenges) if challenges else "none mentioned"
    goals_text = "; ".join(goals) if goals else "none mentioned"
    answer_text = (
        "\n".join(f"- {k}: {v}" for k, v in answers.items()) if answers else "none"
    )

    return f"""Assess the readiness/maturity stage of this business.

Business description: {description}
Current challenges: {challenge_text}
Goals: {goals_text}
Stage hint from text: {raw_readiness_hint or 'none'}
Follow-up answers provided:
{answer_text}

Valid readiness levels: idea_stage, mvp, early_traction, scaling, mature, unknown

Return a JSON object:
{{
  "readiness_level": "<one of the valid readiness levels>",
  "readiness_confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentence explanation>",
  "raw_readiness": "<key evidence phrase>"
}}
"""


# ---------------------------------------------------------------------------
# Need-routing prompt
# ---------------------------------------------------------------------------

def build_need_routing_prompt(
    description: str,
    persona: str,
    industry: str,
    readiness_level: str,
    challenges: list[str],
    goals: list[str],
    answers: dict[str, str],
) -> str:
    challenge_text = "; ".join(challenges) if challenges else "none"
    goals_text = "; ".join(goals) if goals else "none"
    answer_text = (
        "\n".join(f"- {k}: {v}" for k, v in answers.items()) if answers else "none"
    )

    return f"""Identify the primary marketing/business need for this profile.

Persona: {persona}
Industry: {industry}
Readiness level: {readiness_level}
Business description: {description}
Challenges: {challenge_text}
Goals: {goals_text}
Additional answers:
{answer_text}

Valid need states:
- brand_awareness
- lead_generation
- customer_retention
- revenue_growth
- product_launch
- market_expansion
- unknown

Return a JSON object:
{{
  "primary_need": "<one valid need state>",
  "secondary_needs": ["<need state>", "<need state>"],
  "need_confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentence explanation>"
}}

secondary_needs can be empty [].
"""


# ---------------------------------------------------------------------------
# Profile builder prompt
# ---------------------------------------------------------------------------

def build_profile_prompt(
    session_id: str,
    persona: str,
    industry: str,
    readiness_level: str,
    primary_need: str,
    secondary_needs: list[str],
    extracted: dict,
    answers: dict[str, str],
    confidence_score: float,
) -> str:
    answer_text = (
        "\n".join(f"- {k}: {v}" for k, v in answers.items()) if answers else "none"
    )
    secondary_text = ", ".join(secondary_needs) if secondary_needs else "none"

    return f"""Build a comprehensive marketing profile for this business.

Session ID: {session_id}
Persona: {persona}
Industry: {industry}
Readiness level: {readiness_level}
Primary need: {primary_need}
Secondary needs: {secondary_text}

Extracted data:
- Business name: {extracted.get('business_name') or 'unknown'}
- Description: {extracted.get('description') or 'N/A'}
- Target audience: {extracted.get('target_audience') or 'unknown'}
- Product/service: {extracted.get('product_or_service') or 'unknown'}
- Revenue model: {extracted.get('revenue_model') or 'unknown'}
- Challenges: {'; '.join(extracted.get('current_challenges', [])) or 'none'}
- Goals: {'; '.join(extracted.get('mentioned_goals', [])) or 'none'}
- Channels mentioned: {'; '.join(extracted.get('mentioned_channels', [])) or 'none'}

Follow-up answers:
{answer_text}

Current confidence score: {confidence_score:.2f}

Return a JSON object:
{{
  "business_name": "<string or null>",
  "target_audience": "<refined description>",
  "product_or_service": "<clear description>",
  "revenue_model": "<description or null>",
  "current_challenges": ["<challenge>"],
  "goals": ["<goal>"],
  "recommended_channels": ["<channel 1>", "<channel 2>", "<channel 3>"],
  "summary": "<3-4 sentence actionable profile summary>"
}}
"""


# ---------------------------------------------------------------------------
# Question answer extraction helper
# ---------------------------------------------------------------------------

def build_answer_extraction_prompt(question_key: str, question_text: str, answer: str) -> str:
    """Extract a normalised fact from a follow-up answer."""
    return f"""A user answered a profile question. Extract the core fact.

Question key: {question_key}
Question: {question_text}
User answer: {answer}

Return a JSON object:
{{
  "extracted_fact": "<concise extracted fact, 1 sentence>",
  "confidence_contribution": <0.0-1.0>
}}
"""


# ---------------------------------------------------------------------------
# Dynamic required questions (slot-based)
# ---------------------------------------------------------------------------


def build_required_question_prompt(
    slot: str,
    raw_input: str,
    extracted: dict,
    baseline: dict,
) -> str:
    """
    Ask the LLM to generate the best next required question for a slot.
    """
    return f"""Generate ONE required follow-up question to fill the missing research slot.\n\nMissing slot: {slot}\n\nUser baseline (structured):\n{baseline}\n\nBusiness description:\n\"\"\"\n{raw_input}\n\"\"\"\n\nExtracted signals (may be incomplete):\n{extracted}\n\nReturn JSON:\n{{\n  \"slot\": \"offer|icp\",\n  \"question\": \"<a single question, concise>\",\n  \"question_key\": \"<a stable key like req_offer_1 or req_icp_1>\",\n  \"reason\": \"<short reason why this is required>\"\n}}\n\nRules:\n- Question must be answerable in 1–2 sentences.\n- If slot=offer: ask what they sell / what the product is.\n- If slot=icp: ask who the ideal customer is (role + industry + size).\n- Do not ask about website unless user already mentioned they have one.\n"""


def build_slot_fill_prompt(
    slot: str,
    question: str,
    answer: str,
) -> str:
    return f"""Extract the slot value from the user's answer.\n\nSlot: {slot}\nQuestion: {question}\nAnswer: {answer}\n\nReturn JSON:\n{{\n  \"slot\": \"offer|icp\",\n  \"value\": \"<one-line slot value>\",\n  \"confidence\": <0.0-1.0>\n}}\n\nRules:\n- Keep value short and specific.\n- If slot=offer: return a one-liner describing what they sell.\n- If slot=icp: return a one-liner describing the ideal customer.\n"""

# ---------------------------------------------------------------------------
# Follow-up question bank (static — question_selector uses these)
# ---------------------------------------------------------------------------

QUESTION_BANK: list[dict] = [
    {
        "key": "declared_stage",
        "text": "What stage are you at right now?",
        "type": "required",
        "context": "Baseline: helps calibrate readiness (idea_stage / mvp / early_traction / scaling / mature / not_sure).",
    },
    {
        "key": "declared_goals",
        "text": "What are your goals? (Select all that apply)",
        "type": "required",
        "context": "Baseline: multi-select goals (brand_awareness, lead_generation, customer_retention, revenue_growth, product_launch, market_expansion, other).",
    },
    {
        "key": "website_url",
        "text": "What is your website URL (or best link to learn about your business)?",
        "type": "required",
        "context": "Research-ready identifier. Improves research accuracy significantly.",
    },
    {
        "key": "product_or_service",
        "text": "In one sentence, what product or service do you sell?",
        "type": "required",
        "context": "Clarifies offer when the raw input is ambiguous.",
    },
    {
        "key": "monthly_revenue",
        "text": "What is your approximate monthly revenue or transaction volume?",
        "type": "required",
        "context": "Helps determine readiness level and growth stage.",
    },
    {
        "key": "team_size",
        "text": "How large is your team currently?",
        "type": "required",
        "context": "Indicates organisational maturity.",
    },
    {
        "key": "primary_channel",
        "text": "What is your primary customer acquisition channel today?",
        "type": "required",
        "context": "Reveals current marketing maturity.",
    },
    {
        "key": "biggest_bottleneck",
        "text": "What is the single biggest bottleneck to your growth right now?",
        "type": "required",
        "context": "Directly maps to need state.",
    },
    {
        "key": "time_in_market",
        "text": "How long has your business been operating?",
        "type": "optional",
        "context": "Supports readiness assessment.",
    },
    {
        "key": "target_market_geo",
        "text": "Which geographic markets are you targeting?",
        "type": "optional",
        "context": "Helps with channel and expansion recommendations.",
    },
    {
        "key": "budget_range",
        "text": "What is your approximate marketing budget per month?",
        "type": "optional",
        "context": "Guides channel recommendations.",
    },
]
