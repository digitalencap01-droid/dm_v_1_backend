"""
Smoke tests for the Profile Intelligence Engine.

Tests are structured into three groups:

1. Unit tests — schemas, normalizer, confidence, question_selector
   (no LLM, no DB required)

2. Orchestrator integration tests — full pipeline with a mocked LLM client
   (no DB required)

3. Route-level tests — FastAPI TestClient with mocked repo + orchestrator
   (no real DB or LLM)

Run with:
    pytest app/tests/profile_engine/ -v
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.profile_engine import (
    ClassifiedProfile,
    ExtractedData,
    FinalProfile,
    IndustryType,
    NeedRouting,
    NeedState,
    PersonaType,
    Question,
    QuestionOption,
    QuestionType,
    ReadinessAssessment,
    ReadinessLevel,
    ResearchSlot,
    SessionState,
    SessionStatus,
)
from app.services.profile_engine.confidence import calculate_confidence
from app.services.profile_engine.normalizer import (
    normalize_industry,
    normalize_need_state,
    normalize_persona,
    normalize_readiness,
)
from app.services.profile_engine.question_selector import select_question
from app.services.profile_engine import dynamic_required
from app.services.profile_engine.orchestrator import _parse_monthly_revenue_to_int


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def base_extracted() -> ExtractedData:
    return ExtractedData(
        business_name="AcmeSaaS",
        description="B2B SaaS platform for marketing automation for SMBs.",
        target_audience="Small and medium-sized businesses",
        product_or_service="Marketing automation software",
        revenue_model="Monthly subscription",
        current_challenges=["low brand awareness", "high churn"],
        mentioned_goals=["double MRR", "reduce churn by 20%"],
        mentioned_channels=["LinkedIn", "cold email"],
        raw_persona_hint="founder",
        raw_industry_hint="saas",
        raw_readiness_hint="early traction",
        confidence_hints={"persona": 0.8, "industry": 0.9, "readiness": 0.7},
    )


@pytest.fixture()
def base_classified() -> ClassifiedProfile:
    return ClassifiedProfile(
        persona=PersonaType.FOUNDER,
        industry=IndustryType.SAAS,
        persona_confidence=0.85,
        industry_confidence=0.90,
    )


@pytest.fixture()
def base_readiness() -> ReadinessAssessment:
    return ReadinessAssessment(
        readiness_level=ReadinessLevel.EARLY_TRACTION,
        readiness_confidence=0.75,
        reasoning="Has paying customers and some traction.",
    )


@pytest.fixture()
def base_need_routing() -> NeedRouting:
    return NeedRouting(
        primary_need=NeedState.LEAD_GENERATION,
        secondary_needs=[NeedState.CUSTOMER_RETENTION],
        need_confidence=0.80,
    )


@pytest.fixture()
def full_state(
    base_extracted,
    base_classified,
    base_readiness,
    base_need_routing,
) -> SessionState:
    sid = uuid.uuid4()
    state = SessionState(
        session_id=sid,
        status=SessionStatus.COLLECTING_INPUT,
        raw_input="We are a B2B SaaS company targeting SMBs...",
        extracted=base_extracted,
        classified=base_classified,
        readiness=base_readiness,
        need_routing=base_need_routing,
        answers={},
        asked_questions=[],
        confidence_score=0.0,
    )
    from app.services.profile_engine.confidence import calculate_confidence
    state.confidence_score = calculate_confidence(state)
    return state


# ===========================================================================
# 1. Normalizer unit tests
# ===========================================================================


class TestNormalizer:
    def test_persona_exact_match(self):
        assert normalize_persona("founder") == PersonaType.FOUNDER

    def test_persona_case_insensitive(self):
        assert normalize_persona("Founder") == PersonaType.FOUNDER

    def test_persona_partial_match(self):
        assert normalize_persona("the founder of xyz") == PersonaType.FOUNDER

    def test_persona_unknown_default(self):
        assert normalize_persona("undefined_role") == PersonaType.UNKNOWN

    def test_persona_none(self):
        assert normalize_persona(None) == PersonaType.UNKNOWN

    def test_industry_saas(self):
        assert normalize_industry("saas") == IndustryType.SAAS

    def test_industry_ecommerce_hyphen(self):
        assert normalize_industry("e-commerce") == IndustryType.ECOMMERCE

    def test_industry_partial_software(self):
        assert normalize_industry("software platform") == IndustryType.SAAS

    def test_industry_none(self):
        assert normalize_industry(None) == IndustryType.OTHER

    def test_readiness_mvp(self):
        assert normalize_readiness("mvp") == ReadinessLevel.MVP

    def test_readiness_early_traction(self):
        assert normalize_readiness("early traction") == ReadinessLevel.EARLY_TRACTION

    def test_readiness_unknown(self):
        assert normalize_readiness("some random text") == ReadinessLevel.UNKNOWN

    def test_need_lead_generation(self):
        assert normalize_need_state("lead generation") == NeedState.LEAD_GENERATION

    def test_need_brand_awareness(self):
        assert normalize_need_state("brand_awareness") == NeedState.BRAND_AWARENESS

    def test_need_unknown(self):
        assert normalize_need_state("xyz_need") == NeedState.UNKNOWN


# ===========================================================================
# 2. Confidence unit tests
# ===========================================================================


class TestConfidence:
    def test_zero_with_no_data(self):
        state = SessionState(session_id=uuid.uuid4())
        score = calculate_confidence(state)
        assert score == 0.0

    def test_increases_with_classified(self, base_classified):
        state = SessionState(
            session_id=uuid.uuid4(),
            classified=base_classified,
        )
        score = calculate_confidence(state)
        assert score > 0.0

    def test_increases_with_all_dimensions(self, full_state):
        score = calculate_confidence(full_state)
        assert 0.5 < score <= 1.0

    def test_answer_bonus(self, full_state):
        score_no_answers = calculate_confidence(full_state)
        full_state.answers = {
            "monthly_revenue": "$10k",
            "team_size": "3 people",
        }
        score_with_answers = calculate_confidence(full_state)
        assert score_with_answers > score_no_answers

    def test_unknown_persona_reduces_score(self):
        classified = ClassifiedProfile(
            persona=PersonaType.UNKNOWN,
            industry=IndustryType.SAAS,
            persona_confidence=0.9,
            industry_confidence=0.9,
        )
        state = SessionState(session_id=uuid.uuid4(), classified=classified)
        score = calculate_confidence(state)
        # UNKNOWN persona reduces the persona sub-score by 70%
        assert score < 0.3  # industry accounts for ~0.18 at most at this point


# ===========================================================================
# 3. Question selector unit tests
# ===========================================================================


class TestQuestionSelector:
    def test_first_question_is_required(self, full_state):
        full_state.confidence_score = 0.0
        full_state.asked_questions = []
        q = select_question(full_state)
        assert q is not None
        assert q.question_type == QuestionType.REQUIRED
        assert q.key == "declared_stage"

    def test_no_question_when_max_reached(self, full_state):
        full_state.asked_questions = [
            "monthly_revenue", "team_size", "primary_channel", "biggest_bottleneck"
        ]
        q = select_question(full_state)
        assert q is None

    def test_no_optional_when_allow_optional_false(self, full_state):
        # Satisfy baseline + adaptive required
        full_state.answers = {
            "declared_stage": "early_traction",
            "declared_goals": "[\"lead_generation\"]",
            "team_size": "10",
            "primary_channel": "LinkedIn",
            "biggest_bottleneck": "lead quality",
        }
        q = select_question(full_state)
        assert q is None

    def test_optional_question_asked_when_low_confidence(self, full_state):
        # Baseline + required satisfied, optional only with allow_optional=True
        full_state.answers = {
            "declared_stage": "mvp",
            "declared_goals": "[\"lead_generation\",\"revenue_growth\"]",
            "team_size": "2",
            "primary_channel": "cold email",
            "biggest_bottleneck": "getting meetings",
        }
        full_state.allow_optional = True
        # Reset max so optional can be asked
        with patch(
            "app.services.profile_engine.question_selector._MAX_QUESTIONS_PER_SESSION",
            10,
        ):
            q = select_question(full_state)
        # Should select first optional question
        assert q is not None
        assert q.question_type == QuestionType.OPTIONAL

    def test_idea_stage_skips_website_url_optional(self, full_state):
        full_state.answers = {
            "declared_stage": "idea_stage",
            "declared_goals": "[\"brand_awareness\"]",
            "team_size": "solo",
        }
        full_state.declared_stage = ReadinessLevel.IDEA_STAGE
        full_state.allow_optional = True
        q = select_question(full_state)
        assert q is not None
        assert q.key != "website_url"

    def test_non_idea_stage_can_receive_website_url_optional(self, full_state):
        full_state.answers = {
            "declared_stage": "mvp",
            "declared_goals": "[\"brand_awareness\"]",
            "team_size": "solo",
        }
        full_state.declared_stage = ReadinessLevel.MVP
        full_state.allow_optional = True
        q = select_question(full_state)
        assert q is not None
        assert q.key == "website_url"


# ===========================================================================
# 3.5. Answer normalization unit tests
# ===========================================================================


class TestAnswerNormalization:
    def test_monthly_revenue_numeric_parsing(self):
        assert _parse_monthly_revenue_to_int("1000rs") == 1000
        assert _parse_monthly_revenue_to_int("₹1,200") == 1200
        assert _parse_monthly_revenue_to_int("10k") == 10000
        assert _parse_monthly_revenue_to_int("1 lakh") == 100000
        assert _parse_monthly_revenue_to_int("2 crore") == 20000000

    def test_monthly_revenue_nullish_is_zero(self):
        assert _parse_monthly_revenue_to_int("no") == 0
        assert _parse_monthly_revenue_to_int("none") == 0
        assert _parse_monthly_revenue_to_int("0") == 0


# ===========================================================================
# 4. Orchestrator integration tests (mocked LLM)
# ===========================================================================


class TestOrchestrator:
    """Tests the orchestrator with a mocked LLM that returns deterministic JSON."""

    def _make_llm_mock(self) -> MagicMock:
        """
        Build an async mock LLMClient that returns plausible JSON for each call.
        """
        mock = MagicMock()
        mock.complete_json = AsyncMock(side_effect=self._llm_side_effect)
        return mock

    @staticmethod
    async def _llm_side_effect(prompt: str, **kwargs) -> dict:
        """Return context-appropriate mock JSON based on prompt content."""
        if "Extract structured business" in prompt:
            return {
                "business_name": "TestCo",
                "description": "A test SaaS company.",
                "target_audience": "Startups",
                "product_or_service": "CRM tool",
                "revenue_model": "SaaS subscription",
                "current_challenges": ["low awareness"],
                "mentioned_goals": ["grow revenue"],
                "mentioned_channels": ["LinkedIn"],
                "raw_persona_hint": "founder",
                "raw_industry_hint": "saas",
                "raw_readiness_hint": "mvp",
                "confidence_hints": {"persona": 0.8, "industry": 0.9, "readiness": 0.7},
            }
        if "Classify the following business" in prompt:
            return {
                "persona": "founder",
                "industry": "saas",
                "persona_confidence": 0.85,
                "industry_confidence": 0.92,
                "raw_persona": "founder",
                "raw_industry": "saas",
            }
        if "maturity stage" in prompt:
            return {
                "readiness_level": "mvp",
                "readiness_confidence": 0.75,
                "reasoning": "Has basic product, small customer base.",
                "raw_readiness": "mvp",
            }
        if "primary marketing/business need" in prompt:
            return {
                "primary_need": "lead_generation",
                "secondary_needs": ["brand_awareness"],
                "need_confidence": 0.82,
                "reasoning": "MVP stage — needs leads.",
            }
        if "Build a comprehensive marketing profile" in prompt:
            return {
                "business_name": "TestCo",
                "target_audience": "B2B startups",
                "product_or_service": "CRM",
                "revenue_model": "SaaS subscription",
                "current_challenges": ["low awareness"],
                "goals": ["grow revenue"],
                "recommended_channels": ["LinkedIn", "content marketing"],
                "summary": "TestCo is an MVP-stage SaaS CRM targeting B2B startups.",
            }
        if "Generate ONE required follow-up question" in prompt:
            return {
                "slot": "offer",
                "question": "In one sentence, what product or service do you sell?",
                "question_key": "req_offer_1",
                "reason": "Need offer clarity for research.",
            }
        if "Extract the slot value from the user's answer" in prompt:
            return {
                "slot": "offer",
                "value": "CRM tool for B2B startups",
                "confidence": 0.9,
            }
        return {}

    @pytest.mark.asyncio
    async def test_process_input_returns_question(self):
        from app.services.profile_engine.orchestrator import ProfileEngineOrchestrator

        orchestrator = ProfileEngineOrchestrator(llm=self._make_llm_mock())
        sid = uuid.uuid4()
        result = await orchestrator.process_input(
            session_id=sid,
            raw_input="We are a B2B SaaS startup targeting other startups with a CRM tool.",
        )

        assert result.session_id == sid
        assert result.error is None
        assert result.status == SessionStatus.ASKING_QUESTIONS
        assert result.next_question is not None
        assert result.next_question.key == "declared_stage"

    @pytest.mark.asyncio
    async def test_baseline_hydrates_extracted_goals_and_description(self):
        from app.services.profile_engine.orchestrator import ProfileEngineOrchestrator

        orchestrator = ProfileEngineOrchestrator(llm=self._make_llm_mock())
        sid = uuid.uuid4()
        state = SessionState(
            session_id=sid,
            status=SessionStatus.ASKING_QUESTIONS,
            raw_input="",
            answers={
                "declared_stage": "idea_stage",
                "team_size": "2",
                "declared_goals": "[\"brand_awareness\"]",
                "target_market_geo": "local",
            },
            required_slots_filled={
                ResearchSlot.OFFER: "t-shirts, hoodies",
                ResearchSlot.ICP: "college students",
            },
        )

        # No extraction results yet; baseline hydration should create extracted fields.
        orchestrator._sync_baseline_fields(state)
        orchestrator._hydrate_extracted_from_baseline(state)
        orchestrator._hydrate_extracted_from_required_slots(state)

        assert state.extracted is not None
        assert state.extracted.mentioned_goals == ["brand_awareness"]
        assert state.extracted.description is not None

    @pytest.mark.asyncio
    async def test_process_answer_updates_state(self, full_state):
        from app.services.profile_engine.orchestrator import ProfileEngineOrchestrator

        orchestrator = ProfileEngineOrchestrator(llm=self._make_llm_mock())
        result = await orchestrator.process_answer(
            state=full_state,
            question_key="declared_stage",
            answer="mvp",
        )

        assert result.session_id == full_state.session_id
        assert result.error is None
        assert "declared_stage" in full_state.answers
        assert result.next_question is not None
        assert result.next_question.key in ("team_size", "declared_goals")

    @pytest.mark.asyncio
    async def test_build_final_profile_returns_profile(self, full_state):
        from app.services.profile_engine.orchestrator import ProfileEngineOrchestrator

        orchestrator = ProfileEngineOrchestrator(llm=self._make_llm_mock())
        # Fill required answers to skip question phase
        full_state.asked_questions = [
            "monthly_revenue", "team_size", "primary_channel", "biggest_bottleneck"
        ]
        full_state.answers = {
            "monthly_revenue": "$20k",
            "team_size": "5",
            "primary_channel": "LinkedIn",
            "biggest_bottleneck": "conversion rate",
        }
        full_state.confidence_score = 0.75
        full_state.status = SessionStatus.BUILDING_PROFILE

        result = await orchestrator.build_final_profile(full_state)

        assert result.status == SessionStatus.COMPLETE
        assert result.profile is not None
        assert result.profile.persona == PersonaType.FOUNDER
        assert result.profile.industry == IndustryType.SAAS


# ===========================================================================
# 5. Schema validation tests
# ===========================================================================


class TestSchemas:
    def test_submit_input_min_length(self):
        from pydantic import ValidationError

        from app.schemas.profile_engine import SubmitInputRequest

        with pytest.raises(ValidationError):
            SubmitInputRequest(raw_input="short")

    def test_submit_input_strips_whitespace(self):
        from app.schemas.profile_engine import SubmitInputRequest

        req = SubmitInputRequest(raw_input="  Hello world, this is my business.  ")
        assert not req.raw_input.startswith(" ")
        assert not req.raw_input.endswith(" ")

    def test_answer_question_request_strips(self):
        from app.schemas.profile_engine import AnswerQuestionRequest

        req = AnswerQuestionRequest(question_key="team_size", answer="  5 people  ")
        assert req.answer == "5 people"

    def test_answer_question_request_list_strips(self):
        from app.schemas.profile_engine import AnswerQuestionRequest

        req = AnswerQuestionRequest(
            question_key="declared_goals",
            answer=[" lead_generation ", " ", "revenue_growth"],
        )
        assert req.answer == ["lead_generation", "revenue_growth"]

    def test_final_profile_confidence_bounds(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FinalProfile(
                session_id=uuid.uuid4(),
                persona=PersonaType.FOUNDER,
                industry=IndustryType.SAAS,
                readiness_level=ReadinessLevel.MVP,
                primary_need=NeedState.LEAD_GENERATION,
                confidence_score=1.5,  # out of range
            )

    def test_session_state_defaults(self):
        state = SessionState(session_id=uuid.uuid4())
        assert state.status == SessionStatus.PENDING
        assert state.answers == {}
        assert state.confidence_score == 0.0

    def test_question_option_requires_text_defaults(self):
        opt = QuestionOption(value="other", label="Other")
        assert opt.requires_text is False
        assert opt.text_placeholder is None


class TestProfileBuilderNullHandling:
    def test_safe_str_treats_null_like_none(self):
        from app.services.profile_engine.profile_builder import _parse_llm_response

        enriched = _parse_llm_response(
            {
                "business_name": "null",
                "target_audience": "None",
                "product_or_service": "  ",
                "revenue_model": "N/A",
                "current_challenges": [],
                "goals": [],
                "recommended_channels": [],
                "summary": "undefined",
            }
        )
        assert enriched["business_name"] is None
        assert enriched["target_audience"] is None
        assert enriched["product_or_service"] is None
        assert enriched["revenue_model"] is None
        assert enriched["summary"] is None


# ===========================================================================
# 6. Dynamic required slots unit tests
# ===========================================================================


class TestDynamicRequired:
    @pytest.mark.asyncio
    async def test_missing_slots_offer_and_icp_when_extraction_empty(self):
        sid = uuid.uuid4()
        state = SessionState(session_id=sid, raw_input="Test business")
        state.extracted = ExtractedData(
            product_or_service=None,
            target_audience=None,
        )
        missing = dynamic_required.missing_required_slots(state)
        assert ResearchSlot.OFFER in missing
        assert ResearchSlot.ICP in missing

    @pytest.mark.asyncio
    async def test_generate_question_and_fill_slot(self):
        # Mock LLM
        llm = MagicMock()

        async def side_effect(prompt: str, **kwargs) -> dict:
            if "Generate ONE required follow-up question" in prompt:
                return {
                    "slot": "offer",
                    "question": "What do you sell?",
                    "question_key": "req_offer_1",
                    "reason": "Required.",
                }
            if "Extract the slot value from the user's answer" in prompt:
                return {"slot": "offer", "value": "Marketing automation SaaS", "confidence": 0.9}
            return {}

        llm.complete_json = AsyncMock(side_effect=side_effect)

        sid = uuid.uuid4()
        state = SessionState(session_id=sid, raw_input="We do something")
        state.extracted = ExtractedData(product_or_service=None, target_audience=None)
        q = await dynamic_required.generate_required_question(state, llm)
        assert q.slot == ResearchSlot.OFFER
        assert q.key.startswith("req_offer_")

        await dynamic_required.fill_slot_from_answer(
            state=state,
            slot=ResearchSlot.OFFER,
            question_text=q.text,
            answer_text="We sell marketing automation SaaS.",
            llm=llm,
        )
        assert state.required_slots_filled[ResearchSlot.OFFER]
        assert ResearchSlot.OFFER not in state.required_slots_missing


# ===========================================================================
# 7. Deterministic fallbacks unit tests
# ===========================================================================


class TestDeterministicFallbacks:
    @pytest.mark.asyncio
    async def test_need_routing_goal_fallback_sets_primary_need(self):
        from app.services.profile_engine.need_routing import route_need

        llm = MagicMock()
        llm.complete_json = AsyncMock(
            return_value={
                "primary_need": "unknown",
                "secondary_needs": [],
                "need_confidence": 0.2,
                "reasoning": "Not enough info.",
            }
        )

        extracted = ExtractedData(description="Test business", current_challenges=[], mentioned_goals=[])
        classified = ClassifiedProfile(persona=PersonaType.FOUNDER, industry=IndustryType.OTHER)
        readiness = ReadinessAssessment(readiness_level=ReadinessLevel.UNKNOWN)
        answers = {"declared_goals": "[\"lead_generation\"]"}

        routing = await route_need(
            extracted=extracted,
            classified=classified,
            readiness=readiness,
            answers=answers,
            llm=llm,
        )
        assert routing.primary_need == NeedState.LEAD_GENERATION
        assert routing.need_confidence >= 0.5

    @pytest.mark.asyncio
    async def test_classifier_offer_fallback_maps_apparel_to_ecommerce(self):
        from app.services.profile_engine.classifier import classify

        llm = MagicMock()
        llm.complete_json = AsyncMock(
            return_value={
                "persona": "unknown",
                "industry": "other",
                "persona_confidence": 0.4,
                "industry_confidence": 0.2,
            }
        )

        extracted = ExtractedData(
            description="Selling hoodies online",
            product_or_service="Hoodies and t-shirts",
            raw_persona_hint=None,
            raw_industry_hint=None,
            confidence_hints={},
        )

        result = await classify(extracted=extracted, llm=llm)
        assert result.industry == IndustryType.ECOMMERCE
        assert result.industry_confidence >= 0.5
