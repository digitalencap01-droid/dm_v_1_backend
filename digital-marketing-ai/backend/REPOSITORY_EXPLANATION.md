# Digital Marketing AI Backend Repository Explanation

## 1. Repository ka short overview

Yeh backend repository ek `FastAPI` based project hai jisme multiple product areas ka scaffold bana hua hai:

- `Profile Engine`: sabse zyada implemented module. Business input se profile build karta hai.
- `Ideation`: folder structure bana hua hai, lekin current codebase mein mostly placeholders hain.
- `Digital Marketing`: APIs, services, repositories ka skeleton bana hua hai.
- `Sales Marketing`: voice, CRM, scoring, enrichment, follow-up, feedback loop ka scaffold + kuch helper files implemented hain.
- `Integrations`, `models`, `tasks`, `middleware`, `core`: mostly future expansion ke liye blank placeholders.

## 2. Current codebase reality check

- Total Python files: `340`
- Non-empty Python files: `59`
- Empty Python files: `281`
- Most real implementation `profile_engine`, `llm client`, kuch `sales_marketing` helper modules, demo HTML aur demo script mein hai.
- Bahut saare files abhi sirf architecture placeholder hain. Unka naam future responsibility batata hai, lekin code abhi likha nahi gaya.

## 3. High-level flow

1. `app/main.py` FastAPI app boot karta hai.
2. `app/api/routes/profile_engine.py` HTTP endpoints expose karta hai.
3. `app/repositories/profile_engine_repository.py` DB persistence handle karta hai.
4. `app/services/profile_engine/*` extraction, classification, readiness, need routing, confidence, questions, aur final profile build karta hai.
5. `app/services/llm/client.py` external LLM provider call karta hai.
6. `app/services/llm/prompts.py` prompts generate karta hai.
7. `app/db/session.py` async SQLAlchemy session deta hai.
8. `app/services/profile_engine/model.py` database tables define karta hai.

## 4. Top-level files explanation

- `README.md`: empty placeholder, project overview likhna baaki hai.
- `.env`: empty local environment file.
- `.env.example`: empty template file, environment variables document karna baaki hai.
- `.gitignore`: virtualenv, db, coverage, cache, env aur build files ignore karta hai.
- `alembic.ini`: empty placeholder for migration config.
- `docker-compose.yml`: empty placeholder for container orchestration.
- `Dockerfile`: empty placeholder for image build.
- `requirements/base.txt`: empty placeholder for base dependencies.
- `requirements/dev.txt`: empty placeholder for dev dependencies.
- `requirements/prod.txt`: empty placeholder for prod dependencies.
- `alembic/env.py`: empty migration environment placeholder.
- `alembic/script.py.mako`: empty migration template placeholder.
- `scripts/run_profile_engine_cerebras.py`: Cerebras API ke against profile engine ko demo mode mein run karta hai, preset answers ke saath full pipeline test karta hai.
- `app/static/index.html`: profile engine ke liye browser demo UI hai jo same-origin API calls karta hai.

## 5. Implemented source files with action summary

### App bootstrap and DB

- `app/main.py`: FastAPI app create karta hai, `.env` load karta hai, CORS configure karta hai, profile engine router mount karta hai, startup par tables create karta hai, aur `/demo` static page serve karta hai.
- `app/db/session.py`: async database URL resolve karta hai, SQLAlchemy async engine banata hai, session factory create karta hai, aur `get_db_session()` dependency ke through request-scoped DB session deta hai.

### API layer

- `app/api/routes/profile_engine.py`: profile engine ke HTTP endpoints define karta hai.
  Action:
  - session start karna
  - raw business input lena
  - answers save karna
  - question thread fetch karna
  - final profile generate karna
  - session status aur final profile return karna

### Schemas

- `app/schemas/profile_engine.py`: poore profile engine ka contract file hai.
  Action:
  - enums define karta hai: persona, industry, readiness, need state, goal, question type, session status
  - Pydantic models define karta hai: extracted data, classified data, readiness, routing, session state, final profile
  - request/response schemas define karta hai
  - input trimming aur validation karta hai

### Repository layer

- `app/repositories/profile_engine_repository.py`: profile engine ka persistence layer hai.
  Action:
  - new session create karta hai
  - session fetch karta hai
  - answers save/update karta hai
  - answer history nikalta hai
  - final profile persist karta hai
  - DB data se `SessionState` reconstruct karta hai
  - extracted metadata blob mein baseline-related flags bhi preserve karta hai

- `app/repositories/digital_marketing/integration_repository.py`: non-empty hai, lekin current code mein koi meaningful class/function exposed nahi milta; likely early placeholder ya partial file.

### LLM layer

- `app/services/llm/client.py`: unified LLM wrapper hai.
  Action:
  - provider detect karta hai: `CEREBRAS_API_KEY` ya `OPENAI_API_KEY`
  - chat completion request bhejta hai
  - JSON response parse karta hai
  - HTTP errors ko domain-specific error mein convert karta hai
  - singleton client return karta hai

- `app/services/llm/prompts.py`: saare prompt builders ka central file hai.
  Action:
  - extraction prompt banata hai
  - classification prompt banata hai
  - readiness prompt banata hai
  - need routing prompt banata hai
  - final profile prompt banata hai
  - answer extraction aur required-question prompts banata hai

### Profile Engine domain services

- `app/services/profile_engine/model.py`: SQLAlchemy models define karta hai.
  Action:
  - `ProfileEngineSession`
  - `ProfileEngineAnswer`
  - `ProfileEngineProfile`
  - timestamps, IDs aur table structure define karta hai

- `app/services/profile_engine/extractor.py`: raw business text se structured extracted data nikalta hai.
  Action:
  - LLM se extraction
  - parse + sanitize
  - minimal fallback banana agar LLM fail ho

- `app/services/profile_engine/classifier.py`: extracted data ko persona aur industry categories mein classify karta hai.
  Action:
  - LLM-based classification
  - fallback heuristics
  - confidence clamp

- `app/services/profile_engine/readiness.py`: business kis stage par hai uska readiness assessment karta hai.
  Action:
  - LLM se stage inference
  - fallback heuristics
  - readiness score normalize karna

- `app/services/profile_engine/need_routing.py`: primary aur secondary needs identify karta hai.
  Action:
  - LLM se need routing
  - heuristic fallback
  - need confidence normalization

- `app/services/profile_engine/normalizer.py`: raw strings ko typed enums mein map karta hai.
  Action:
  - persona normalize
  - industry normalize
  - readiness normalize
  - need state aur need list normalize

- `app/services/profile_engine/confidence.py`: session confidence score calculate karta hai.
  Action:
  - classified data
  - readiness data
  - need routing
  - answers presence
  in sab ko combine karke total confidence banata hai

- `app/services/profile_engine/question_selector.py`: next question choose karta hai.
  Action:
  - required question select karna
  - optional question allow/deny karna
  - current session state ke basis par next prompt return karna

- `app/services/profile_engine/required_fields.py`: batata hai kaunse question keys abhi required hain.

- `app/services/profile_engine/dynamic_required.py`: missing research slots ke liye smart follow-up generate karta hai.
  Action:
  - missing slots identify karna
  - LLM se required question banana
  - answer se slot fill karne ki koshish
  - fallback question return karna

- `app/services/profile_engine/profile_builder.py`: final `FinalProfile` generate karta hai.
  Action:
  - all collected state ko final structured profile mein convert karna
  - LLM output parse karna
  - safe string/list cleanup

- `app/services/profile_engine/orchestrator.py`: entire profile engine ka main brain hai.
  Action:
  - initial extraction/classification/readiness/routing pipeline chalana
  - baseline answers sync karna
  - answer process karna
  - confidence update karna
  - next step decide karna
  - optional ya required question flow control karna
  - final profile build trigger karna

### AI utility files

- `app/services/ai/orchestrator.py`: non-empty hai lekin currently major executable orchestration content visible nahi hai; likely future AI orchestration module.
- `app/services/ai/schemas.py`: non-empty placeholder/partial schema file for AI-specific typed structures.

### Digital marketing implemented file

- `app/services/digital_marketing/content_generation_service.py`: non-empty hai; naming ke hisaab se campaign/content generation ke liye service entry point hai, lekin current file state partial lagti hai.

### Sales marketing implemented helpers

- `app/services/sales_marketing/analysis/action_items_extractor.py`: meeting/call notes se action items nikalne ke liye reserved helper.
- `app/services/sales_marketing/analysis/meeting_history_comparator.py`: current call ko past meeting context se compare karne ke liye helper.
- `app/services/sales_marketing/analysis/next_action_recommender.py`: sales next-step suggestion ke liye helper.
- `app/services/sales_marketing/analysis/transcript_processor.py`: transcripts ko downstream analysis ke liye preprocess karne ka intended file.
- `app/services/sales_marketing/analysis/notetakers/__init__.py`: notetaker adapter package marker.
- `app/services/sales_marketing/analysis/notetakers/fireflies_adapter.py`: Fireflies integration adapter ka placeholder/partial implementation.
- `app/services/sales_marketing/analysis/notetakers/gong_adapter.py`: Gong integration adapter ka placeholder/partial implementation.
- `app/services/sales_marketing/analysis/notetakers/otter_adapter.py`: Otter integration adapter ka placeholder/partial implementation.

- `app/services/sales_marketing/crm/adapters/hubspot_adapter.py`: HubSpot CRM adapter ka non-empty stub/partial file.
- `app/services/sales_marketing/crm/adapters/salesforce_adapter.py`: Salesforce CRM adapter ka non-empty stub/partial file.

- `app/services/sales_marketing/feedback_loop/content_optimizer.py`: feedback ke basis par content improve karne ke liye intended helper.
- `app/services/sales_marketing/feedback_loop/outcome_collector.py`: outcomes collect karne ke liye helper.
- `app/services/sales_marketing/feedback_loop/rep_coaching_engine.py`: rep coaching recommendations ke liye helper.
- `app/services/sales_marketing/feedback_loop/score_model_retrainer.py`: score model retraining workflow ka reserved file.

- `app/services/sales_marketing/follow_up/one_click_approver.py`: follow-up approval workflow ka helper.

- `app/services/sales_marketing/lead_generation/lead_service.py`: lead generation orchestration entry point ka partial/non-empty file.
- `app/services/sales_marketing/lead_generation/channels/__init__.py`: lead channels package marker.
- `app/services/sales_marketing/lead_generation/channels/inbound_web.py`: inbound web leads handle karne ke liye file.
- `app/services/sales_marketing/lead_generation/channels/outbound_email.py`: outbound email lead source handler.
- `app/services/sales_marketing/lead_generation/channels/paid_ads_webhook.py`: paid ads webhook lead ingestion file.
- `app/services/sales_marketing/lead_generation/channels/social_scraper.py`: social/source scraping based lead acquisition file.
- `app/services/sales_marketing/lead_generation/enrichment/__init__.py`: enrichment package marker.
- `app/services/sales_marketing/lead_generation/enrichment/clearbit_service.py`: Clearbit enrichment stub/partial file.
- `app/services/sales_marketing/lead_generation/enrichment/clay_service.py`: Clay enrichment stub/partial file.
- `app/services/sales_marketing/lead_generation/enrichment/hunter_service.py`: Hunter enrichment stub/partial file.
- `app/services/sales_marketing/lead_generation/enrichment/orchestrator.py`: multiple enrichment providers coordinate karne ka intended file.

- `app/services/sales_marketing/scoring/adapters/einstein_scoring.py`: Einstein scoring adapter stub/partial file.
- `app/services/sales_marketing/scoring/adapters/hubspot_scoring.py`: HubSpot scoring adapter stub/partial file.

- `app/services/sales_marketing/voice/llm/response_generator.py`: voice agent ke liye LLM response generate karne ka file.
- `app/services/sales_marketing/voice/pre_call/brief_generator.py`: call se pehle rep/agent briefing banane ke liye file.
- `app/services/sales_marketing/voice/telephony/bland_ai_client.py`: Bland AI telephony provider wrapper ka partial file.
- `app/services/sales_marketing/voice/telephony/retell_client.py`: Retell telephony wrapper ka partial file.
- `app/services/sales_marketing/voice/transfer/availability_checker.py`: warm handoff ya transfer se pehle availability check karne ka helper.
- `app/services/sales_marketing/voice/transfer/whisper_coach.py`: call ke dauran whisper coaching ke liye helper.
- `app/services/sales_marketing/voice/tts/elevenlabs_service.py`: ElevenLabs text-to-speech provider ka partial file.

### Tests

- `app/tests/profile_engine/test_profile_engine.py`: profile engine ka main automated test suite hai.
  Action:
  - normalizer tests
  - confidence tests
  - question selector tests
  - orchestrator tests
  - schema validation tests
  - dynamic required question flow tests

## 6. Empty placeholder files inventory

Neeche listed files current repository mein `empty` hain. Inka naam intended responsibility batata hai, lekin actual code abhi implement nahi hua.

### Empty API files

`app/api/__init__.py`, `app/api/routes/__init__.py`, `app/api/v1/__init__.py`, `app/api/v1/auth/__init__.py`, `app/api/v1/auth/routes.py`, `app/api/v1/auth/schemas.py`, `app/api/v1/auth/service.py`, `app/api/v1/deps.py`, `app/api/v1/digital_marketing/__init__.py`, `app/api/v1/digital_marketing/repository.py`, `app/api/v1/digital_marketing/routes.py`, `app/api/v1/digital_marketing/schemas.py`, `app/api/v1/digital_marketing/service.py`, `app/api/v1/health/__init__.py`, `app/api/v1/health/checks.py`, `app/api/v1/health/routes.py`, `app/api/v1/health/schemas.py`, `app/api/v1/ideation/__init__.py`, `app/api/v1/ideation/repository.py`, `app/api/v1/ideation/routes.py`, `app/api/v1/ideation/schemas.py`, `app/api/v1/ideation/service.py`, `app/api/v1/router.py`, `app/api/v1/sales_marketing/__init__.py`, `app/api/v1/sales_marketing/repository.py`, `app/api/v1/sales_marketing/routes.py`, `app/api/v1/sales_marketing/schemas.py`, `app/api/v1/sales_marketing/service.py`, `app/api/v1/users/__init__.py`, `app/api/v1/users/routes.py`, `app/api/v1/users/schemas.py`, `app/api/v1/users/service.py`: API surface ke planned placeholders.

### Empty core files

`app/core/__init__.py`, `app/core/celery.py`, `app/core/config.py`, `app/core/constants.py`, `app/core/database.py`, `app/core/exceptions.py`, `app/core/logging.py`, `app/core/redis.py`, `app/core/security.py`: central app infra ke placeholders.

### Empty DB files

`app/db/__init__.py`, `app/db/base.py`, `app/db/seed.py`: DB base metadata aur seed logic ke placeholders.

### Empty integrations

`app/integrations/__init__.py`, `app/integrations/ads/__init__.py`, `app/integrations/ads/google_ads.py`, `app/integrations/ads/linkedin_ads.py`, `app/integrations/ads/meta_ads.py`, `app/integrations/analytics/__init__.py`, `app/integrations/analytics/ga4.py`, `app/integrations/analytics/mixpanel.py`, `app/integrations/crm/__init__.py`, `app/integrations/crm/hubspot.py`, `app/integrations/crm/salesforce.py`, `app/integrations/ecommerce/__init__.py`, `app/integrations/ecommerce/shopify.py`, `app/integrations/llm/__init__.py`, `app/integrations/llm/openai_client.py`, `app/integrations/messaging/__init__.py`, `app/integrations/messaging/sendgrid.py`, `app/integrations/messaging/twilio.py`, `app/integrations/voice/__init__.py`, `app/integrations/voice/bland_ai.py`, `app/integrations/voice/retell.py`, `app/integrations/voice/vapi.py`, `app/integrations/webhooks/__init__.py`, `app/integrations/webhooks/inbound.py`, `app/integrations/webhooks/outbound.py`: external providers ke future adapters.

### Empty middleware

`app/middleware/__init__.py`, `app/middleware/audit.py`, `app/middleware/auth.py`, `app/middleware/rate_limit.py`, `app/middleware/request_context.py`: request middleware scaffolding.

### Empty models

`app/models/__init__.py`, `app/models/attribution_report.py`, `app/models/audience_segment.py`, `app/models/audit_log.py`, `app/models/automation_blueprint.py`, `app/models/automation_recommendation.py`, `app/models/automation_task.py`, `app/models/budget_optimization.py`, `app/models/call_analysis.py`, `app/models/call_transcript.py`, `app/models/campaign.py`, `app/models/campaign_channel.py`, `app/models/campaign_content.py`, `app/models/consent_record.py`, `app/models/contract.py`, `app/models/crm_sync_log.py`, `app/models/customer_event.py`, `app/models/customer_profile.py`, `app/models/dashboard_snapshot.py`, `app/models/deal.py`, `app/models/follow_up_email.py`, `app/models/generated_idea.py`, `app/models/idea_validation_report.py`, `app/models/ideation_profile.py`, `app/models/ideation_session.py`, `app/models/ideation_step_state.py`, `app/models/integration_account.py`, `app/models/journey.py`, `app/models/journey_step.py`, `app/models/lead_enrichment.py`, `app/models/lead_score.py`, `app/models/lead_score_sales.py`, `app/models/marketing_workspace.py`, `app/models/nurture_sequence.py`, `app/models/outcome_feedback.py`, `app/models/profitability_report.py`, `app/models/role.py`, `app/models/sales_lead.py`, `app/models/segment_member.py`, `app/models/supplier.py`, `app/models/supplier_outreach.py`, `app/models/supplier_shortlist.py`, `app/models/suppression_entry.py`, `app/models/user.py`, `app/models/voice_session.py`, `app/models/webhook_event.py`: ORM/domain models ka future schema layer.

### Empty repositories

`app/repositories/__init__.py`, `app/repositories/base.py`, `app/repositories/digital_marketing/__init__.py`, `app/repositories/digital_marketing/analytics_repository.py`, `app/repositories/digital_marketing/campaign_repository.py`, `app/repositories/digital_marketing/compliance_repository.py`, `app/repositories/digital_marketing/customer_repository.py`, `app/repositories/digital_marketing/journey_repository.py`, `app/repositories/digital_marketing/segment_repository.py`, `app/repositories/digital_marketing/workspace_repository.py`, `app/repositories/ideation/__init__.py`, `app/repositories/ideation/automation_repository.py`, `app/repositories/ideation/idea_repository.py`, `app/repositories/ideation/profile_repository.py`, `app/repositories/ideation/profitability_repository.py`, `app/repositories/ideation/session_repository.py`, `app/repositories/ideation/supplier_repository.py`, `app/repositories/ideation/validation_repository.py`, `app/repositories/sales_marketing/__init__.py`, `app/repositories/sales_marketing/analysis_repository.py`, `app/repositories/sales_marketing/crm_repository.py`, `app/repositories/sales_marketing/deal_repository.py`, `app/repositories/sales_marketing/feedback_repository.py`, `app/repositories/sales_marketing/lead_repository.py`, `app/repositories/sales_marketing/scoring_repository.py`, `app/repositories/sales_marketing/voice_repository.py`, `app/repositories/user_repository.py`: DB access layers ke placeholders.

### Empty schemas

`app/schemas/__init__.py`, `app/schemas/common.py`, `app/schemas/pagination.py`, `app/schemas/response.py`: common response/pagination/schema placeholders.

### Empty services

`app/services/ai/__init__.py`, `app/services/ai/llm_client.py`, `app/services/ai/prompt_builder.py`, `app/services/digital_marketing/__init__.py`, `app/services/digital_marketing/analytics_service.py`, `app/services/digital_marketing/attribution_service.py`, `app/services/digital_marketing/campaign_service.py`, `app/services/digital_marketing/cdp_service.py`, `app/services/digital_marketing/compliance_service.py`, `app/services/digital_marketing/integration_service.py`, `app/services/digital_marketing/journey_service.py`, `app/services/digital_marketing/lead_scoring_service.py`, `app/services/digital_marketing/optimization_service.py`, `app/services/digital_marketing/segmentation_service.py`, `app/services/digital_marketing/workspace_service.py`, `app/services/files/__init__.py`, `app/services/files/storage_service.py`, `app/services/ideation/__init__.py`, `app/services/ideation/automation_blueprint_service.py`, `app/services/ideation/dna_service.py`, `app/services/ideation/idea_generation_service.py`, `app/services/ideation/outreach_service.py`, `app/services/ideation/profile_service.py`, `app/services/ideation/profitability_service.py`, `app/services/ideation/recommendation_service.py`, `app/services/ideation/session_service.py`, `app/services/ideation/supplier_service.py`, `app/services/ideation/validation_service.py`, `app/services/llm/__init__.py`, `app/services/notifications/__init__.py`, `app/services/notifications/email_service.py`, `app/services/notifications/in_app_service.py`, `app/services/notifications/sms_service.py`, `app/services/profile_engine/__init__.py`, `app/services/sales_marketing/__init__.py`, `app/services/sales_marketing/analysis/__init__.py`, `app/services/sales_marketing/analysis/deal_health_scorer.py`, `app/services/sales_marketing/analysis/objection_detector.py`, `app/services/sales_marketing/analysis/sentiment_analyser.py`, `app/services/sales_marketing/close/__init__.py`, `app/services/sales_marketing/close/contract_sender.py`, `app/services/sales_marketing/close/docusign_adapter.py`, `app/services/sales_marketing/close/pandadoc_adapter.py`, `app/services/sales_marketing/crm/__init__.py`, `app/services/sales_marketing/crm/adapters/__init__.py`, `app/services/sales_marketing/crm/adapters/factory.py`, `app/services/sales_marketing/crm/escalation_service.py`, `app/services/sales_marketing/crm/note_writer.py`, `app/services/sales_marketing/crm/stage_updater.py`, `app/services/sales_marketing/crm/task_creator.py`, `app/services/sales_marketing/feedback_loop/__init__.py`, `app/services/sales_marketing/follow_up/__init__.py`, `app/services/sales_marketing/follow_up/email_drafter.py`, `app/services/sales_marketing/follow_up/slack_notifier.py`, `app/services/sales_marketing/lead_generation/__init__.py`, `app/services/sales_marketing/nurture/__init__.py`, `app/services/sales_marketing/nurture/booking_service.py`, `app/services/sales_marketing/nurture/drip_sequence.py`, `app/services/sales_marketing/nurture/routing_rules.py`, `app/services/sales_marketing/nurture/scheduler.py`, `app/services/sales_marketing/scoring/__init__.py`, `app/services/sales_marketing/scoring/adapters/__init__.py`, `app/services/sales_marketing/scoring/adapters/custom_ml.py`, `app/services/sales_marketing/scoring/engine.py`, `app/services/sales_marketing/scoring/schemas.py`, `app/services/sales_marketing/scoring/thresholds.py`, `app/services/sales_marketing/voice/__init__.py`, `app/services/sales_marketing/voice/llm/__init__.py`, `app/services/sales_marketing/voice/llm/claude_adapter.py`, `app/services/sales_marketing/voice/llm/context_window.py`, `app/services/sales_marketing/voice/llm/gpt4o_adapter.py`, `app/services/sales_marketing/voice/llm/intent_detector.py`, `app/services/sales_marketing/voice/pre_call/__init__.py`, `app/services/sales_marketing/voice/pre_call/talk_tracks.py`, `app/services/sales_marketing/voice/stt/__init__.py`, `app/services/sales_marketing/voice/stt/deepgram_service.py`, `app/services/sales_marketing/voice/stt/whisper_service.py`, `app/services/sales_marketing/voice/telephony/__init__.py`, `app/services/sales_marketing/voice/telephony/factory.py`, `app/services/sales_marketing/voice/telephony/vapi_client.py`, `app/services/sales_marketing/voice/transfer/__init__.py`, `app/services/sales_marketing/voice/transfer/triggers.py`, `app/services/sales_marketing/voice/transfer/warm_handoff.py`, `app/services/sales_marketing/voice/tts/__init__.py`, `app/services/sales_marketing/voice/tts/playai_service.py`, `app/services/sales_marketing/voice/voice_agent.py`: future business logic files.

### Empty tasks

`app/tasks/__init__.py`, `app/tasks/analytics_tasks.py`, `app/tasks/campaign_execution_tasks.py`, `app/tasks/compliance_tasks.py`, `app/tasks/content_generation_tasks.py`, `app/tasks/crm_sync_tasks.py`, `app/tasks/enrichment_tasks.py`, `app/tasks/feedback_loop_tasks.py`, `app/tasks/ideation_tasks.py`, `app/tasks/integration_sync_tasks.py`, `app/tasks/lead_scoring_tasks.py`, `app/tasks/supplier_tasks.py`, `app/tasks/voice_tasks.py`: background jobs ke placeholders.

### Empty tests

`app/tests/__init__.py`, `app/tests/conftest.py`, `app/tests/digital_marketing/__init__.py`, `app/tests/digital_marketing/test_analytics.py`, `app/tests/digital_marketing/test_campaigns.py`, `app/tests/digital_marketing/test_cdp.py`, `app/tests/digital_marketing/test_compliance.py`, `app/tests/digital_marketing/test_journeys.py`, `app/tests/digital_marketing/test_segmentation.py`, `app/tests/ideation/__init__.py`, `app/tests/ideation/test_automation_blueprint.py`, `app/tests/ideation/test_profitability.py`, `app/tests/ideation/test_session.py`, `app/tests/ideation/test_validation.py`, `app/tests/profile_engine/__init__.py`, `app/tests/sales_marketing/__init__.py`, `app/tests/sales_marketing/test_analysis.py`, `app/tests/sales_marketing/test_close.py`, `app/tests/sales_marketing/test_crm_sync.py`, `app/tests/sales_marketing/test_lead_generation.py`, `app/tests/sales_marketing/test_scoring.py`, `app/tests/sales_marketing/test_voice_agent.py`, `app/tests/test_auth.py`: planned automated tests, mostly not implemented.

### Empty utils

`app/utils/__init__.py`, `app/utils/calculators.py`, `app/utils/dates.py`, `app/utils/enums.py`, `app/utils/helpers.py`, `app/utils/validators.py`: helper utility placeholders.

## 7. Folder-wise practical understanding

- `app/api`: HTTP entrypoints rakhne ke liye.
- `app/services`: business logic ka main home.
- `app/repositories`: DB read/write abstraction ke liye.
- `app/schemas`: request/response and typed domain models.
- `app/models`: future SQLAlchemy/domain entities.
- `app/integrations`: external APIs/providers.
- `app/tasks`: async/background processing.
- `app/tests`: unit/integration tests.

## 8. Kis files par abhi real kaam hua dikhta hai

Aapke IDE context aur current repository state ke hisaab se yeh files sabse relevant/active lagti hain:

- `app/api/routes/profile_engine.py`
  Action: session lifecycle, input submission, Q/A flow, message history, final profile generation.

- `app/repositories/profile_engine_repository.py`
  Action: session state save/load, answer upsert, profile persistence, DB reconstruction.

- `app/schemas/profile_engine.py`
  Action: saare profile engine enums, models, validators, response contracts.

- `app/services/profile_engine/orchestrator.py`
  Action: poora decision engine aur control flow.

- `app/services/profile_engine/extractor.py`
  Action: business text ko structured extracted data mein todna.

- `app/services/profile_engine/classifier.py`
  Action: persona aur industry infer karna.

- `app/services/profile_engine/readiness.py`
  Action: business stage/readiness detect karna.

- `app/services/profile_engine/need_routing.py`
  Action: business need routing karna.

- `app/services/profile_engine/profile_builder.py`
  Action: final profile object banana.

- `app/services/profile_engine/question_selector.py`
  Action: next question choose karna.

- `app/services/profile_engine/dynamic_required.py`
  Action: missing research slots ke liye dynamic follow-up questions banana.

- `app/services/llm/client.py`
  Action: Cerebras/OpenAI API call karna.

- `app/services/llm/prompts.py`
  Action: LLM prompts build karna.

- `app/db/session.py`
  Action: DB connection/session management.

- `app/main.py`
  Action: app boot, router mount, CORS, tables create, demo page serve.

- `app/tests/profile_engine/test_profile_engine.py`
  Action: implemented profile engine behavior verify karna.

## 9. Aapke currently open files ka direct status

- `app/api/routes/profile_engine.py`: implemented, active backend HTTP logic.
- `app/repositories/profile_engine_repository.py`: implemented, active persistence logic.
- `app/services/ideation/idea_generation_service.py`: empty placeholder.
- `app/services/ideation/outreach_service.py`: empty placeholder.
- `app/repositories/ideation/profile_repository.py`: empty placeholder.
- `.env`: present but empty.

## 10. Git status based note

Current `git status --short` ke hisaab se source code files modified track nahi ho rahe. Visible untracked items yeh the:

- `../../.venv/`
- `uvicorn.err.log`
- `uvicorn.out.log`

Iska matlab recent visible coding progress zyada tar existing implemented files mein hai, lekin git working tree mein abhi tracked source modifications reflect nahi ho rahi.

## 11. Final takeaway

Is repository ka strongest completed area `Profile Engine` hai. Baaki modules ka architecture aur naming kaafi strong hai, lekin implementation abhi initial scaffold stage mein hai. Agar aap next development karna chahte ho, sabse logical order hoga:

1. `ideation` module implement karna
2. `api/v1` routers ko actual routes se wire karna
3. `models` aur `repositories` ko real DB schemas se fill karna
4. `requirements` aur infra files ko complete karna
5. placeholder tests ko actual test suites mein convert karna
