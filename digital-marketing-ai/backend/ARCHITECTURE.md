# Digital Marketing AI Backend — Architecture & Product Overview

## Product Vision

**Digital Marketing AI** is an intelligent platform designed to help founders, marketers, and agencies rapidly build, analyze, and optimize their digital marketing strategies. The core of the product is the **Profile Intelligence Engine**, which uses AI (including LLMs) to understand a business, assess its readiness, classify its persona and industry, and recommend tailored marketing actions and strategies.

### What We Aim to Do

- **Automate Marketing Profiling:** Instantly generate a comprehensive marketing profile for any business or founder based on a simple description and a few follow-up questions.
- **Personalized Recommendations:** Use AI to classify business personas, industries, readiness levels, and needs, then recommend the best marketing channels, strategies, and next steps.
- **Reduce Onboarding Friction:** Replace lengthy forms and manual interviews with a conversational, AI-driven flow.
- **Centralize Data:** Store all session data, answers, and profiles for analytics, reporting, and continuous improvement.

---

## High-Level Architecture

```mermaid
flowchart TD
    User[User (Web/Client)] -->|HTTP| API[FastAPI Backend]
    API -->|Routes| Orchestrator[Profile Engine Orchestrator]
    Orchestrator -->|Business Logic| LLM[LLM Client]
    Orchestrator -->|Persistence| Repo[Profile Engine Repository]
    Repo -->|DB Ops| DB[(Database)]
    Orchestrator -->|Profile Builder, Classifier, Readiness, etc.| Services[Profile Engine Services]
    API -->|Docs| Swagger[Swagger UI]
```

---

## Component Breakdown

### 1. **API Layer (FastAPI)**
- **Location:** `app/main.py`, `app/api/routes/profile_engine.py`
- **Purpose:** Exposes REST endpoints for session management, input submission, answering questions, and retrieving profiles.
- **Endpoints:**
  - Start session
  - Submit business input
  - Answer follow-up questions
  - Get session status/profile

### 2. **Profile Engine Orchestrator**
- **Location:** `app/services/profile_engine/orchestrator.py`
- **Purpose:** Controls the flow of the profiling pipeline. Decides what happens at each step (e.g., ask next question, build profile).
- **Pipeline Steps:**
  1. **Extract:** Parse and normalize business input.
  2. **Classify:** Assign persona and industry.
  3. **Readiness:** Assess business maturity.
  4. **Need Routing:** Identify primary/secondary marketing needs.
  5. **Confidence:** Score the AI’s certainty.
  6. **Question Selector:** Ask for missing info if needed.
  7. **Profile Builder:** Assemble the final profile.

### 3. **Repository Layer**
- **Location:** `app/repositories/profile_engine_repository.py`
- **Purpose:** Handles all database operations (sessions, answers, profiles).
- **Responsibilities:**
  - Create/load/update sessions
  - Save/load answers
  - Save/load final profiles

### 4. **Schemas & Models**
- **Location:** `app/schemas/profile_engine.py`, `app/models/`
- **Purpose:** Define all data contracts (requests, responses, internal state, DB models).
- **Key Schemas:**
  - SessionState
  - FinalProfile
  - Question, Answer, Profile DTOs

### 5. **Profile Engine Services**
- **Location:** `app/services/profile_engine/`
- **Purpose:** Modular services for extraction, classification, readiness, need routing, profile building, etc.
- **LLM Integration:** Calls out to LLMs for advanced extraction/classification.

### 6. **Database**
- **Location:** `app/models/`, `app/db/`
- **Purpose:** Stores all persistent data (sessions, answers, profiles).
- **Migrations:** Managed with Alembic.

---

## Flow Explanation

1. **Session Start:** User initiates a session via API.
2. **Input Submission:** User submits a free-form business description.
3. **Pipeline Execution:** Orchestrator extracts info, classifies persona/industry, assesses readiness, and identifies needs.
4. **Dynamic Q&A:** If more info is needed, the orchestrator asks targeted follow-up questions.
5. **Profile Assembly:** Once enough data is collected, the orchestrator builds a comprehensive marketing profile.
6. **Profile Delivery:** The final profile is returned to the user and stored for future reference.

---

## Data Model (Profile Example)

A completed profile includes:
- Persona type (e.g., Founder, Marketer, Agency)
- Industry (e.g., SaaS, Ecommerce)
- Readiness level (e.g., MVP, Scaling)
- Primary/secondary needs (e.g., Lead Generation, Brand Awareness)
- Business details (name, audience, product, revenue model)
- Challenges, goals, recommended channels, summary
- AI confidence score

---

## Why This Matters

- **For Users:** Instantly get actionable marketing insights and a clear path forward, tailored to your business.
- **For Teams:** Centralize and automate onboarding, reduce manual work, and improve data quality.
- **For Product:** Scalable, modular, and ready for advanced analytics and continuous improvement.

---
