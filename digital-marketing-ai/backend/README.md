# Digital Marketing AI Backend

FastAPI-based backend for the Digital Marketing AI platform with profile engine capabilities.

## Prerequisites

- Python 3.10+
- pip
- Virtual environment (venv or similar)

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements/dev.txt
```

### 4. Environment Configuration

Create a `.env` file in the backend directory:

```bash
cp .env.example .env  # if available, or create manually
```

Set required environment variables:
- `CORS_ORIGINS` - Comma-separated list of allowed origins (default: localhost:3000, 5173, 4173)
- `DATABASE_URL` - Database connection string (optional, uses SQLite by default)
- Other integration credentials (if needed)

## Running the Server

### Development Mode (with auto-reload)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will be available at:
- **API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Demo Page**: http://localhost:8000/demo/

## Project Structure

```
app/
├── main.py                 # FastAPI application entry point
├── api/
│   └── routes/
│       └── profile_engine.py  # Profile engine endpoints
├── services/
│   └── profile_engine/     # Profile engine business logic
├── models/                 # SQLAlchemy database models
├── schemas/                # Pydantic schemas
├── core/
│   ├── config.py          # Configuration
│   ├── database.py        # Database setup
│   └── security.py        # Security utilities
└── db/
    ├── session.py         # Database session
    └── seed.py            # Database seeding
```

## API Endpoints

### Profile Engine
- `POST /api/v1/profile` - Submit profile data
- `GET /api/v1/profile/{id}` - Retrieve profile

See `/docs` for full API documentation.

## Database

- Default: SQLite (local_profile_engine.db)
- Migrations: Uses Alembic in the parent directory
- Tables are auto-created on startup from `ProfileEngineBase.metadata.create_all`

## Troubleshooting

### "Attribute 'app' not found in module 'app.main'"
- Ensure `main.py` exists and contains the `app = FastAPI(...)` definition
- Check that the virtual environment is activated

### ImportError for dependencies
```bash
# Reinstall dependencies
pip install -r requirements/dev.txt --force-reinstall
```

### Port already in use
```bash
# Change port
uvicorn app.main:app --reload --port 8001
```

## Development

### Run Tests
```bash
pytest
```

### Code Formatting & Linting
```bash
black .
ruff check .
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
