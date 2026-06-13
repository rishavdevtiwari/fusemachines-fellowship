# Week 2 — ClassicModels API Setup Guide

This project is a FastAPI-based REST API designed to interact with a PostgreSQL database populated with the `ClassicModels` sample dataset. It includes support for database routing, schemas, CRUD operations, and asynchronous concurrent endpoint execution.

---

## Prerequisites

Before starting, ensure you have the following installed on your machine:
1. **Python 3.10+**
2. **Docker Desktop** (for PostgreSQL deployment)

---

## Directory Structure

```text
Week2_SoftEng/
└── project/
    ├── crud/               # CRUD business logic
    ├── routers/            # FastAPI Endpoint routes
    ├── schemas/            # Pydantic data schemas
    ├── database.py         # SQLAlchemy configuration
    ├── logger.py           # Logging module
    ├── main.py             # FastAPI App definition & startup
    ├── requirements.txt    # Project Python dependencies
    ├── docker-compose.yml  # Docker Compose config for Postgres DB
    └── seed.sql            # Database seed script for ClassicModels
```

---

## Setup Instructions

### 1. Environment Setup

Navigate into the project directory and create a virtual environment:

```bash
cd project
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

1. Copy the `.env.example` file to `.env`:
   ```bash
   copy .env.example .env
   ```
2. Open the `.env` file and configure your Postgres connection parameters:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=classicmodels
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/classicmodels
```

---

### 3. Initialize & Seed Database via Docker

Use Docker Compose to run a PostgreSQL container and seed the database using the provided `seed.sql`:

```bash
docker compose up -d
```

*Note: The container uses the `seed.sql` inside the `./docker-entrypoint-initdb.d/` container path to automatically build schemas and insert seed data on its first execution.*

To verify the container is running:
```bash
docker ps
```

---

### 4. Running the Development Server

Start the FastAPI development server using `uvicorn`:

```bash
uvicorn main:app --reload
```

The application will start at:
- **API Server URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Redoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints

- **`GET /overall_counts`**: Returns total counts of all entities concurrently using `asyncio.gather` tasks.
- **`GET /products` / `POST /products`**: Get lists or create new product entities.
- **`GET /customers` / `POST /customers`**: Manage customer records.
