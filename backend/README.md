# Industrial Defect Detection API — Backend

> FastAPI backend for serving a YOLOv8-based real-time defect detection model on metal surface images.

---

## Overview

This is the backend service for the **Real-Time Industrial Defect Detection System**. It exposes a REST API built with FastAPI that will eventually accept images of metal surfaces and return detected defects with bounding boxes, confidence scores, and class labels.

---

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/         # Route registrations
│   │   └── endpoints/      # Endpoint handler functions
│   ├── core/
│   │   ├── config.py       # Environment and app configuration
│   │   ├── logger.py       # Centralized logging setup
│   │   └── settings.py     # Pydantic settings management
│   ├── services/           # Business logic (model inference, etc.)
│   ├── models/             # ML model loaders and DB models
│   ├── schemas/            # Pydantic request/response schemas
│   ├── utils/              # Shared helper functions
│   └── main.py             # FastAPI app entry point
├── logs/                   # Application log files (auto-created)
├── docs/                   # API documentation and notes
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
cd backend

# create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at **http://localhost:8000**

---

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | Welcome message                      |
| GET    | `/health`         | Service health check                 |
| POST   | `/predict/image`  | Upload image for defect prediction   |
| POST   | `/predict/video`  | Upload video for defect prediction   |

### API Documentation (auto-generated)

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Folder Explanation

| Folder          | Purpose                                                  |
|-----------------|----------------------------------------------------------|
| `app/api/`      | All API route definitions and endpoint handlers          |
| `app/core/`     | App-wide configuration, settings, and logging            |
| `app/services/` | Business logic — model loading, inference pipelines      |
| `app/models/`   | ML model wrappers and database models                    |
| `app/schemas/`  | Pydantic schemas for input validation and JSON responses |
| `app/utils/`    | Reusable utility functions                               |
| `logs/`         | Log files generated during runtime                       |
| `docs/`         | Additional documentation                                 |
| `tests/`        | Automated tests                                          |

---

## Tech Stack

- **FastAPI** — async web framework
- **Uvicorn** — ASGI server
- **Pydantic** — data validation
- **OpenCV** — image processing
- **Ultralytics** — YOLOv8 inference
- **Python Logging** — structured logging

---

## License

MIT — see the root [LICENSE](../LICENSE) file.
