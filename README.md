# Attendance System with Face Recognition

> Experimental / Smaller Projects portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

A practical, self-contained demo of face recognition applied to a real workflow: attendance tracking.

## 2. Architecture

```text
Camera -> Face Detection + Recognition -> Match Against Enrolled Faces -> Log Check-In -> Dashboard
```

## 3. Technology Stack

- Python
- face_recognition / dlib
- OpenCV
- SQLite/PostgreSQL
- Streamlit

## 4. Feature List

- Face enrollment
- Face recognition matching
- Check-in logging
- Attendance history
- Simple dashboard

## 5. Implementation Plan

1. Phase 1: Enrollment and recognition pipeline
2. Phase 2: Check-in logging and history storage
3. Phase 3: Simple attendance dashboard

## 6. Repository Structure

```text
face-recognition-attendance/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd face-recognition-attendance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Document the commands used to run training, ingestion, or the main pipeline, e.g.:

```bash
# Phase 1: enroll a person from one or more face images (.npy or image files)
python -m src.main enroll --person-id alice --name "Alice Doe" \
    --images samples/alice_1.npy samples/alice_2.npy
python -m src.main list
python -m src.main remove --person-id alice
```

The default `hash` backend is a deterministic, dependency-free stand-in used for
offline runs and tests. Install the `recognition` extra and pass `--backend dlib`
to use real `face_recognition` encodings.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t face-recognition-attendance .
docker run -p 8000:8000 face-recognition-attendance
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-18_
