# Architecture Notes: Attendance System with Face Recognition

## Pipeline

```text
Camera -> Face Detection + Recognition -> Match Against Enrolled Faces -> Log Check-In -> Dashboard
```

## Components

- Face enrollment
- Face recognition matching
- Check-in logging
- Attendance history
- Simple dashboard

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
