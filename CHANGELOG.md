# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Initial repo scaffold: `data/`, `notebooks/`, `src/axonbead_ml/` package layout, `docs/`, `tests/`.
- `README.md` with project overview, roadmap, and structure explanation.
- `docs/data_card.md` template (Datasheets for Datasets convention).
- `pyproject.toml` with Week 1 dependencies; AxonBead installed as a git dependency.
- `.gitignore` covering Python, DVC, MLflow, Docker, and secrets.
- DVC initialized
- Built annotation tool (annotate.py)
- Annotated first 60 pictures (data/raw)
- Refactored shared data loading to src/
- Added classical baseline detector + point-matching evaluator
- Set up MLflow with SQLite backend

### Fixed

- Fixed missing empty-folder tracking
- switched aicsimageio → bioio to debug dependency issue (aicsimageio is discontinued)