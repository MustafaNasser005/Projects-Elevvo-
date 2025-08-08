# Movie Recommender System

This project provides a simple content-based movie recommender using TMDB datasets.

## Setup

1. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Place model files inside `assets/` (these are not tracked in git):
- `movies.pkl`
- `movie_dict.pkl`
- `similarity.pkl`

If you don't have these, generate them by running the notebook `Movie-Recommendation-System.ipynb`.

## Run the app

```bash
python app.py
```

## Notes
- Model files under `assets/` are ignored by git by design.
- Large CSVs are included for reproducibility; consider using data download scripts for production.
