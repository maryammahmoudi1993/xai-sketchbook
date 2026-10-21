# xai-sketchbook

Quick, reproducible sketches on explainable and trustworthy AI.

Each folder in `sketches/` is one question, one small experiment, and one honest result, including the parts that didn't work. These are explorations, not polished libraries.

## Sketches

| # | Folder | Question | Post |
|---|--------|----------|------|
| 01 | [sketches/01-early-career-signals](sketches/01-early-career-signals) | Which first-3-year signals predict a top-5% cited AI researcher, and does a top coauthor matter more at lower-tier institutions? | [LinkedIn](https://www.linkedin.com/in/maryam-mahmoudi-data-scientist) |

(One row per sketch, newest at the bottom.)

## Ground rules

- **Public data only.** Every sketch uses openly available datasets, linked in its README.
- **Reproducible.** Fixed seeds, pinned dependencies, no hidden steps.
- **Small.** Each sketch runs in under 5 minutes on a laptop CPU.
- **Honest.** Each README says what the sketch does not show and what didn't work.

## Run a sketch

```bash
git clone https://github.com/maryammahmoudi1993/xai-sketchbook.git
cd xai-sketchbook/sketches/01-early-career-signals
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Repository layout

```
xai-sketchbook/
├── README.md
├── LICENSE
├── common/
│   ├── SKETCH_README_TEMPLATE.md   # copy into each new sketch folder as README.md
│   └── ...                         # shared helpers (plotting, data loading)
└── sketches/
    └── NN-short-topic-name/
        ├── README.md
        ├── main.py                 # or notebook.ipynb
        ├── requirements.txt
        └── figures/
```

## About

I'm a PhD researcher working on explainable AI for medical and trustworthy AI applications.

- LinkedIn: <link>
- X: <link>
- Website: <link>

## License

MIT, see [LICENSE](LICENSE).
