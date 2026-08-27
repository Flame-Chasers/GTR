# Dataset access statistics

This directory is isolated from the GTR+ research implementation. It contains only:

- a small GitHub Pages redirect page;
- GoatCounter integration for approximate unique-visit counting;
- an SVG line-chart generator;
- the generated chart displayed in the repository README.

It does **not** modify `train_ps.py`, `models/`, `data/`, `configs/`, checkpoints, or training/evaluation scripts.

## Statistics definition

The chart reports **daily unique visits (approx.)** to the tracked dataset link.

It does not report:

- verified Google Drive downloads;
- identifiable individual people;
- historical visits made before the tracked link was enabled.

Visitors who bypass the tracked GitHub Pages URL and open Google Drive directly are not counted.

## Files

```text
dataset_stats/
├── config.json
├── assets/
│   └── dataset-visitors.svg
├── scripts/
│   ├── build_site.py
│   └── update_chart.py
└── site/
    ├── index.html
    └── dataset/
        └── index.html
```

The only required file outside this directory is:

```text
.github/workflows/dataset-stats.yml
```

GitHub only recognizes workflow files from `.github/workflows/`.
