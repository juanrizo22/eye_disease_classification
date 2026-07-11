import splitfolders

splitfolders.ratio(
    "model 3/data/raw",
    output='data',
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)