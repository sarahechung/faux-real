Detect whether an image is AI-generated or human-made using EfficientNet-B0 (transfer learning).

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Place your dataset in `data/` with subfolders `ai/` and `human/`.

3. Train the model:
```bash
python -m src.train --data-dir data --epochs 5
```

4. Evaluate the model:
```bash
python -m src.eval --data-dir data --checkpoint models/best_model.pt
```

## Key Metrics
The training script prints accuracy, precision, recall, and F1.  
Success = F1 >= 0.80 and Precision >= 0.80.
