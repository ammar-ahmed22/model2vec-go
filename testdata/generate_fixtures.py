"""Regenerate the Python ground-truth embedding fixtures.

Run from the repo root:

    python testdata/generate_fixtures.py

Requires the upstream `model2vec` Python package:

    pip install model2vec
"""
from model2vec import StaticModel
import json

model = StaticModel.from_pretrained("testdata/test-model-float32")
short = model.encode(["hello world"]).tolist()
long_text = " ".join(["hello"] * 1000)
long = model.encode([long_text]).tolist()

with open("testdata/embeddings_short.json", "w") as f:
    json.dump(short, f)
with open("testdata/embeddings_long.json", "w") as f:
    json.dump(long, f)

vq_model = StaticModel.from_pretrained("testdata/test-model-vocab-quantized")
vq = vq_model.encode([long_text]).tolist()
with open("testdata/embeddings_vocab_quantized.json", "w") as f:
    json.dump(vq, f)
