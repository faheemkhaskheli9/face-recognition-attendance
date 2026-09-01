# Sample images

`alice_1.npy`, `alice_2.npy`, `bob_1.npy` are small synthetic RGB arrays
(64x64x3, deterministic random noise) used to exercise the enrollment pipeline
and the VS Code launch configs offline. They are **not** real faces.

Regenerate them with:

```bash
python - <<'PY'
import numpy as np
for i, name in enumerate(["alice_1", "alice_2", "bob_1"]):
    seed = 1 if name.startswith("alice") else 2
    arr = (np.random.default_rng(seed * 10 + i).random((64, 64, 3)) * 255).astype("uint8")
    np.save(f"samples/{name}.npy", arr)
PY
```

For a real run, point `--images` at ordinary photo files (Pillow reads them) and
use `--backend dlib`.
