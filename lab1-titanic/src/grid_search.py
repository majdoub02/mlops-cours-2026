import subprocess
import sys
from itertools import product

configs = [
    *[("random_forest", ["--n_estimators", str(n), "--max_depth", str(d)])
      for n, d in product([50, 100, 200], [3, 5, 10])],
    *[("gradient_boost", ["--n_estimators", str(n), "--max_depth", str(d)])
      for n, d in product([50, 100], [3, 5])],
    *[("logistic", ["--C", str(c)]) for c in [0.01, 0.1, 1.0, 10.0]],
]

print(f"Lancement de {len(configs)} runs...")
for i, (model, params) in enumerate(configs, 1):
    cmd = [sys.executable, "src/train.py", "--model", model] + params
    print(f"[{i}/{len(configs)}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
print("Tous les runs terminés.")
