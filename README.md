## Estadística Avanzada (Bootstrap, Permutaciones, FDR, PCA)

Pipeline práctico sin SciPy que demuestra:
- **Bootstrap** para IC de la media (percentiles).
- **Tests por permutación**: diferencia de medias y ANOVA 1 vía.
- **Corrección por FDR** (Benjamini–Hochberg).
- **PCA** con estandarización previa.
- Correlaciones de Pearson y visualizaciones básicas.

Archivo principal: `advanced_stats/advanced_stats.py` (CLI con `--demo`).

## Requisitos
- Python 3.10+
- Librerías: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (ya listadas en `requirements.txt`).

## Cómo correr

Desde `advanced_stats/`:
```bash
python advanced_stats.py --demo
# Opcional
python advanced_stats.py --demo --n 1000 --seed 7
```

Se guardan en `advanced_stats/figs/`:
- `mean_ci_by_group.csv` — media y **IC bootstrap** por grupo (A/B/C).
- `summary.csv` — diferencia de medias (B−A, C−A), p-valores de permutaciones, **FDR (BH)** y decisiones.
- `correlation_matrix.csv` — matriz de Pearson para `X1`, `X2`, `X3`, `metric`.
- `metric_hist.png` — histograma + **Normal** ajustada (µ, σ muestral).
- `pca_scree.png` — **scree plot** (varianza explicada).
- `pca_scatter.png` — PC1 vs PC2 coloreado por grupo.
- `perm_null_diff_B_vs_A.png` — nulo por permutación para B−A.

## Dataset sintético
`make_dataset(n, seed)` crea 3 grupos (A/B/C), tres features (`X1..X3`) y una métrica continua con:
- efecto lineal de `X1`, `X2`, `X3`,
- efecto de grupo,
- ruido gaussiano.

## Notas de implementación
- `mean_ci_bootstrap`: remuestreo con reemplazo (`n_boot` controlable), percentiles para IC.
- `perm_test_diff_means`: dos colas; usa barajado del pool y corrección leve `(hits+1)/(n_perm+1)`.
- `perm_test_anova`: **F** de ANOVA 1 vía calculada a mano; p-valor por permutación.
- `benjamini_hochberg`: retorna vector de rechazos y umbral crítico global.
- `PCA`: datos estandarizados con `StandardScaler`; gráficos `scree` y dispersión (PC1/PC2).

## Uso desde Python (opcional)
```python
import pandas as pd
from advanced_stats import make_dataset, mean_ci_bootstrap, perm_test_diff_means

df = make_dataset(n=600, seed=2025)
m, lo, hi = mean_ci_bootstrap(df.loc[df.group=='A', 'metric'].values, n_boot=3000)
_, p = perm_test_diff_means(df.loc[df.group=='B','metric'].values, df.loc[df.group=='A','metric'].values)
print(m, lo, hi, p)
```

## Consejos
- Si los cálculos tardan, baja `--n` o los `n_perm`/`n_boot` dentro del script.
- Reproducibilidad: fija `--seed` (por defecto 2025).
- Si `matplotlib` no está, se omiten gráficos con un mensaje en consola.
