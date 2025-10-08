#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Estadística "avanzada" sin SciPy: bootstrap, tests por permutación, FDR y PCA.
# Librerías: numpy, pandas, matplotlib, scikit-learn (todas ya usadas en el repo).
# Estilo pragmático y sin sobre-ingeniería.

from pathlib import Path
import argparse
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -------------------------- utils --------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mean_ci_bootstrap(x: np.ndarray, n_boot: int = 5000, ci: float = 95, seed: int = 2025) -> tuple[float, float, float]:
    """IC bootstrap para la media (percentiles)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boots = np.empty(n_boot, dtype=float)
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(x[idx]))
    alpha = (100 - ci) / 100 / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return float(np.mean(x)), lo, hi


def perm_test_diff_means(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 2025) -> tuple[float, float]:
    """Test por permutación para diferencia de medias (two-sided)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = float(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        stat = float(np.mean(pool[:na]) - np.mean(pool[na:]))
        if abs(stat) >= abs(obs):
            hits += 1
    pval = (hits + 1) / (n_perm + 1)  # corrección leve
    return obs, pval


def f_stat_oneway(groups: list[np.ndarray]) -> float:
    """F de ANOVA de 1 vía (para usar con permutaciones)."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    n_groups = len(groups)
    n_total = sum(len(g) for g in groups)
    overall = float(np.mean(np.concatenate(groups)))
    ss_between = sum(len(g) * (float(np.mean(g)) - overall) ** 2 for g in groups)
    ss_within = sum(float(np.sum((g - float(np.mean(g))) ** 2)) for g in groups)
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    return ms_between / ms_within if ms_within > 0 else math.inf


def perm_test_anova(groups: list[np.ndarray], n_perm: int = 5000, seed: int = 2025) -> tuple[float, float]:
    """ANOVA 1 vía por permutación (p-valor >= F observado)."""
    rng = np.random.default_rng(seed)
    obs = f_stat_oneway(groups)
    pool = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        parts = []
        i = 0
        for s in sizes:
            parts.append(pool[i:i + s])
            i += s
        stat = f_stat_oneway(parts)
        if stat >= obs:
            hits += 1
    pval = (hits + 1) / (n_perm + 1)
    return obs, pval


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """BH-FDR: retorna vector de rechazos y umbral crítico."""
    p = np.array(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    crit = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= crit
    k = np.max(np.where(passed)[0]) + 1 if np.any(passed) else 0
    thresh = crit[k - 1] if k > 0 else 0.0
    reject = p <= thresh
    return reject, float(thresh)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """PDF normal (sin SciPy)."""
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------------- dataset sintético ----------------------

def make_dataset(n: int = 600, seed: int = 2025) -> pd.DataFrame:
    """
    Dataset con 3 grupos (A/B/C), 3 features y una métrica continua.
    - 'metric' tiene efecto de grupo + efecto lineal de X1, X2 y ruido.
    """
    rng = np.random.default_rng(seed)

    groups = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25])
    X1 = rng.normal(0, 1, size=n)
    X2 = rng.normal(0, 1, size=n)
    X3 = rng.normal(0, 1, size=n)

    base = 50.0 + 4.0 * X1 - 2.5 * X2 + 0.7 * X3
    group_effect = np.where(groups == "A", 0.0, np.where(groups == "B", 1.8, -0.7))
    noise = rng.normal(0, 2.0, size=n)

    metric = base + group_effect + noise

    df = pd.DataFrame({"group": groups, "X1": X1, "X2": X2, "X3": X3, "metric": metric})
    return df


# ------------------------- gráficos -------------------------

def plot_hist_with_gaussian(x: np.ndarray, outpng: Path, title: str) -> None:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    plt.figure()
    plt.hist(x, bins=30, density=True, alpha=0.8)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    plt.plot(xs, normal_pdf(xs, mu, sigma))
    plt.title(title)
    plt.xlabel("valor")
    plt.ylabel("densidad")
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close()


def plot_perm_null(null_stats: np.ndarray, observed: float, outpng: Path, title: str) -> None:
    plt.figure()
    plt.hist(null_stats, bins=40, alpha=0.85)
    plt.axvline(observed, linestyle="--")
    plt.title(title)
    plt.xlabel("estadística (nulo por permutación)")
    plt.ylabel("frecuencia")
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close()


def plot_scree(ev_ratio: np.ndarray, outpng: Path) -> None:
    plt.figure()
    plt.plot(range(1, len(ev_ratio) + 1), ev_ratio, marker="o")
    plt.xlabel("componente")
    plt.ylabel("varianza explicada")
    plt.title("Scree plot (PCA)")
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close()


def plot_pca_scatter(X_std: np.ndarray, group: np.ndarray, outpng: Path) -> None:
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_std)
    plt.figure()
    for g in np.unique(group):
        idx = (group == g)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=14, label=str(g))
    plt.legend(title="group")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA scatter")
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close()


# ------------------- pipeline de análisis -------------------

def run_demo(n: int, seed: int, outdir: Path) -> None:
    ensure_dir(outdir)
    df = make_dataset(n=n, seed=seed)

    # 1) IC bootstrap por grupo (media de 'metric')
    rows_ci = []
    for g, sub in df.groupby("group"):
        mean_, lo, hi = mean_ci_bootstrap(sub["metric"].to_numpy(), n_boot=3000, ci=95, seed=seed)
        rows_ci.append({"group": g, "metric_mean": mean_, "ci_lo": lo, "ci_hi": hi})
    df_ci = pd.DataFrame(rows_ci)
    df_ci.to_csv(outdir / "mean_ci_by_group.csv", index=False)

    # 2) Test por permutación: diff de medias B vs A y C vs A
    obs_BA, p_BA = perm_test_diff_means(
        df.loc[df.group == "B", "metric"].to_numpy(),
        df.loc[df.group == "A", "metric"].to_numpy(),
        n_perm=4000, seed=seed,
    )
    obs_CA, p_CA = perm_test_diff_means(
        df.loc[df.group == "C", "metric"].to_numpy(),
        df.loc[df.group == "A", "metric"].to_numpy(),
        n_perm=4000, seed=seed,
    )

    # 3) ANOVA 1 vía por permutación (A/B/C)
    gA = df.loc[df.group == "A", "metric"].to_numpy()
    gB = df.loc[df.group == "B", "metric"].to_numpy()
    gC = df.loc[df.group == "C", "metric"].to_numpy()
    F_obs, p_anova = perm_test_anova([gA, gB, gC], n_perm=4000, seed=seed)

    # 4) Corrección por FDR (BH) para los p-values de B vs A y C vs A
    reject, thr = benjamini_hochberg([p_BA, p_CA], alpha=0.05)

    # 5) Correlación entre features (Pearson)
    corr = df[["X1", "X2", "X3", "metric"]].corr(method="pearson")
    corr.to_csv(outdir / "correlation_matrix.csv")

    # 6) PCA (estandarizando X1..X3 y metric)
    X = df[["X1", "X2", "X3", "metric"]].to_numpy()
    X_std = StandardScaler().fit_transform(X)
    pca_all = PCA().fit(X_std)
    ev_ratio = pca_all.explained_variance_ratio_

    # 7) Gráficos
    plot_hist_with_gaussian(df["metric"].to_numpy(), outdir / "metric_hist.png", "Distribución de 'metric'")
    plot_scree(ev_ratio, outdir / "pca_scree.png")
    plot_pca_scatter(X_std, df["group"].to_numpy(), outdir / "pca_scatter.png")

    # 8) Guardar resumen general
    summary = {
        "n_rows": len(df),
        "perm_diff_B_vs_A": obs_BA,
        "pvalue_B_vs_A": p_BA,
        "perm_diff_C_vs_A": obs_CA,
        "pvalue_C_vs_A": p_CA,
        "anova_F_obs": F_obs,
        "anova_pvalue": p_anova,
        "fdr_threshold": thr,
        "reject_B_vs_A": bool(reject[0]),
        "reject_C_vs_A": bool(reject[1]),
    }
    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)

    # 9) Null hist (opcional rápido): guardamos sólo para diff B vs A
    #    Para no recalcular, repetimos las permutaciones una vez más pero guardando.
    rng = np.random.default_rng(seed)
    pool = np.concatenate([gB, gA])
    na = len(gB)
    null_stats = np.empty(2000, dtype=float)
    for i in range(2000):
        rng.shuffle(pool)
        null_stats[i] = float(np.mean(pool[:na]) - np.mean(pool[na:]))
    plot_perm_null(null_stats, obs_BA, outdir / "perm_null_diff_B_vs_A.png",
                   "Null por permutación: diff medias (B - A)")

    # 10) Imprimir un vistazo para consola
    print("[demo] CI por grupo:\n", df_ci)
    print("\n[demo] perm diff (B-A):", obs_BA, "p=", p_BA, " | (C-A):", obs_CA, "p=", p_CA)
    print("[demo] anova F:", F_obs, " p=", p_anova)
    print("[demo] FDR BH threshold:", thr, " reject:", reject)
    print("[demo] corr pearson:\n", corr.round(3))
    print("[OK] Figuras y CSVs en ->", outdir)


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estadística avanzada: bootstrap, permutaciones, FDR y PCA (demo)")
    p.add_argument("--n", type=int, default=600, help="n filas del dataset sintético")
    p.add_argument("--seed", type=int, default=2025, help="semilla RNG")
    p.add_argument("--demo", action="store_true", help="ejecutar pipeline completo y guardar artefactos")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(__file__).parent / "figs"
    ensure_dir(outdir)

    if not args.demo:
        print("Nada que hacer. Usa --demo (y parámetros opcionales --n/--seed).")
        return

    t0 = time.time()
    run_demo(n=args.n, seed=args.seed, outdir=outdir)
    print(f"[tiempo] {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
