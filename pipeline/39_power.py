#!/usr/bin/env python3
"""Etapa 39: Análise de potência para as correlações parciais.

Calcula a menor magnitude |rho| detectável a 80% de potência nos cenários da
dissertação, e a potência do hold-out de Terror para |rho| = 0,30. Método
analítico via transformação z de Fisher, com a correção de variância de
Bonett & Wright (2000) para a correlação de Spearman:

    SE = sqrt(1.06 / (n - 3 - k)),   k = nº de variáveis de controle

    |rho|_min = tanh( (z_{alpha_eff/2} + z_{potência}) * SE )

O nível efetivo usa a cota conservadora alpha/m (o limiar mais estrito que o
procedimento BH pode impor, atingido quando há uma única descoberta entre as
m hipóteses). É uma cota inferior da potência real do BH: com mais descobertas
o limiar relaxa e a potência sobe.

Nota de auditoria: os valores 0,30 (Crime) e 0,21 (Drama) publicados no artigo
do SBPO não são reproduzíveis por este método (correspondem a um alpha efetivo
de ~0,015); o valor agregado 0,18 é reproduzível. A dissertação usa os valores
desta etapa.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from statistics import NormalDist

from src.utils import RESULTS, ensure_dirs

ALPHA = 0.05
M_HYP = 18          # hipóteses por par gênero × alvo (BH)
K_CONTROLS = 2      # ano + log_budget
POWER_TARGET = 0.80
SPEARMAN_VAR = 1.06  # Bonett & Wright (2000)

# (rótulo, alvo, n efetivo) — n conferidos em genre_budget_controlled.txt
SCENARIOS = [
    ("Agregado", "IMDb",           462),
    ("Agregado", "log(BoxOffice)", 440),
    ("Crime",    "IMDb",           122),
    ("Crime",    "log(BoxOffice)", 116),
    ("Drama",    "IMDb",           250),
    ("Drama",    "log(BoxOffice)", 240),
]

# potência do hold-out: (gênero, n_HO, |rho| suposto, alpha do critério)
HOLDOUT = [
    ("Suspense", 28, 0.30, 0.10),
    ("Terror",   15, 0.30, 0.10),
    ("Comédia",  45, 0.30, 0.10),
    ("Ação",     46, 0.30, 0.10),
]

ND = NormalDist()


def se_fisher(n: int, k: int = K_CONTROLS) -> float:
    return math.sqrt(SPEARMAN_VAR / (n - 3 - k))


def min_detectable_rho(n: int, alpha_eff: float, power: float = POWER_TARGET) -> float:
    z_a = ND.inv_cdf(1.0 - alpha_eff / 2.0)
    z_b = ND.inv_cdf(power)
    return math.tanh((z_a + z_b) * se_fisher(n))


def power_at(rho: float, n: int, alpha: float) -> float:
    z_a = ND.inv_cdf(1.0 - alpha / 2.0)
    return ND.cdf(abs(math.atanh(rho)) / se_fisher(n) - z_a)


def main():
    ensure_dirs()
    out = []
    out.append("=" * 74)
    out.append("  ANÁLISE DE POTÊNCIA (pipeline/39)")
    out.append("=" * 74)
    out.append(f"\n  Método: Fisher z, var {SPEARMAN_VAR} (Bonett & Wright 2000), "
               f"df = n - 3 - {K_CONTROLS}")
    out.append(f"  Potência alvo: {POWER_TARGET:.0%}; alpha efetivo (cota BH): "
               f"{ALPHA}/{M_HYP} = {ALPHA/M_HYP:.5f}, bicaudal\n")

    alpha_eff = ALPHA / M_HYP
    out.append(f"  ── Menor |rho| detectável a {POWER_TARGET:.0%} de potência ──")
    out.append(f"    {'Cenário':<10s} {'Alvo':<15s} {'n':>5s} {'|rho|_min':>10s}")
    out.append(f"    {'-'*44}")
    rows = []
    for label, target, n in SCENARIOS:
        r = min_detectable_rho(n, alpha_eff)
        rows.append({"scenario": label, "target": target, "n": n,
                     "alpha_eff": round(alpha_eff, 5),
                     "rho_min_80": round(r, 3)})
        out.append(f"    {label:<10s} {target:<15s} {n:>5d} {r:>10.3f}")

    out.append(f"\n  ── Potência no hold-out para |rho| suposto (bicaudal) ──")
    out.append(f"    {'Gênero':<10s} {'n_HO':>5s} {'|rho|':>6s} {'alpha':>6s} "
               f"{'potência':>9s}")
    out.append(f"    {'-'*40}")
    for label, n, rho, alpha in HOLDOUT:
        p = power_at(rho, n, alpha)
        rows.append({"scenario": f"HO-{label}", "target": "-", "n": n,
                     "alpha_eff": alpha, "rho_min_80": None,
                     "power_rho030": round(p, 3)})
        out.append(f"    {label:<10s} {n:>5d} {rho:>6.2f} {alpha:>6.2f} {p:>9.3f}")

    txt = "\n".join(out)
    print(txt)
    (RESULTS / "power_analysis.txt").write_text(txt + "\n", encoding="utf-8")

    import csv
    with open(RESULTS / "power_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "target", "n", "alpha_eff",
                                          "rho_min_80", "power_rho030"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n✓ Relatório: {RESULTS / 'power_analysis.txt'}")
    print(f"✓ Tabela:    {RESULTS / 'power_analysis.csv'}")


if __name__ == "__main__":
    main()
