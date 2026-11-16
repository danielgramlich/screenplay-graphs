"""
Camada 3 — Scoring aplicado.

Transforma as 44 métricas dos grafos emocionais em:
  1. Índices interpretativos (estabilidade, tensão, recuperação, etc.)
  2. Perfil narrativo categórico
  3. Estimativa de IMDb
  4. Estimativa de ROI / revenue tendency
  5. Tendência comercial
  6. Explicação textual

PRINCÍPIO DE DESIGN:
  - Não usa modelo black-box
  - Cada score é derivado de regras transparentes
  - Os pesos vêm dos achados empíricos dos scripts 08-18
  - A lógica é interpretável e auditável

BASE EMPÍRICA (resumo dos achados):
  - Estabilidade (autoloop) → favorece IMDb (β=+1.19 global, +7.80 Horror)
  - Tensão (pr_neg) → prejudica IMDb, favorece box office
  - Recuperação (neg→pos, pr_pos) → independente de tensão
  - Tensão não resolvida → prejudica IMDb, favorece box office, U invertido para revenue
  - Mudança no final → Horror penaliza positividade gratuita; Drama penaliza recuperação fácil
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# ETAPA A — ÍNDICES INTERPRETATIVOS
# ═══════════════════════════════════════════════════════════════════

def compute_indices(metrics: dict) -> dict:
    """Transforma as 44 métricas em 5 índices interpretativos.

    Args:
        metrics: dict com as métricas do grafo (chaves gv5_*, gv3_*, gv3a3_*)

    Returns:
        dict com stability, tension, recovery, unresolved, ending_shift
        e sub-componentes para auditoria.
    """

    # ── 1. STABILITY (Estabilidade Emocional) ──
    # Base empírica: autoloop alto → melhor IMDb (β=+1.19 global, +7.80 Horror)
    # Fontes: gv3_autoloop (filme inteiro) e gv3a3_autoloop (último terço)
    autoloop_full = metrics.get("gv3_autoloop", 0.39)     # média corpus: 0.389
    autoloop_lt = metrics.get("gv3a3_autoloop", 0.39)     # média corpus: 0.389
    stability = (autoloop_full * 0.6 + autoloop_lt * 0.4)  # peso maior no filme inteiro
    # Normalizar para 0-1 (autoloop típico: 0.25-0.55)
    stability_score = np.clip((stability - 0.25) / 0.30, 0, 1)

    # ── 2. TENSION (Tensão) ──
    # Base empírica: pr_neg alto → prejudica IMDb (ρ=-0.148), favorece box office
    # pos_to_neg alto → mais intensidade emocional
    pr_neg = metrics.get("gv3_pr_neg", 0.31)              # média corpus: 0.312
    pos_to_neg = metrics.get("gv3_pos_to_neg", 0.08)      # média corpus: 0.084
    # Combinar: pr_neg pesa mais (sinal mais forte)
    tension_raw = pr_neg * 0.7 + pos_to_neg * 3.0 * 0.3   # escalar pos_to_neg
    # Normalizar para 0-1 (range típico: 0.15-0.50)
    tension_score = np.clip((tension_raw - 0.15) / 0.35, 0, 1)

    # ── 3. RECOVERY (Recuperação) ──
    # Base empírica: independente de tensão (ρ=+0.02)
    # neg_to_pos → alívio narrativo; pr_pos → positividade dominante
    pr_pos = metrics.get("gv3_pr_pos", 0.34)              # média corpus: 0.335
    neg_to_pos = metrics.get("gv3_neg_to_pos", 0.08)      # média corpus: 0.085
    recovery_raw = pr_pos * 0.7 + neg_to_pos * 3.0 * 0.3
    recovery_score = np.clip((recovery_raw - 0.15) / 0.35, 0, 1)

    # ── 4. UNRESOLVED TENSION (Tensão Não Resolvida) ──
    # Base empírica: tensão - recuperação
    # Positivo = mais tensão que resolução → prejudica IMDb, favorece box office
    # U invertido para revenue (Q3 ótimo, Q4 cai)
    unresolved = tension_score - recovery_score  # range: -1 a +1

    # ── 5. ENDING SHIFT (Mudança no Final) ──
    # Base empírica: Horror penaliza Δ_pos, Drama penaliza Δ_recovery
    # Calculado como magnitude da mudança entre filme inteiro e último terço
    pr_neg_lt = metrics.get("gv3a3_pr_neg", pr_neg)
    pr_pos_lt = metrics.get("gv3a3_pr_pos", pr_pos)
    neg_to_pos_lt = metrics.get("gv3a3_neg_to_pos", neg_to_pos)

    delta_neg = pr_neg_lt - pr_neg       # positivo = final mais negativo
    delta_pos = pr_pos_lt - pr_pos       # positivo = final mais positivo
    delta_recovery = neg_to_pos_lt - neg_to_pos  # positivo = mais recuperação no final

    # Magnitude total da mudança (0 = consistente, alto = muda muito)
    ending_shift = abs(delta_neg) + abs(delta_pos) + abs(delta_recovery)
    # Normalizar pelo P90 do corpus (0.233) em vez de divisor fixo 0.15
    # Isso evita saturação: com 0.15, 33% saturavam; com P90, apenas 10%
    ending_shift_score = np.clip(ending_shift / 0.233, 0, 1)

    return {
        # Scores principais (0-1)
        "stability": round(float(stability_score), 3),
        "tension": round(float(tension_score), 3),
        "recovery": round(float(recovery_score), 3),
        "unresolved": round(float(unresolved), 3),
        "ending_shift": round(float(ending_shift_score), 3),

        # Sub-componentes para auditoria
        "autoloop_full": round(float(autoloop_full), 4),
        "autoloop_lt": round(float(autoloop_lt), 4),
        "pr_neg": round(float(pr_neg), 4),
        "pr_pos": round(float(pr_pos), 4),
        "delta_neg": round(float(delta_neg), 4),
        "delta_pos": round(float(delta_pos), 4),
        "delta_recovery": round(float(delta_recovery), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# ETAPA B — PERFIL NARRATIVO
# ═══════════════════════════════════════════════════════════════════

# Definições dos perfis com regras transparentes
PROFILES = {
    "Consistente": {
        "desc": "Alta estabilidade, pouca oscilação, tom controlado",
        "rule": lambda i: i["stability"] > 0.55 and abs(i["unresolved"]) < 0.15,
    },
    "Prestígio": {
        "desc": "Estabilidade alta com tensão moderada e pouca mudança no final",
        "rule": lambda i: i["stability"] > 0.45 and 0.25 < i["tension"] < 0.55 and i["ending_shift"] < 0.4,
    },
    "Sombrio controlado": {
        "desc": "Tensão alta mas estável, sem recuperação fácil",
        "rule": lambda i: i["tension"] > 0.55 and i["stability"] > 0.40 and i["recovery"] < 0.40,
    },
    "Catártico": {
        "desc": "Tensão alta seguida de recuperação forte",
        "rule": lambda i: i["tension"] > 0.45 and i["recovery"] > 0.55,
    },
    "Comercial": {
        "desc": "Alta intensidade emocional, muita oscilação",
        "rule": lambda i: i["tension"] > 0.50 and i["stability"] < 0.40,
    },
    "Intenso": {
        "desc": "Tensão muito alta com pouca recuperação",
        "rule": lambda i: i["tension"] > 0.65 and i["recovery"] < 0.35,
    },
    "Desequilibrado": {
        "desc": "Muita mudança no final, inconsistência tonal",
        "rule": lambda i: i["ending_shift"] > 0.6 or (i["stability"] < 0.30),
    },
}


def classify_profile(indices: dict) -> dict:
    """Classifica o perfil narrativo do filme.

    Testa cada perfil por ordem de especificidade.
    Se nenhum encaixa, retorna 'Neutro'.
    """
    for name, profile in PROFILES.items():
        if profile["rule"](indices):
            return {"profile": name, "description": profile["desc"]}

    return {"profile": "Neutro", "description": "Sem padrão narrativo dominante"}


# ═══════════════════════════════════════════════════════════════════
# ETAPA C — ESTIMATIVA DE IMDb
# ═══════════════════════════════════════════════════════════════════

# Medianas observadas por gênero no corpus
GENRE_MEDIANS_IMDB = {
    "Thriller": 6.7, "Horror": 6.2, "Comedy": 6.8, "Drama": 7.2,
    "Action": 6.6, "Crime": 6.9, "Romance": 7.0, "Adventure": 7.0,
    "default": 6.9,
}

# Pesos empíricos derivados dos modelos OLS (scripts 14, 17, 18)
# Normalizados para contribuição em pontos de IMDb
IMDB_WEIGHTS = {
    "stability":  +0.40,   # β=+1.19 global, escalado
    "tension":    -0.25,   # pr_neg prejudica IMDb
    "unresolved": -0.15,   # tensão não resolvida prejudica IMDb (ρ=-0.093)
    "ending_shift": -0.10, # mudanças bruscas no final penalizam
}

# Ajustes por gênero (multiplicadores sobre o peso base)
GENRE_IMDB_MODIFIERS = {
    "Horror":   {"stability": 2.5},   # β=+7.80 em Horror
    "Thriller": {"tension": 1.5, "stability": 0.8},  # tensão mais importante
    "Comedy":   {"tension": 1.3},     # negatividade prejudica mais
    "Drama":    {"stability": 0.5, "tension": 0.5},   # efeitos diluídos
}


def estimate_imdb(indices: dict, genre: str = "default") -> dict:
    """Estima faixa de IMDb baseada nos índices narrativos.

    Não é previsão exata — é estimativa baseada em padrões empíricos.

    Returns:
        dict com estimate, range_low, range_high, confidence, basis
    """
    # Âncora: mediana do gênero
    primary_genre = _extract_primary_genre(genre)
    anchor = GENRE_MEDIANS_IMDB.get(primary_genre, GENRE_MEDIANS_IMDB["default"])

    # Calcular ajuste
    modifiers = GENRE_IMDB_MODIFIERS.get(primary_genre, {})
    adjustment = 0.0
    contributions = []

    for feature, base_weight in IMDB_WEIGHTS.items():
        value = indices.get(feature, 0)
        genre_mult = modifiers.get(feature, 1.0)
        weight = base_weight * genre_mult

        # Centralizar: desvio em relação ao neutro (0.5 para scores 0-1, 0 para unresolved)
        if feature == "unresolved":
            deviation = value  # já centrado em 0
        else:
            deviation = value - 0.5

        contribution = weight * deviation
        adjustment += contribution
        contributions.append({
            "feature": feature,
            "value": round(value, 3),
            "weight": round(weight, 3),
            "contribution": round(contribution, 3),
        })

    estimate = anchor + adjustment

    # ── Penalização seletiva para cauda baixa ──
    # Só aplica quando MÚLTIPLOS sinais extremos coexistem.
    # Base empírica: filmes com IMDb<5 têm stab=0.35, tens=0.50
    # Filmes normais com 1 sinal ruim NÃO devem ser penalizados.
    low_penalty = 0.0
    stab = indices.get("stability", 0.5)
    tens = indices.get("tension", 0.5)
    unres = indices.get("unresolved", 0)
    shift = indices.get("ending_shift", 0)

    # Contar sinais extremos (cada um abaixo P10 ou acima P90)
    bad_signals = 0
    if stab < 0.20:   bad_signals += 1  # P10 estabilidade
    if tens > 0.71:    bad_signals += 1  # P90 tensão
    if unres > 0.41:   bad_signals += 1  # P90 unresolved
    if shift > 0.80:   bad_signals += 1  # P80 ending shift (novo)

    # Só penalizar se 2+ sinais extremos coexistem
    if bad_signals >= 3:
        low_penalty = -0.5
    elif bad_signals >= 2:
        low_penalty = -0.3

    estimate = estimate + low_penalty
    estimate = np.clip(estimate, 2.0, 9.5)

    # Incerteza: ±0.7 base, mais se ending_shift alto
    uncertainty = 0.7 + indices.get("ending_shift", 0) * 0.3
    range_low = round(max(2.0, estimate - uncertainty), 1)
    range_high = round(min(9.5, estimate + uncertainty), 1)

    # Confiança baseada em quão típico é o perfil
    conf = _compute_confidence(indices, primary_genre)

    return {
        "estimate": round(float(estimate), 1),
        "range": [range_low, range_high],
        "anchor": anchor,
        "adjustment": round(float(adjustment), 3),
        "confidence": conf,
        "contributions": sorted(contributions, key=lambda c: abs(c["contribution"]), reverse=True),
    }


# ═══════════════════════════════════════════════════════════════════
# ETAPA D — ESTIMATIVA DE ROI / REVENUE
# ═══════════════════════════════════════════════════════════════════

# Base empírica: revenue multiple tem ótimo em Q3 de tensão
# Q1=0.30, Q2=0.43, Q3=0.46, Q4=0.22 (log_revenue_multiple)
ROI_WEIGHTS = {
    "stability":  +0.15,   # leve efeito positivo
    "tension":     0.0,    # efeito é não-linear, tratado separado
    "unresolved":  0.0,    # U invertido, tratado separado
    "recovery":   +0.10,   # alguma recuperação ajuda
}


def estimate_roi(indices: dict, genre: str = "default") -> dict:
    """Estima tendência de retorno financeiro (revenue multiple).

    Usa lógica de U invertido: tensão moderada é ótima.
    """
    tension = indices.get("tension", 0.5)
    unresolved = indices.get("unresolved", 0)
    stability = indices.get("stability", 0.5)
    recovery = indices.get("recovery", 0.5)

    # Componente de tensão: U invertido centrado em 0.45
    # Penalizar extremos (muito baixo ou muito alto)
    tension_opt = 0.45
    tension_penalty = -((tension - tension_opt) ** 2) * 2.0

    # Componente de unresolved: U invertido centrado em 0.05
    unresolved_opt = 0.05
    unresolved_penalty = -((unresolved - unresolved_opt) ** 2) * 1.5

    # Score bruto (0 = neutro)
    score = (
        tension_penalty
        + unresolved_penalty
        + ROI_WEIGHTS["stability"] * (stability - 0.5)
        + ROI_WEIGHTS["recovery"] * (recovery - 0.5)
    )

    # Converter para categorias
    if score > 0.05:
        tendency = "favorável"
        level = "alto"
    elif score > -0.05:
        tendency = "neutro"
        level = "médio"
    else:
        tendency = "desfavorável"
        level = "baixo"

    return {
        "score": round(float(score), 3),
        "tendency": tendency,
        "level": level,
        "tension_contribution": round(float(tension_penalty), 3),
        "unresolved_contribution": round(float(unresolved_penalty), 3),
    }


# ═══════════════════════════════════════════════════════════════════
# ETAPA E — TENDÊNCIA COMERCIAL
# ═══════════════════════════════════════════════════════════════════

# Base empírica: box office favorece tensão alta, estabilidade baixa
# Budget domina (R²=0.30 só com controles), narrativa é secundária
BO_WEIGHTS = {
    "tension":    +0.30,   # mais tensão → mais bilheteira
    "stability":  -0.10,   # filmes instáveis vendem mais (paradoxo IMDb)
    "recovery":   +0.05,   # alguma recuperação ajuda
}


def estimate_box_office_tendency(indices: dict, genre: str = "default") -> dict:
    """Estima tendência comercial baseada na estrutura narrativa.

    IMPORTANTE: budget domina box office. Esta estimativa captura apenas
    a componente narrativa, que é secundária.
    """
    score = 0.0
    for feature, weight in BO_WEIGHTS.items():
        value = indices.get(feature, 0.5)
        score += weight * (value - 0.5)

    if score > 0.05:
        tendency = "alto"
    elif score > -0.03:
        tendency = "médio"
    else:
        tendency = "baixo"

    return {
        "score": round(float(score), 3),
        "tendency": tendency,
        "caveat": "Budget e marketing dominam box office. Esta estimativa reflete apenas a componente narrativa.",
    }


# ═══════════════════════════════════════════════════════════════════
# ETAPA F — EXPLICAÇÃO TEXTUAL
# ═══════════════════════════════════════════════════════════════════

def generate_explanation(indices: dict, imdb_est: dict, roi_est: dict,
                         bo_est: dict, profile: dict, genre: str = "default",
                         quality: dict = None) -> list[str]:
    """Gera explicação textual interpretável dos resultados.

    Returns:
        Lista de frases explicativas.
    """
    primary = _extract_primary_genre(genre)
    lines = []

    # Perfil
    lines.append(f"Perfil narrativo: {profile['profile']} — {profile['description']}.")

    # Estabilidade
    stab = indices["stability"]
    if stab > 0.55:
        lines.append("O roteiro apresenta alta estabilidade emocional, padrão associado "
                      "a melhor avaliação no IMDb, especialmente em filmes de tensão.")
    elif stab < 0.35:
        lines.append("O roteiro tem baixa estabilidade emocional — mudanças frequentes "
                      "de tom entre cenas. Isso tende a prejudicar a avaliação de qualidade "
                      "mas pode favorecer interesse comercial.")

    # Tensão
    tens = indices["tension"]
    if tens > 0.60:
        lines.append("A tensão narrativa é elevada. Isso pode favorecer bilheteira "
                      "mas tende a reduzir a avaliação IMDb.")
        if primary in ("Thriller", "Horror"):
            lines.append(f"Em {primary}, tensão controlada é essencial — "
                          "excesso pode prejudicar.")
    elif tens < 0.30:
        lines.append("A tensão narrativa é baixa. Filmes com pouca intensidade emocional "
                      "tendem a ter avaliação moderada e menor apelo comercial.")

    # Unresolved
    unres = indices["unresolved"]
    if unres > 0.20:
        lines.append("Há tensão significativamente não resolvida — o filme acumula "
                      "mais negatividade do que resolve. Isso tende a prejudicar "
                      "IMDb mas pode favorecer impacto comercial.")
    elif unres < -0.20:
        lines.append("O roteiro resolve mais do que tensiona — predomínio de recuperação "
                      "emocional. Isso é típico de filmes com finais resolutivos.")

    # Ending shift
    shift = indices["ending_shift"]
    if shift > 0.50:
        lines.append("O tom emocional muda significativamente no último terço. "
                      "Mudanças bruscas no final tendem a dividir a avaliação.")
        if primary == "Horror":
            lines.append("Em Horror, finais que mudam para positivo sem justificativa "
                          "narrativa tendem a ser pior avaliados.")

    # Estimativas
    imdb_range = imdb_est["range"]
    lines.append(f"Estimativa de IMDb: {imdb_range[0]}–{imdb_range[1]} "
                 f"(âncora do gênero: {imdb_est['anchor']}, confiança: {imdb_est['confidence']}).")

    lines.append(f"Tendência comercial (narrativa): {bo_est['tendency']}. "
                 f"{bo_est['caveat']}")

    lines.append(f"Tendência de retorno: {roi_est['tendency']}.")

    # Contribuições principais
    top_contrib = imdb_est["contributions"][:3]
    if top_contrib:
        contrib_strs = []
        for c in top_contrib:
            direction = "favorece" if c["contribution"] > 0 else "reduz"
            feature_names = {
                "stability": "estabilidade emocional",
                "tension": "nível de tensão",
                "unresolved": "tensão não resolvida",
                "ending_shift": "mudança no final",
            }
            name = feature_names.get(c["feature"], c["feature"])
            contrib_strs.append(f"{name} {direction} ({c['contribution']:+.2f})")
        lines.append("Fatores principais: " + "; ".join(contrib_strs) + ".")

    # Quality score
    if quality:
        q = quality["score"]
        label = quality["label"]
        if q >= 0.70:
            lines.append(f"Confiabilidade: {label} ({q:.0%}). A estimativa apresenta "
                          "boa confiabilidade — estrutura emocional consistente e dados suficientes.")
        elif q >= 0.50:
            lines.append(f"Confiabilidade: {label} ({q:.0%}). A estimativa tem confiabilidade "
                          "moderada — alguns componentes podem estar incompletos ou inconsistentes.")
        else:
            lines.append(f"Confiabilidade: {label} ({q:.0%}). A estimativa deve ser interpretada "
                          "com cautela — dados insuficientes ou estrutura emocional inconsistente.")

    return lines


# ═══════════════════════════════════════════════════════════════════
# ETAPA G — QUALITY SCORE (confiabilidade da estimativa)
# ═══════════════════════════════════════════════════════════════════
#
# O quality_score mede quão ROBUSTA é a evidência narrativa disponível.
# NÃO mede qualidade do filme nem sucesso esperado.
# Score alto = "o sistema tem dados suficientes e coerentes para estimar"
# Score baixo = "a estimativa deve ser interpretada com cautela"

# Pesos de cada componente (facilmente ajustáveis)
QUALITY_WEIGHTS = {
    "scene_coverage":    0.30,  # quantidade de dados disponíveis
    "ending_coverage":   0.20,  # dados do último terço
    "emotional_richness": 0.25, # variabilidade emocional (nem plano nem caótico)
    "structural_consistency": 0.25,  # coerência interna dos sinais
}


def compute_quality_score(metrics: dict, indices: dict) -> dict:
    """Calcula score de confiabilidade da estimativa.

    Args:
        metrics: dict com as métricas brutas do grafo (gv5_*, gv3_*, gv3a3_*)
        indices: dict com os índices já computados (stability, tension, etc.)

    Returns:
        dict com score (0-1), label, e componentes para auditoria.
    """

    # ── 1. SCENE COVERAGE ──
    # Baseado na entropia V5 como proxy de quantidade de dados.
    # Filmes com poucas cenas geram grafos com poucos estados ativos,
    # resultando em entropia muito baixa.
    # Distribuição corpus: P10=2.8, P50=3.6, P90=4.1
    entropy_v5 = metrics.get("gv5_entropy", 0)
    # Escalar usando range empírico: P10=2.8, P50=3.6, P90=4.1
    # Centrar no P50 para que metade fique abaixo de 0.5
    scene_score = np.clip((entropy_v5 - 2.5) / 1.8, 0, 1)

    # ── 2. ENDING COVERAGE ──
    # Verificar se as métricas do último terço existem e são informativas.
    # Se gv3a3_entropy é muito baixa ou missing, o último terço tem poucos dados.
    entropy_lt = metrics.get("gv3a3_entropy", None)
    if entropy_lt is None or np.isnan(entropy_lt):
        ending_score = 0.0  # sem dados do último terço
    else:
        # Escalar para discriminar: P10=2.1, P50=2.9
        ending_score = np.clip((entropy_lt - 2.0) / 1.2, 0, 1)

    # ── 3. EMOTIONAL RICHNESS ──
    # A curva emocional precisa ter variabilidade suficiente para ser informativa,
    # mas não tanta que seja ruído.
    # Proxy: reciprocidade V3 (mede oscilação entre estados)
    # Distribuição corpus: P10=0.72, P50=0.91, P90=0.98
    # Ótimo = perto da mediana. Muito baixo = sem variação. Muito alto = caos.
    reciprocity = metrics.get("gv3_reciprocity", 0.9)
    # Penalizar extremos: ótimo em 0.85
    richness_optimal = 0.88
    richness_deviation = abs(reciprocity - richness_optimal)
    # Mais agressivo: desvios > 0.15 já penalizam forte
    richness_score = np.clip(1.0 - richness_deviation / 0.20, 0, 1)

    # ── 4. STRUCTURAL CONSISTENCY ──
    # Verificar se os sinais internos são coerentes.
    # Incoerências possíveis:
    #   a) Tensão alta + estabilidade alta = contradição (raramente natural)
    #   b) Autoloop V5 muito diferente de autoloop V3a3 = inconsistência entre partes
    stability = indices.get("stability", 0.5)
    tension = indices.get("tension", 0.5)
    autoloop_full = indices.get("autoloop_full", 0.39)
    autoloop_lt = indices.get("autoloop_lt", 0.39)

    consistency_penalties = 0.0

    # a) Tensão alta + estabilidade alta (contradição)
    if tension > 0.55 and stability > 0.55:
        consistency_penalties += 0.25
    # b) Tensão alta + recuperação alta (contradição diferente)
    recovery = indices.get("recovery", 0.5)
    if tension > 0.55 and recovery > 0.55:
        consistency_penalties += 0.15
    # c) Divergência grande entre autoloop filme inteiro e último terço
    autoloop_diff = abs(autoloop_full - autoloop_lt)
    if autoloop_diff > 0.10:
        consistency_penalties += 0.25
    elif autoloop_diff > 0.05:
        consistency_penalties += 0.10

    consistency_score = np.clip(1.0 - consistency_penalties, 0, 1)

    # ── COMBINAR ──
    quality = (
        QUALITY_WEIGHTS["scene_coverage"] * scene_score
        + QUALITY_WEIGHTS["ending_coverage"] * ending_score
        + QUALITY_WEIGHTS["emotional_richness"] * richness_score
        + QUALITY_WEIGHTS["structural_consistency"] * consistency_score
    )
    quality = round(float(np.clip(quality, 0, 1)), 3)

    return {
        "score": quality,
        "label": quality_label(quality),
        "components": {
            "scene_coverage": round(float(scene_score), 3),
            "ending_coverage": round(float(ending_score), 3),
            "emotional_richness": round(float(richness_score), 3),
            "structural_consistency": round(float(consistency_score), 3),
        },
    }


def quality_label(score: float) -> str:
    """Converte quality_score numérico em label interpretável."""
    if score >= 0.80:
        return "alta confiança"
    if score >= 0.60:
        return "confiável"
    if score >= 0.40:
        return "moderada"
    return "baixa confiança"


# ═══════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════

def score_film(metrics: dict, genre: str = "default") -> dict:
    """Pipeline completo de scoring para um filme.

    Args:
        metrics: dict com as 44 métricas do grafo
        genre: string de gênero(s) do filme

    Returns:
        dict completo com índices, perfil, estimativas e explicação.
    """
    indices = compute_indices(metrics)
    profile = classify_profile(indices)
    imdb_est = estimate_imdb(indices, genre)
    roi_est = estimate_roi(indices, genre)
    bo_est = estimate_box_office_tendency(indices, genre)
    quality = compute_quality_score(metrics, indices)
    explanation = generate_explanation(
        indices, imdb_est, roi_est, bo_est, profile, genre, quality)

    return {
        "indices": indices,
        "profile": profile,
        "imdb_estimate": imdb_est,
        "roi_estimate": roi_est,
        "box_office_tendency": bo_est,
        "quality": quality,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _extract_primary_genre(genre_str: str) -> str:
    """Extrai o gênero principal de uma string multi-label."""
    if not genre_str or genre_str == "default":
        return "default"
    # Prioridade: gêneros com modelos mais fortes primeiro
    priority = ["Horror", "Thriller", "Comedy", "Crime", "Drama",
                "Action", "Romance", "Adventure"]
    for g in priority:
        if g.lower() in genre_str.lower():
            return g
    return genre_str.split(",")[0].strip()


def _compute_confidence(indices: dict, genre: str) -> str:
    """Determina confiança da estimativa."""
    score = 0

    # Gêneros com suporte empírico forte
    if genre in ("Horror", "Thriller"):
        score += 2
    elif genre in ("Comedy", "Crime"):
        score += 1

    # Perfil extremo → mais informação → mais confiança
    if indices["stability"] > 0.6 or indices["stability"] < 0.3:
        score += 1
    if indices["tension"] > 0.6 or indices["tension"] < 0.3:
        score += 1

    # Pouca mudança no final → mais previsível
    if indices["ending_shift"] < 0.3:
        score += 1

    if score >= 4:
        return "moderada"
    if score >= 2:
        return "baixa"
    return "muito baixa"
