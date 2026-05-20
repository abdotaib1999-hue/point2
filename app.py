# =============================================================================
# APPLICATION SISMIQUE — MÉTHODE N2 + SPECTRE RPA 2024
# =============================================================================
# Auteur   : Application générée avec architecture professionnelle
# Version  : 1.0.0
# Méthode  : N2 (Fajfar 1999 / EC8 Annexe B)
# Spectre  : RPA 99 v2003 — paramètres à vérifier contre RPA 2024 officiel
#
# Références :
#   [1] Fajfar P. (1999). Capacity Spectrum Method Based on Inelastic Demand
#       Spectra. EESD 28(9), 979–993.
#   [2] EN 1998-1:2004 (Eurocode 8), Annexe B — Méthode N2 simplifiée.
#   [3] RPA 99 version 2003 (DTR BC 2.48). À mettre à jour selon RPA 2024.
# =============================================================================

import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import interpolate
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calcul Sismique N2 — RPA 2024",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES ET PARAMÈTRES RÉGLEMENTAIRES RPA
# ─────────────────────────────────────────────────────────────────────────────
# NOTE IMPORTANTE :
#   Les valeurs ci-dessous sont issues de RPA 99 v2003.
#   Elles doivent être vérifiées et mises à jour selon le document officiel
#   RPA 2024 (DTR BC 2.48 — 2024). Toutes les variables sont explicitement
#   nommées pour faciliter la mise à jour réglementaire.

G_MS2 = 9.81  # Accélération gravitationnelle (m/s²)

# Coefficient sismique de zone A (fraction de g) — RPA 99 v2003 Table 4.1
# Groupes d'importance : 1A, 1B, 2, 3
RPA_ZONE_A: dict[str, dict[str, float]] = {
    "Zone I":   {"Groupe 1A": 0.10, "Groupe 1B": 0.10, "Groupe 2": 0.08, "Groupe 3": 0.05},
    "Zone IIa": {"Groupe 1A": 0.15, "Groupe 1B": 0.15, "Groupe 2": 0.12, "Groupe 3": 0.08},
    "Zone IIb": {"Groupe 1A": 0.20, "Groupe 1B": 0.20, "Groupe 2": 0.16, "Groupe 3": 0.11},
    "Zone III": {"Groupe 1A": 0.25, "Groupe 1B": 0.25, "Groupe 2": 0.20, "Groupe 3": 0.14},
}

# Facteur de site S et périodes caractéristiques T1, T2 (s) — RPA 99 v2003 Table 4.3
# S1 = Rocher/sol très ferme | S2 = Sol ferme | S3 = Sol meuble | S4 = Sol très meuble
RPA_SOIL: dict[str, dict] = {
    "S1 — Rocher / sol très ferme":   {"S": 1.0, "T1": 0.15, "T2": 0.30},
    "S2 — Sol ferme":                  {"S": 1.2, "T1": 0.15, "T2": 0.40},
    "S3 — Sol meuble":                 {"S": 1.5, "T1": 0.15, "T2": 0.50},
    "S4 — Sol très meuble / liquefiable": {"S": 1.8, "T1": 0.20, "T2": 0.70},
}

# Borne inférieure du facteur d'amortissement η — RPA / EC8
ETA_MIN = 0.55

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CHARGEMENT ET VALIDATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def load_file(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Charge un fichier CSV ou Excel. Retourne (df, erreur)."""
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    try:
        if name.endswith(".csv"):
            for sep in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(io.BytesIO(content), sep=sep)
                    if df.shape[1] >= 2:
                        return df, None
                except Exception:
                    continue
            return None, "Impossible de détecter le séparateur CSV. Essayez `,` `;` ou `TAB`."
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
            return df, None
        else:
            return None, "Format non supporté. Utilisez .csv, .xlsx ou .xls"
    except Exception as exc:
        return None, f"Erreur de lecture : {exc}"


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Détecte automatiquement les colonnes déplacement et effort."""
    disp_kw = ["déplac", "deplac", "displ", "delta", "disp", " d ", " u ", "drift"]
    force_kw = ["effort", "force", "cisail", "shear", "base", " v ", " f ", "kn", "charge"]
    cols_lower = {c.lower(): c for c in df.columns}
    col_d = col_f = None
    for kw in disp_kw:
        for cl, co in cols_lower.items():
            if kw in cl:
                col_d = co
                break
        if col_d:
            break
    for kw in force_kw:
        for cl, co in cols_lower.items():
            if kw in cl and co != col_d:
                col_f = co
                break
        if col_f:
            break
    cols = list(df.columns)
    return col_d or cols[0], col_f or (cols[1] if len(cols) > 1 else cols[0])


def validate_pushover(
    df: pd.DataFrame, col_d: str, col_f: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Valide et nettoie les données pushover.
    Retourne (disp, force, avertissements). Lève ValueError si bloquant.
    """
    msgs: list[str] = []

    for col in [col_d, col_f]:
        if col not in df.columns:
            raise ValueError(f"Colonne introuvable : « {col} ».")

    disp = pd.to_numeric(df[col_d], errors="coerce").to_numpy(dtype=float)
    force = pd.to_numeric(df[col_f], errors="coerce").to_numpy(dtype=float)

    mask = ~(np.isnan(disp) | np.isnan(force))
    if (n := int((~mask).sum())) > 0:
        msgs.append(f"{n} ligne(s) ignorée(s) (valeurs non numériques / NaN).")
    disp, force = disp[mask], force[mask]

    if len(disp) < 4:
        raise ValueError("Moins de 4 points valides — vérifiez votre fichier.")

    # Tri par déplacement croissant
    idx = np.argsort(disp)
    if not np.array_equal(idx, np.arange(len(disp))):
        msgs.append("Données triées par déplacement croissant.")
        disp, force = disp[idx], force[idx]

    # Vérification doublons en déplacement
    if (n_dup := int(np.sum(np.diff(disp) == 0))) > 0:
        msgs.append(f"{n_dup} doublon(s) en déplacement — interpolation peut être imprécise.")

    # Vérification monotonicité stricte
    if not np.all(np.diff(disp) > 0):
        raise ValueError(
            "Déplacements non strictement croissants même après tri. "
            "Présence de doublons ou valeurs identiques."
        )

    # Avertissement post-pic
    peak_idx = int(np.argmax(force))
    if peak_idx < len(force) - 2:
        msgs.append(
            f"Post-pic détecté dès le point {peak_idx + 1}/{len(force)} "
            f"(dégradation). La courbe sera tronquée à l'ultime si nécessaire."
        )

    return disp, force, msgs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SPECTRE RPA 2024 (formulation RPA 99 v2003)
# ─────────────────────────────────────────────────────────────────────────────

def rpa_spectrum_sa(
    T: float | np.ndarray,
    A: float,
    S: float,
    T1: float,
    T2: float,
    xi: float = 0.05,
) -> float | np.ndarray:
    """
    Spectre d'accélération élastique RPA (en m/s²).

    Formulation (RPA 99 v2003, Article 4.2.3) :
      [0, T1] : Sa = A·S·[1 + (T/T1)·(2,5η − 1)]·g
      [T1, T2]: Sa = A·S·2,5·η·g                          ← plateau
      [T2, 3s]: Sa = A·S·2,5·η·(T2/T)^(2/3)·g
      [>3s]   : Sa = A·S·2,5·η·(T2/3)^(2/3)·(3/T)^(5/3)·g

    η = √(7/(2+ξ%)) ≥ ηmin  — facteur de correction d'amortissement
    """
    scalar = np.isscalar(T)
    T = np.atleast_1d(np.asarray(T, dtype=float))
    T = np.where(T <= 0, 1e-6, T)

    eta = max(np.sqrt(7.0 / (2.0 + xi * 100.0)), ETA_MIN)
    plateau = A * S * 2.5 * eta * G_MS2

    Sa = np.where(
        T <= T1,
        A * S * (1.0 + (T / T1) * (2.5 * eta - 1.0)) * G_MS2,
        np.where(
            T <= T2,
            plateau,
            np.where(
                T <= 3.0,
                plateau * (T2 / T) ** (2.0 / 3.0),
                plateau * (T2 / 3.0) ** (2.0 / 3.0) * (3.0 / T) ** (5.0 / 3.0),
            ),
        ),
    )
    return float(Sa[0]) if scalar else Sa


def rpa_spectrum_sd(T, A, S, T1, T2, xi=0.05):
    """Déplacement spectral élastique Sd = Sa·T²/(4π²) en mètres."""
    Sa = rpa_spectrum_sa(T, A, S, T1, T2, xi)
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Sd = Sa * T**2 / (4.0 * np.pi**2)
    return Sd if not np.isscalar(T) else float(Sd[0])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TRANSFORMATION MDOF → SDOF (format ADRS)
# ─────────────────────────────────────────────────────────────────────────────

def mdof_to_sdof(
    disp_cm: np.ndarray,
    force_kN: np.ndarray,
    mass_kg: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convertit la courbe pushover MDOF en spectre de capacité SDOF (ADRS).

    Transformation N2 (Fajfar 1999, Eq. 1–3) :
      d*   = D_toit / (Γ · φ_toit)    — déplacement SDOF [m]
      F*   = V_base / Γ               — force SDOF [N]
      m*   = F*_y / Sa_y (déduit)     — masse effective

    Hypothèse : φ_toit normalisée à 1 (mode fondamental normalisé au sommet).
    Γ est fourni par l'utilisateur (typiquement 1.0–1.5).

    Retourne : Sd [m], Sa [m/s²], m_star [kg]
    """
    m_star_kg = mass_kg / gamma  # masse effective du 1er mode (approximation)
    Sd_m = (disp_cm / 100.0) / gamma          # déplacement SDOF en mètres
    Sa_ms2 = (force_kN * 1e3) / m_star_kg     # accélération SDOF en m/s²
    return Sd_m, Sa_ms2, m_star_kg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — IDÉALISATION BILINÉAIRE (énergie égale, EC8 Annexe B)
# ─────────────────────────────────────────────────────────────────────────────

def bilinear_idealize(
    Sd: np.ndarray,
    Sa: np.ndarray,
    du_fraction: float = 0.85,
) -> dict:
    """
    Construit l'idéalisation bilinéaire élasto-parfaitement plastique
    selon la méthode de l'énergie égale (EC8 Annexe B.3 / Fajfar 1999).

    Principe :
      Aire sous la bilinéaire de 0 à d*_u = Aire sous la courbe réelle.
      F*_y · (d*_u − d*_y/2) = ∫₀^{d*_u} F*(d) dd

    Rigidité initiale K* :
      Estimée par régression forcée à l'origine sur les premiers points
      (jusqu'à 60% du pic), puis corrigée pour respecter l'égalité d'énergie.
      Cette approche est robuste pour les courbes sans plateau initial net.

    Retourne un dict avec : Sa_y, Sd_y, Sa_u, Sd_u, T_star, K_star
    """
    # --- Point ultime ---
    Sa_peak = float(np.max(Sa))
    Sa_threshold = du_fraction * Sa_peak

    above = np.where(Sa >= Sa_threshold)[0]
    if len(above) == 0:
        raise ValueError("Aucun point au-delà du seuil ultime. Réduisez la fraction du pic.")
    u_idx = int(above[-1])
    Sd_u = float(Sd[u_idx])
    Sa_u = float(Sa[u_idx])

    if Sd_u <= 0:
        raise ValueError("Déplacement ultime nul ou négatif.")

    # --- Énergie réelle jusqu'au point ultime ---
    E_real = float(np.trapz(Sa[: u_idx + 1], Sd[: u_idx + 1]))

    # --- Estimation de la rigidité initiale K* ---
    # Régression forcée à l'origine sur les 30% premiers points (min 3 pts)
    n_init = max(3, int(0.30 * len(Sd)))
    # Force la régression à passer par (0, 0)
    K_star = float(np.sum(Sa[:n_init] * Sd[:n_init]) / np.sum(Sd[:n_init] ** 2))

    # --- Résolution quadratique pour F*_y (méthode énergie égale) ---
    # F*_y · d*_u − F*_y² / (2 K*) = E_real
    # → F*_y² / (2K*) − F*_y · Sd_u + E_real = 0
    a = 1.0 / (2.0 * K_star)
    b = -Sd_u
    c = E_real
    discriminant = b**2 - 4.0 * a * c

    if discriminant >= 0:
        s1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        s2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        # Solutions physiquement admissibles : 0 < Sa_y ≤ Sa_peak
        candidates = [s for s in (s1, s2) if 0 < s <= Sa_peak * 1.05]
        Sa_y = float(min(candidates)) if candidates else 0.6 * Sa_peak
    else:
        # Cas dégénéré : utilisation du 60% du pic (conservative)
        Sa_y = 0.6 * Sa_peak

    Sd_y = Sa_y / K_star

    # --- Sécurité : Sd_y doit être < Sd_u ---
    if Sd_y >= Sd_u * 0.95:
        Sd_y = Sd_u * 0.80
        Sa_y = K_star * Sd_y

    # --- Période effective du SDOF ---
    T_star = 2.0 * np.pi * np.sqrt(Sd_y / Sa_y) if Sa_y > 0 else 0.0

    return {
        "Sa_y": float(Sa_y),   # m/s²
        "Sd_y": float(Sd_y),   # m
        "Sa_u": float(Sa_u),   # m/s²
        "Sd_u": float(Sd_u),   # m
        "T_star": float(T_star),
        "K_star": float(K_star),
        "E_real": float(E_real),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MÉTHODE N2 : CALCUL DU POINT DE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def n2_method(
    Sd_cap: np.ndarray,
    Sa_cap: np.ndarray,
    bil: dict,
    A: float,
    S: float,
    T1_soil: float,
    T2_soil: float,
    xi: float,
    gamma: float,
) -> dict:
    """
    Méthode N2 — Calcul du déplacement cible et du point de performance.

    Algorithme (Fajfar 1999, Eq. 4–8 ; EC8 Annexe B.4–B.5) :
    ───────────────────────────────────────────────────────────────────────────
    1.  Se* = Sa_RPA_élastique(T*)      — demande spectrale élastique au SDOF
    2.  Sd_el* = Se* · T*² / (4π²)     — déplacement élastique correspondant
    3.  Si Se* ≤ Sa*_y  →  réponse élastique, Sd_t* = Sd_el*
    4.  Sinon (inélastique) :
        Rμ = Se* / Sa*_y               — facteur de réduction par ductilité
        a) T* ≥ Tc  →  règle du déplacement égal :
              Sd_t* = Sd_el*  ;  μ = Rμ
        b) T* < Tc  →  amplification courte période (Fajfar 1999, Eq. 7) :
              μ    = (Rμ − 1) · Tc/T* + 1
              Sd_t* = μ · Sd*_y
    5.  Sa_perf = interp(Sd_t*, courbe capacité SDOF)
    6.  d_cible_MDOF = Γ · Sd_t*
    ───────────────────────────────────────────────────────────────────────────
    Choix pédagogique : variante EC8/Fajfar linéaire T*/Tc, plus transparente
    qu'une itération CSM (Chopra-Goel) et suffisante pour une analyse au 1er ordre.
    """
    T_star = bil["T_star"]
    Sa_y = bil["Sa_y"]
    Sd_y = bil["Sd_y"]
    Tc = T2_soil  # Période limite de la branche plateau (T2 du sol)

    # — Étape 1 : demande élastique —
    Se_star = float(rpa_spectrum_sa(T_star, A, S, T1_soil, T2_soil, xi))
    Sd_el_star = Se_star * T_star**2 / (4.0 * np.pi**2)

    # — Étape 2 : détermination du régime —
    if Se_star <= Sa_y:
        # Réponse élastique
        Sd_t_star = Sd_el_star
        mu_demand = 1.0
        R_mu = 1.0
        regime = "Élastique"
    else:
        R_mu = Se_star / Sa_y
        if T_star >= Tc:
            # Règle du déplacement égal (période moyenne / longue)
            Sd_t_star = Sd_el_star
            mu_demand = R_mu
            regime = f"Inélastique — T* ≥ Tc (déplacement égal)"
        else:
            # Courte période : formule Fajfar 1999
            mu_demand = (R_mu - 1.0) * (Tc / max(T_star, 1e-4)) + 1.0
            Sd_t_star = mu_demand * Sd_y
            regime = f"Inélastique — T* < Tc (amplification courte période)"

    # — Étape 3 : lecture sur la courbe de capacité —
    f_interp = interpolate.interp1d(
        Sd_cap, Sa_cap, kind="linear", bounds_error=False, fill_value=(Sa_cap[0], Sa_cap[-1])
    )
    Sa_perf = float(f_interp(Sd_t_star))

    # Dépassement de la courbe de capacité
    capacity_exceeded = Sd_t_star > Sd_cap[-1]

    # — Étape 4 : retour MDOF —
    d_target_m = gamma * Sd_t_star      # déplacement toit MDOF [m]
    d_target_cm = d_target_m * 100.0    # en cm

    return {
        "T_star": T_star,
        "Se_star_ms2": Se_star,
        "Se_star_g": Se_star / G_MS2,
        "Sd_el_star_m": Sd_el_star,
        "Sd_el_star_cm": Sd_el_star * 100.0,
        "Sa_y_ms2": Sa_y,
        "Sa_y_g": Sa_y / G_MS2,
        "Sd_y_m": Sd_y,
        "Sd_y_cm": Sd_y * 100.0,
        "R_mu": R_mu,
        "mu_demand": mu_demand,
        "regime": regime,
        "Sd_target_star_m": Sd_t_star,
        "Sd_target_star_cm": Sd_t_star * 100.0,
        "Sa_performance_ms2": Sa_perf,
        "Sa_performance_g": Sa_perf / G_MS2,
        "d_target_cm": d_target_cm,
        "capacity_exceeded": capacity_exceeded,
        "Tc": Tc,
    }


def interp_pushover_force(disp_cm: np.ndarray, force_kN: np.ndarray, d_cm: float) -> float:
    """Force pushover interpolée (kN) au déplacement cible (cm)."""
    f = interpolate.interp1d(
        disp_cm, force_kN, kind="linear", bounds_error=False,
        fill_value=(force_kN[0], force_kN[-1])
    )
    return float(f(d_cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VISUALISATIONS PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "pushover": "#1565C0",
    "bilinear": "#2E7D32",
    "demand":   "#D32F2F",
    "pp":       "#F57F17",
    "T_line":   "#7B1FA2",
    "fill":     "rgba(213, 0, 0, 0.08)",
}


def fig_pushover(disp_cm, force_kN, d_pp=None, V_pp=None) -> go.Figure:
    """Courbe pushover MDOF avec point de performance."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp_cm, y=force_kN, mode="lines",
        name="Courbe pushover",
        line=dict(color=COLORS["pushover"], width=2.5),
        hovertemplate="d = %{x:.2f} cm<br>V = %{y:.1f} kN<extra></extra>",
    ))
    if d_pp is not None:
        fig.add_trace(go.Scatter(
            x=[d_pp], y=[V_pp], mode="markers",
            name=f"Point de performance ({d_pp:.2f} cm | {V_pp:.0f} kN)",
            marker=dict(symbol="star", size=16, color=COLORS["pp"],
                        line=dict(color="black", width=1)),
        ))
        fig.add_vline(x=d_pp, line_dash="dash", line_color=COLORS["pp"], opacity=0.6,
                      annotation_text=f"  d={d_pp:.2f} cm", annotation_position="top right")
    fig.update_layout(
        title="Courbe de capacité Pushover (MDOF)",
        xaxis_title="Déplacement au sommet (cm)",
        yaxis_title="Effort tranchant à la base (kN)",
        height=420, template="plotly_white", hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def fig_adrs(
    Sd_cap_m, Sa_cap_ms2,
    bil: dict,
    res: dict,
    A, S, T1_soil, T2_soil, xi,
) -> go.Figure:
    """
    Graphique ADRS complet :
      - Spectre de demande élastique RPA 2024
      - Courbe de capacité SDOF
      - Bilinéaire idéalisée
      - Lignes de période T* constante
      - Point de performance
    """
    fig = go.Figure()

    T_arr = np.linspace(0.01, 4.0, 600)
    Sa_dem = rpa_spectrum_sa(T_arr, A, S, T1_soil, T2_soil, xi)
    Sd_dem = Sa_dem * T_arr**2 / (4.0 * np.pi**2)

    # Spectre de demande
    fig.add_trace(go.Scatter(
        x=Sd_dem * 100, y=Sa_dem / G_MS2,
        mode="lines", name="Spectre demande RPA (élastique)",
        line=dict(color=COLORS["demand"], width=2.2, dash="dot"),
        fill="tozeroy", fillcolor=COLORS["fill"],
    ))

    # Ligne T* constante
    T_star = res["T_star"]
    Sa_line = np.linspace(0, max(Sa_cap_ms2) * 1.3, 100)
    Sd_line = Sa_line * T_star**2 / (4.0 * np.pi**2)
    fig.add_trace(go.Scatter(
        x=Sd_line * 100, y=Sa_line / G_MS2,
        mode="lines", name=f"T* = {T_star:.3f} s",
        line=dict(color=COLORS["T_line"], width=1.5, dash="longdash"),
    ))

    # Capacité SDOF
    fig.add_trace(go.Scatter(
        x=Sd_cap_m * 100, y=Sa_cap_ms2 / G_MS2,
        mode="lines", name="Capacité SDOF (courbe réelle)",
        line=dict(color=COLORS["pushover"], width=2.5),
        hovertemplate="Sd = %{x:.3f} cm<br>Sa = %{y:.4f} g<extra></extra>",
    ))

    # Bilinéaire
    Sd_bil = [0, bil["Sd_y"] * 100, bil["Sd_u"] * 100]
    Sa_bil = [0, bil["Sa_y"] / G_MS2, bil["Sa_y"] / G_MS2]
    fig.add_trace(go.Scatter(
        x=Sd_bil, y=Sa_bil,
        mode="lines", name="Bilinéaire idéalisée",
        line=dict(color=COLORS["bilinear"], width=2, dash="dash"),
    ))

    # Points bilinéaires
    fig.add_trace(go.Scatter(
        x=[bil["Sd_y"] * 100, bil["Sd_u"] * 100],
        y=[bil["Sa_y"] / G_MS2, bil["Sa_u"] / G_MS2],
        mode="markers", name="Sd_y et Sd_u",
        marker=dict(symbol="circle", size=9, color=COLORS["bilinear"],
                    line=dict(color="white", width=1)),
    ))

    # Point de performance
    Sd_pp = res["Sd_target_star_cm"]
    Sa_pp = res["Sa_performance_g"]
    fig.add_trace(go.Scatter(
        x=[Sd_pp], y=[Sa_pp],
        mode="markers+text", name="Point de performance",
        marker=dict(symbol="star", size=18, color=COLORS["pp"],
                    line=dict(color="black", width=1.5)),
        text=[f"  ({Sd_pp:.2f} cm | {Sa_pp:.3f} g)"],
        textposition="top right", textfont=dict(size=11),
    ))

    fig.update_layout(
        title="Format ADRS — Capacité vs Demande RPA 2024 (Méthode N2)",
        xaxis_title="Déplacement spectral Sd (cm)",
        yaxis_title="Accélération spectrale Sa (g)",
        height=520, template="plotly_white", hovermode="closest",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgray", borderwidth=1),
    )
    return fig


def fig_spectrum_T(A, S, T1_soil, T2_soil, xi, T_star=None) -> go.Figure:
    """Spectre RPA 2024 classique (T vs Sa)."""
    T_arr = np.concatenate([[0.0], np.linspace(0.01, 4.0, 500)])
    Sa_arr = rpa_spectrum_sa(T_arr, A, S, T1_soil, T2_soil, xi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T_arr, y=Sa_arr / G_MS2,
        mode="lines", name="Spectre élastique RPA",
        line=dict(color=COLORS["demand"], width=2.5),
        fill="tozeroy", fillcolor=COLORS["fill"],
        hovertemplate="T = %{x:.3f} s<br>Sa = %{y:.4f} g<extra></extra>",
    ))
    if T_star is not None:
        Se = rpa_spectrum_sa(T_star, A, S, T1_soil, T2_soil, xi)
        fig.add_trace(go.Scatter(
            x=[T_star], y=[Se / G_MS2],
            mode="markers+text", name=f"T* = {T_star:.3f} s",
            marker=dict(symbol="x", size=12, color=COLORS["T_line"],
                        line=dict(width=3)),
            text=[f"  T*={T_star:.3f}s\n  Se={Se/G_MS2:.3f}g"],
            textposition="top right",
        ))
        fig.add_vline(x=T_star, line_dash="dash", line_color=COLORS["T_line"], opacity=0.5)
    fig.update_layout(
        title="Spectre de réponse élastique RPA 2024",
        xaxis_title="Période T (s)", yaxis_title="Sa (g)",
        height=380, template="plotly_white", hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXPORT EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def export_to_excel(
    disp_cm, force_kN,
    Sd_cap_m, Sa_cap_ms2,
    bil: dict,
    res: dict,
    params: dict,
) -> bytes:
    """Génère un classeur Excel avec données, ADRS, bilinéaire et résultats."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # — Feuille 1 : Pushover brut —
        pd.DataFrame({"Déplacement (cm)": disp_cm, "Effort tranchant (kN)": force_kN}).to_excel(
            writer, sheet_name="Pushover MDOF", index=False
        )

        # — Feuille 2 : Capacité SDOF —
        pd.DataFrame({
            "Sd (m)": Sd_cap_m, "Sa (m/s²)": Sa_cap_ms2,
            "Sd (cm)": Sd_cap_m * 100, "Sa (g)": Sa_cap_ms2 / G_MS2,
        }).to_excel(writer, sheet_name="Capacité SDOF", index=False)

        # — Feuille 3 : Bilinéaire —
        Sd_bil_pts = [0, bil["Sd_y"] * 100, bil["Sd_u"] * 100]
        Sa_bil_pts = [0, bil["Sa_y"] / G_MS2, bil["Sa_y"] / G_MS2]
        pd.DataFrame({"Sd bilinéaire (cm)": Sd_bil_pts, "Sa bilinéaire (g)": Sa_bil_pts}).to_excel(
            writer, sheet_name="Bilinéaire", index=False
        )

        # — Feuille 4 : Spectre RPA —
        T_arr = np.linspace(0.01, 4.0, 400)
        Sa_sp = rpa_spectrum_sa(T_arr, params["A"], params["S"], params["T1"], params["T2"], params["xi"])
        pd.DataFrame({
            "T (s)": T_arr, "Sa (m/s²)": Sa_sp, "Sa (g)": Sa_sp / G_MS2,
            "Sd (cm)": Sa_sp * T_arr**2 / (4 * np.pi**2) * 100,
        }).to_excel(writer, sheet_name="Spectre RPA", index=False)

        # — Feuille 5 : Résultats N2 —
        labels = [
            "── Paramètres ──",
            "Zone sismique", "Groupe d'importance", "Coefficient A",
            "Type de sol", "Facteur de site S", "T1 sol (s)", "T2 sol (s)",
            "Amortissement ξ (%)", "Facteur de comportement R",
            "Masse sismique (t)", "Facteur de participation Γ",
            "── Bilinéaire SDOF ──",
            "Sa_y — Accélération de plastification (g)",
            "Sd_y — Déplacement de plastification (cm)",
            "Sa_u — Accélération ultime (g)",
            "Sd_u — Déplacement ultime (cm)",
            "T* — Période effective SDOF (s)",
            "── Méthode N2 ──",
            "Se*(T*) — Demande élastique (g)",
            "Sd_el* — Déplacement élastique (cm)",
            "Rμ — Facteur de réduction par ductilité",
            "μ — Ductilité demandée",
            "Régime de comportement",
            "Sd_t* — Déplacement cible SDOF (cm)",
            "d_cible — Déplacement toit MDOF (cm)",
            "Sa_perf — Accélération au point de performance (g)",
            "V_base — Effort tranchant au point de performance (kN)",
        ]
        values = [
            "─────────────────────",
            params["zone"], params["groupe"], f"{params['A']:.4f}",
            params["soil"], f"{params['S']:.3f}", f"{params['T1']:.3f}", f"{params['T2']:.3f}",
            f"{params['xi']*100:.1f}", f"{params['R']:.2f}",
            f"{params['mass_t']:.1f}", f"{params['gamma']:.4f}",
            "─────────────────────",
            f"{bil['Sa_y']/G_MS2:.5f}", f"{bil['Sd_y']*100:.5f}",
            f"{bil['Sa_u']/G_MS2:.5f}", f"{bil['Sd_u']*100:.5f}",
            f"{bil['T_star']:.5f}",
            "─────────────────────",
            f"{res['Se_star_g']:.5f}", f"{res['Sd_el_star_cm']:.5f}",
            f"{res['R_mu']:.4f}", f"{res['mu_demand']:.4f}",
            res["regime"],
            f"{res['Sd_target_star_cm']:.5f}",
            f"{res['d_target_cm']:.5f}",
            f"{res['Sa_performance_g']:.5f}",
            f"{params.get('V_pp_kN', 'N/A')}",
        ]
        pd.DataFrame({"Paramètre": labels, "Valeur": values}).to_excel(
            writer, sheet_name="Résultats N2", index=False
        )

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — INTERFACE STREAMLIT PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_parameters() -> dict:
    """Construit la barre latérale et retourne un dict de paramètres."""
    st.sidebar.header("⚙️ Paramètres de calcul")

    # ── Sismicité ──────────────────────────────────────────────────────────
    st.sidebar.subheader("🌍 Sismicité — RPA 2024")
    zone = st.sidebar.selectbox("Zone sismique", list(RPA_ZONE_A.keys()), index=1)
    groupe = st.sidebar.selectbox(
        "Groupe d'importance", list(RPA_ZONE_A[zone].keys()), index=1
    )
    A_auto = RPA_ZONE_A[zone][groupe]
    st.sidebar.info(f"Coefficient de zone **A = {A_auto}**")
    custom_A = st.sidebar.checkbox("Personnaliser A")
    A = st.sidebar.number_input("A (perso)", 0.01, 0.60, float(A_auto), 0.005) if custom_A else A_auto

    # ── Sol ────────────────────────────────────────────────────────────────
    st.sidebar.subheader("🪨 Sol")
    soil = st.sidebar.selectbox("Classe de sol", list(RPA_SOIL.keys()), index=1)
    sp = RPA_SOIL[soil]
    S, T1, T2 = sp["S"], sp["T1"], sp["T2"]
    c1, c2, c3 = st.sidebar.columns(3)
    c1.metric("S", f"{S:.2f}")
    c2.metric("T1", f"{T1:.2f}s")
    c3.metric("T2", f"{T2:.2f}s")
    if st.sidebar.checkbox("Modifier S, T1, T2 (avancé)"):
        S  = st.sidebar.number_input("S",  0.5, 3.0,  float(S),  0.05)
        T1 = st.sidebar.number_input("T1 (s)", 0.05, 0.50, float(T1), 0.01)
        T2 = st.sidebar.number_input("T2 (s)", 0.10, 2.00, float(T2), 0.05)

    # ── Amortissement et comportement ─────────────────────────────────────
    st.sidebar.subheader("📉 Amortissement")
    xi_pct = st.sidebar.slider("Amortissement ξ (%)", 2, 20, 5, 1)
    xi = xi_pct / 100.0
    R = st.sidebar.number_input(
        "Facteur de comportement R",
        1.0, 10.0, 1.0, 0.5,
        help="R = 1 pour la méthode N2 standard (le spectre élastique est utilisé "
             "en interne ; la non-linéarité est captée par la bilinéaire).",
    )

    # ── Structure ─────────────────────────────────────────────────────────
    st.sidebar.subheader("🏢 Structure")
    mass_t = st.sidebar.number_input("Masse sismique (tonnes)", 1.0, 1e6, 500.0, 50.0)
    gamma = st.sidebar.number_input(
        "Facteur de participation Γ", 0.5, 2.5, 1.0, 0.05,
        help="Γ = Σ(mᵢφᵢ) / Σ(mᵢφᵢ²). Sans analyse modale, Γ ≈ 1.0–1.3.",
    )
    unit_d = st.sidebar.radio("Unité des déplacements (fichier)", ["cm", "mm"])

    # ── Options avancées ──────────────────────────────────────────────────
    st.sidebar.subheader("🔧 Options")
    du_frac = st.sidebar.slider(
        "Fraction du pic → point ultime", 0.50, 1.00, 0.85, 0.05,
        help="Point ultime = dernier point où Sa ≥ fraction × Sa_max.",
    )

    return dict(
        zone=zone, groupe=groupe, A=A, soil=soil,
        S=S, T1=T1, T2=T2, xi=xi, R=R,
        mass_t=mass_t, mass_kg=mass_t * 1e3,
        gamma=gamma, unit_d=unit_d, du_frac=du_frac,
    )


def tab_data(p: dict):
    """Onglet chargement des données."""
    st.header("📂 Données pushover")

    # Fichier exemple téléchargeable
    df_ex = pd.DataFrame({
        "Déplacement_cm": [0, 0.5, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 35],
        "Effort_kN":       [0, 180, 340, 600, 980, 1250, 1480, 1590, 1650, 1670, 1650, 1600, 1530],
    })
    buf = io.BytesIO()
    df_ex.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Télécharger un fichier exemple (CSV)",
        buf.getvalue(), "exemple_pushover.csv", "text/csv",
    )

    uploaded = st.file_uploader(
        "Importer la courbe pushover (CSV ou Excel)",
        type=["csv", "xlsx", "xls"],
        help="2 colonnes minimum : déplacement et effort tranchant.",
    )

    if uploaded is None:
        st.info("Aucun fichier importé — téléchargez l'exemple ci-dessus pour tester l'application.")
        return

    df_raw, err = load_file(uploaded)
    if err:
        st.error(f"❌ {err}")
        return

    st.success(f"✅ Fichier chargé : **{len(df_raw)} lignes**, **{len(df_raw.columns)} colonnes**")

    # Sélection colonnes
    col_d_auto, col_f_auto = detect_columns(df_raw)
    c1, c2 = st.columns(2)
    col_d = c1.selectbox(
        "Colonne **déplacement**", df_raw.columns.tolist(),
        index=df_raw.columns.tolist().index(col_d_auto) if col_d_auto in df_raw.columns else 0,
    )
    col_f = c2.selectbox(
        "Colonne **effort tranchant**", df_raw.columns.tolist(),
        index=df_raw.columns.tolist().index(col_f_auto) if col_f_auto in df_raw.columns else min(1, len(df_raw.columns) - 1),
    )

    with st.expander("👁️ Aperçu des données brutes"):
        st.dataframe(df_raw.head(30), use_container_width=True)

    # Validation
    try:
        disp_raw, force_raw, msgs = validate_pushover(df_raw, col_d, col_f)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        return

    for m in msgs:
        st.warning(m)

    # Conversion unités
    disp_cm = disp_raw / 10.0 if p["unit_d"] == "mm" else disp_raw.copy()
    if p["unit_d"] == "mm":
        st.info("Conversion automatique mm → cm appliquée.")

    # Statistiques rapides
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Nb points", len(disp_cm))
    s2.metric("Dépl. max", f"{disp_cm.max():.2f} cm")
    s3.metric("Effort max", f"{force_raw.max():.0f} kN")
    s4.metric("Rigidité init. app.", f"{force_raw[1]/disp_cm[1]:.0f} kN/cm" if disp_cm[1] > 0 else "—")

    # Stockage session
    st.session_state.update({
        "disp_cm": disp_cm,
        "force_kN": force_raw,
        "data_ok": True,
    })

    fig = fig_pushover(disp_cm, force_raw)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("⬇️ Télécharger figure (HTML)", fig.to_html(),
                       "pushover.html", "text/html")


def tab_n2(p: dict):
    """Onglet principal : analyse N2."""
    st.header("📈 Analyse N2 — Point de performance")

    if not st.session_state.get("data_ok", False):
        st.info("👆 Importez d'abord la courbe pushover dans l'onglet **Données**.")
        return

    disp_cm = st.session_state["disp_cm"]
    force_kN = st.session_state["force_kN"]

    with st.spinner("Calcul en cours…"):
        try:
            # 1. Conversion MDOF → SDOF
            Sd_cap, Sa_cap, m_star = mdof_to_sdof(
                disp_cm, force_kN, p["mass_kg"], p["gamma"]
            )

            # 2. Bilinéaire
            bil = bilinear_idealize(Sd_cap, Sa_cap, p["du_frac"])

            # 3. Méthode N2
            res = n2_method(
                Sd_cap, Sa_cap, bil,
                p["A"], p["S"], p["T1"], p["T2"], p["xi"], p["gamma"],
            )

            # 4. Effort sur courbe pushover
            V_pp = interp_pushover_force(disp_cm, force_kN, res["d_target_cm"])
            res["V_pp_kN"] = V_pp
            p["V_pp_kN"] = V_pp

        except Exception as exc:
            st.error(f"❌ Erreur de calcul : {exc}")
            st.exception(exc)
            return

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.subheader("🎯 Point de performance")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Déplacement cible", f"{res['d_target_cm']:.3f} cm",
              help="Déplacement au sommet de la structure (MDOF)")
    k2.metric("Effort tranchant", f"{V_pp:.0f} kN")
    k3.metric("Ductilité μ", f"{res['mu_demand']:.3f}")
    k4.metric("Période T*", f"{res['T_star']:.4f} s")
    k5.metric("Rμ", f"{res['R_mu']:.3f}")

    # Étiquette de comportement colorée
    mu = res["mu_demand"]
    if mu <= 1.0:
        st.success(f"✅ **{res['regime']}**")
    elif mu <= 3.0:
        st.warning(f"⚠️ **{res['regime']}** — ductilité demandée : {mu:.2f}")
    elif mu <= 6.0:
        st.error(f"🔴 **{res['regime']}** — ductilité demandée : {mu:.2f} (forte demande inélastique)")
    else:
        st.error(f"🚨 **{res['regime']}** — μ = {mu:.2f} **TRÈS ÉLEVÉE** — vérifiez la structure !")

    if res["capacity_exceeded"]:
        st.error("⚠️ Le déplacement cible **dépasse l'étendue de la courbe de capacité**. "
                 "Résultats à interpréter avec précaution.")

    # ── Tableau récapitulatif ──────────────────────────────────────────────
    with st.expander("📋 Tableau détaillé des résultats"):
        df_res = pd.DataFrame({
            "Paramètre": [
                "T* — Période effective SDOF (s)",
                "Se*(T*) — Demande élastique spectrale (g)",
                "Sd_el* — Déplacement élastique SDOF (cm)",
                "Sa_y — Accélération de plastification (g)",
                "Sd_y — Déplacement de plastification (cm)",
                "Sa_u — Accélération ultime (g)",
                "Sd_u — Déplacement ultime (cm)",
                "Tc — Période limite sol (s)",
                "Comparaison T* / Tc",
                "Rμ — Facteur de réduction par ductilité",
                "μ — Ductilité demandée",
                "Régime de comportement",
                "Sd_t* — Déplacement cible SDOF (cm)",
                "d_cible — Déplacement toit MDOF (cm)",
                "Sa_perf (g)",
                "V_base au point de performance (kN)",
            ],
            "Valeur": [
                f"{res['T_star']:.5f}",
                f"{res['Se_star_g']:.5f}",
                f"{res['Sd_el_star_cm']:.5f}",
                f"{bil['Sa_y']/G_MS2:.5f}",
                f"{bil['Sd_y']*100:.5f}",
                f"{bil['Sa_u']/G_MS2:.5f}",
                f"{bil['Sd_u']*100:.5f}",
                f"{res['Tc']:.3f}",
                f"T* {'≥' if res['T_star'] >= res['Tc'] else '<'} Tc",
                f"{res['R_mu']:.5f}",
                f"{res['mu_demand']:.5f}",
                res["regime"],
                f"{res['Sd_target_star_cm']:.5f}",
                f"{res['d_target_cm']:.5f}",
                f"{res['Sa_performance_g']:.5f}",
                f"{V_pp:.2f}",
            ],
        })
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    # ── Graphique ADRS ─────────────────────────────────────────────────────
    st.subheader("📊 Diagramme ADRS — Capacité vs Demande")
    f_adrs = fig_adrs(Sd_cap, Sa_cap, bil, res, p["A"], p["S"], p["T1"], p["T2"], p["xi"])
    st.plotly_chart(f_adrs, use_container_width=True)
    st.download_button("⬇️ ADRS (HTML)", f_adrs.to_html(), "adrs.html", "text/html")

    # ── Courbe pushover + PP ───────────────────────────────────────────────
    st.subheader("📊 Courbe pushover avec point de performance")
    f_push = fig_pushover(disp_cm, force_kN, res["d_target_cm"], V_pp)
    st.plotly_chart(f_push, use_container_width=True)
    st.download_button("⬇️ Pushover (HTML)", f_push.to_html(), "pushover_pp.html", "text/html")

    # ── Export Excel ───────────────────────────────────────────────────────
    st.subheader("📥 Export des résultats")
    xlsx_bytes = export_to_excel(
        disp_cm, force_kN, Sd_cap, Sa_cap, bil, res, p
    )
    st.download_button(
        "⬇️ Télécharger résultats complets (Excel)",
        xlsx_bytes, "resultats_n2_rpa2024.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def tab_spectrum(p: dict):
    """Onglet spectre RPA 2024."""
    st.header("📊 Spectre de réponse RPA 2024")

    T_star_disp = None
    if st.session_state.get("data_ok"):
        try:
            Sd_c, Sa_c, _ = mdof_to_sdof(
                st.session_state["disp_cm"], st.session_state["force_kN"],
                p["mass_kg"], p["gamma"],
            )
            bil_tmp = bilinear_idealize(Sd_c, Sa_c, p["du_frac"])
            T_star_disp = bil_tmp["T_star"]
        except Exception:
            pass

    fig_sp = fig_spectrum_T(p["A"], p["S"], p["T1"], p["T2"], p["xi"], T_star_disp)
    st.plotly_chart(fig_sp, use_container_width=True)

    # Tableau des paramètres
    eta_val = max(np.sqrt(7.0 / (2.0 + p["xi"] * 100.0)), ETA_MIN)
    plateau_g = p["A"] * p["S"] * 2.5 * eta_val
    df_params = pd.DataFrame({
        "Paramètre": [
            "Zone sismique", "Groupe d'importance", "Coefficient A",
            "Type de sol", "Facteur de site S", "T1 sol (s)", "T2 sol (s)",
            "Amortissement ξ (%)", "η = √(7/(2+ξ%)) ≥ 0.55",
            "Plateau du spectre A·S·2,5·η (g)",
            "Période SDOF T* (s)" if T_star_disp else "T* (s)",
        ],
        "Valeur": [
            p["zone"], p["groupe"], f"{p['A']:.4f}",
            p["soil"], f"{p['S']:.3f}", f"{p['T1']:.3f}", f"{p['T2']:.3f}",
            f"{p['xi']*100:.1f}", f"{eta_val:.4f}",
            f"{plateau_g:.4f}",
            f"{T_star_disp:.4f}" if T_star_disp else "— (importer des données)",
        ],
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Spectre (HTML)", fig_sp.to_html(), "spectre.html", "text/html")


def tab_hypotheses():
    """Onglet hypothèses et documentation."""
    st.header("ℹ️ Hypothèses, méthode et limitations")

    st.markdown("""
    ---
    ### Méthode N2 — Fajfar (1999) / EC8 Annexe B

    | Étape | Description |
    |-------|-------------|
    | **1** | Analyse pushover MDOF (fournie par l'utilisateur) |
    | **2** | Transformation MDOF→SDOF : d* = D_toit/Γ ; F* = V_base/Γ |
    | **3** | Idéalisation bilinéaire (énergie égale, EC8 Annexe B.3) |
    | **4** | Période effective : T* = 2π√(Sd_y*/Sa_y*) |
    | **5** | Demande élastique à T* : Se* = Sa_RPA(T*) |
    | **6** | Régime élastique si Se* ≤ Sa_y → Sd_t* = Se*·T*²/4π² |
    | **7** | Régime inélastique : Rμ = Se*/Sa_y |
    | — | T* ≥ Tc → Sd_t* = Sd_el* (déplacement égal) |
    | — | T* < Tc → μ = (Rμ−1)·Tc/T*+1 ; Sd_t* = μ·Sd_y* |
    | **8** | Retour MDOF : d_cible = Γ·Sd_t* |

    ---
    ### Spectre RPA 2024

    > ⚠️ **Note réglementaire** : Les coefficients implémentés sont ceux de **RPA 99 v2003**.
    > Les valeurs du **RPA 2024** (DTR BC 2.48 — 2024) doivent être vérifiées contre
    > le document officiel. Toutes les variables sont explicitement nommées dans le
    > fichier `app.py` (section `RPA_ZONE_A` et `RPA_SOIL`) pour faciliter la mise à jour.

    **Formulation** (4 branches) :

    | Domaine | Expression |
    |---------|-----------|
    | 0 ≤ T ≤ T₁ | Sa = A·S·[1 + (T/T₁)·(2,5η−1)]·g |
    | T₁ ≤ T ≤ T₂ | Sa = A·S·2,5·η·g  *(plateau)* |
    | T₂ ≤ T ≤ 3s | Sa = A·S·2,5·η·(T₂/T)^(2/3)·g |
    | T > 3s | Sa = A·S·2,5·η·(T₂/3)^(2/3)·(3/T)^(5/3)·g |

    avec η = √(7/(2+ξ%)) ≥ 0,55

    ---
    ### Idéalisation bilinéaire

    Méthode de l'énergie égale (EC8 Annexe B.3) :

    **Condition** : Aire sous la bilinéaire = Aire sous la courbe réelle jusqu'à d*_u

    F*_y · (d*_u − d*_y/2) = ∫₀^{d*u} F*(d) dd

    La rigidité initiale K* est estimée par régression forcée à l'origine
    sur les premiers points (≤ 30 % de la courbe), puis la résolution est quadratique.

    ---
    ### Hypothèses et domaine de validité

    | Hypothèse | Détail |
    |-----------|--------|
    | Mode fondamental | Valide si T₁ < 2 s et structure régulière (pas de torsion dominante) |
    | Γ unique | Fourni par l'utilisateur ; obtenu par analyse modale |
    | φ_toit = 1 | Déformée normalisée au sommet (convention standard) |
    | Bilinéaire EPP | Modèle élasto-parfaitement plastique (sans écrouissage) |
    | Spectre moyen | Pas de variabilité probabiliste (déterministe) |
    | Amortissement constant | 5 % par défaut ; η corrige pour d'autres valeurs |

    ---
    ### Limitations connues

    - Précision réduite pour les structures avec **fort post-pic** ou dégradation de rigidité
    - Moins précis pour **T* < 0,1 s** (structures très rigides)
    - Ne tient pas compte des **effets P-Δ** (doit être inclus dans la courbe pushover)
    - Distribution de forces latérales supposée **proportionnelle au 1ᵉʳ mode**
    - Pour les structures **irrégulières**, une analyse temporelle non-linéaire est recommandée

    ---
    ### Références

    1. Fajfar P. (1999). *Capacity Spectrum Method Based on Inelastic Demand Spectra.*  
       Earthquake Engineering and Structural Dynamics, 28(9), 979–993.
    2. EN 1998-1:2004 — Eurocode 8 : Calcul des structures pour leur résistance aux séismes.  
       Annexe B : Méthode N2 simplifiée.
    3. RPA 99 version 2003 — Règles Parasismiques Algériennes (DTR BC 2.48).
    """)


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Initialisation de l'état de session
    if "data_ok" not in st.session_state:
        st.session_state["data_ok"] = False

    # En-tête
    st.title("🏗️ Calcul sismique — Méthode N2 + RPA 2024")
    st.caption(
        "Calcul du point de performance par la méthode N2 (Fajfar 1999 / EC8 Annexe B) "
        "avec spectre de demande RPA 2024 | Version 1.0.0"
    )
    st.divider()

    # Paramètres (sidebar)
    p = sidebar_parameters()

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Données",
        "📈 Analyse N2",
        "📊 Spectres",
        "ℹ️ Hypothèses",
    ])

    with tab1:
        tab_data(p)
    with tab2:
        tab_n2(p)
    with tab3:
        tab_spectrum(p)
    with tab4:
        tab_hypotheses()


if __name__ == "__main__":
    main()
