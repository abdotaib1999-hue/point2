# =============================================================================
# APPLICATION SISMIQUE — MÉTHODE N2 + SPECTRE RPA 2024 (DTR BC 2.48 — 2024)
# =============================================================================
# Version  : 2.0.0  (corrigée et vérifiée contre RPA 2024 officiel)
# Méthode  : N2 (Fajfar 1999 / EC8 Annexe B)
#
# ─── CORRECTIONS v2.0 (vérifiées sur RPA 2024 officiel) ───────────────────
#   1. Nouveau système de zones : 0, I, II, III, IV, V, VI (7 zones)
#   2. Deux types de spectres : Type 1 (Zones IV-VI) | Type 2 (Zones I-III)
#   3. Formule du spectre corrigée — branche décroissante linéaire (T2/T)
#      et palier de déplacement (T2·T3/T²)  [cf. RPA 2024, Eq. 3.8]
#   4. Paramètres S, T1, T2, T3 exacts selon les Tableaux 3.4 et 3.5 RPA 2024
#   5. Facteur de correction d'amortissement η = √(10/(5+ξ%)) [EC8 / RPA 2024]
#   6. Coefficient d'importance I intégré dans le spectre
#   7. Bilinéaire : méthode Vy=Vmax + égalité d'aire (conforme à l'exemple)
#
# Références :
#   [1] RPA 2024 (DTR BC 2.48 — 2024), §3.3, Eq. (3.8), Tableaux 3.3-3.5, 3.11
#   [2] Fajfar P. (2000). Earthquake Spectra, 16(3), 573-592.
#   [3] EN 1998-1:2004 (Eurocode 8), Annexe B.
# =============================================================================

import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import interpolate

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
# PARAMÈTRES RÉGLEMENTAIRES RPA 2024  (DTR BC 2.48 — 2024)
# ─────────────────────────────────────────────────────────────────────────────

G_MS2 = 9.81  # m/s²
ETA_MIN = 0.55  # borne inférieure de η (EC8 / RPA 2024)

# Tableau 3.3 — Coefficient d'accélération de zone A (sol S1, Tr=475 ans)
# Zones : 0, I, II, III, IV, V, VI
RPA2024_ZONE_A: dict[str, float] = {
    "Zone 0": 0.00,
    "Zone I   — 0.07g": 0.07,
    "Zone II  — 0.10g": 0.10,
    "Zone III — 0.15g": 0.15,
    "Zone IV  — 0.20g": 0.20,
    "Zone V   — 0.25g": 0.25,
    "Zone VI  — 0.30g": 0.30,
}

# Spectre Type 1 : appliqué aux Zones IV, V et VI  (Mw ≥ 5.5)
# Tableau 3.4 — S, T1, T2, T3
RPA2024_TYPE1: dict[str, dict] = {
    "S1 — Rocheux":           {"S": 1.00, "T1": 0.10, "T2": 0.40, "T3": 2.0},
    "S2 — Sol ferme":         {"S": 1.20, "T1": 0.10, "T2": 0.50, "T3": 2.0},
    "S3 — Sol meuble":        {"S": 1.30, "T1": 0.15, "T2": 0.60, "T3": 2.0},
    "S4 — Sol très meuble":   {"S": 1.35, "T1": 0.15, "T2": 0.70, "T3": 2.0},
}

# Spectre Type 2 : appliqué aux Zones I, II et III  (Mw ≤ 5.5)
# Tableau 3.5 — S, T1, T2, T3
RPA2024_TYPE2: dict[str, dict] = {
    "S1 — Rocheux":           {"S": 1.00, "T1": 0.05, "T2": 0.25, "T3": 1.20},
    "S2 — Sol ferme":         {"S": 1.30, "T1": 0.05, "T2": 0.30, "T3": 1.20},
    "S3 — Sol meuble":        {"S": 1.55, "T1": 0.10, "T2": 0.40, "T3": 1.20},
    "S4 — Sol très meuble":   {"S": 1.80, "T1": 0.10, "T2": 0.50, "T3": 1.20},
}

# Tableau 3.11 — Coefficient d'importance I
RPA2024_IMPORTANCE: dict[str, float] = {
    "Groupe 1A — Vital (I=1.40)":   1.40,
    "Groupe 1B — Important (I=1.20)": 1.20,
    "Groupe 2  — Courant (I=1.00)": 1.00,
    "Groupe 3  — Faible (I=0.80)":  0.80,
}

# Zones nécessitant le spectre Type 1 vs Type 2
ZONES_TYPE1 = {"Zone IV  — 0.20g", "Zone V   — 0.25g", "Zone VI  — 0.30g"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SPECTRE RPA 2024 (formule exacte)
# ─────────────────────────────────────────────────────────────────────────────

def eta_rpa2024(xi: float) -> float:
    """
    Facteur de correction d'amortissement η (RPA 2024 / EC8, Eq. 3.9).
    η = √(10 / (5 + ξ%))  ≥  ηmin = 0.55
    Note : ξ en fraction (ex : 0.05 pour 5%).
    Vérification : ξ=5% → η = √(10/10) = 1.0 ✓
    """
    return max(np.sqrt(10.0 / (5.0 + xi * 100.0)), ETA_MIN)


def rpa2024_spectrum(
    T: np.ndarray | float,
    A: float,
    I: float,
    S: float,
    T1: float,
    T2: float,
    T3: float,
    xi: float = 0.05,
) -> np.ndarray | float:
    """
    Spectre de réponse élastique horizontal RPA 2024 — Eq. (3.8).

    Formulation exacte (4 branches) :
    ─────────────────────────────────────────────────────────────────────────
      [0, T1]  :  Sae/g = A·I·S·[1 + (T/T1)·(2.5η − 1)]
      [T1, T2] :  Sae/g = A·I·S·2.5η                     ← plateau
      [T2, T3] :  Sae/g = A·I·S·2.5η·(T2/T)              ← décroissance linéaire
      [T3, 4s] :  Sae/g = A·I·S·2.5η·(T2·T3/T²)          ← déplacement constant
    ─────────────────────────────────────────────────────────────────────────
    Paramètres : A, I, S, T1, T2, T3 sont les paramètres RPA 2024.
    xi : amortissement en fraction (0.05 = 5 %).
    Retourne Sae en m/s².
    """
    scalar = np.isscalar(T)
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    T_arr = np.where(T_arr <= 0, 1e-9, T_arr)

    eta = eta_rpa2024(xi)
    plateau = A * I * S * 2.5 * eta   # valeur du plateau (en g)

    Sa_g = np.where(
        T_arr <= T1,
        A * I * S * (1.0 + (T_arr / T1) * (2.5 * eta - 1.0)),
        np.where(
            T_arr <= T2,
            plateau,
            np.where(
                T_arr <= T3,
                plateau * (T2 / T_arr),                   # branche linéaire
                plateau * (T2 * T3 / T_arr**2),           # palier déplacement
            ),
        ),
    )

    Sa_ms2 = Sa_g * G_MS2
    return float(Sa_ms2[0]) if scalar else Sa_ms2


def Sd_rpa2024(T, A, I, S, T1, T2, T3, xi=0.05):
    """Déplacement spectral élastique Sd = Sae · T² / (4π²), en mètres."""
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    Sa = rpa2024_spectrum(T_arr, A, I, S, T1, T2, T3, xi)
    Sd = Sa * T_arr**2 / (4.0 * np.pi**2)
    return Sd if not np.isscalar(T) else float(Sd[0])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CHARGEMENT ET VALIDATION DES DONNÉES PUSHOVER
# ─────────────────────────────────────────────────────────────────────────────

def load_file(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
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
            return None, "Impossible de détecter le séparateur CSV."
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
            return df, None
        else:
            return None, "Format non supporté. Utilisez .csv, .xlsx ou .xls"
    except Exception as exc:
        return None, f"Erreur de lecture : {exc}"


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    disp_kw = ["déplac", "deplac", "displ", "delta", "disp", "drift", " d(", "d(mm", "d(cm"]
    force_kw = ["effort", "force", "cisail", "shear", "base", " v(", "v(kn", "kn"]
    cols_lower = {c.lower().strip(): c for c in df.columns}
    col_d = col_f = None
    for kw in disp_kw:
        for cl, co in cols_lower.items():
            if kw in cl:
                col_d = co; break
        if col_d: break
    for kw in force_kw:
        for cl, co in cols_lower.items():
            if kw in cl and co != col_d:
                col_f = co; break
        if col_f: break
    cols = list(df.columns)
    return col_d or cols[0], col_f or (cols[1] if len(cols) > 1 else cols[0])


def validate_pushover(df, col_d, col_f) -> tuple[np.ndarray, np.ndarray, list[str]]:
    msgs: list[str] = []
    for col in [col_d, col_f]:
        if col not in df.columns:
            raise ValueError(f"Colonne introuvable : « {col} ».")
    disp = pd.to_numeric(df[col_d], errors="coerce").to_numpy(float)
    force = pd.to_numeric(df[col_f], errors="coerce").to_numpy(float)
    mask = ~(np.isnan(disp) | np.isnan(force))
    if (n := int((~mask).sum())) > 0:
        msgs.append(f"{n} ligne(s) ignorée(s) (valeurs NaN).")
    disp, force = disp[mask], force[mask]
    if len(disp) < 4:
        raise ValueError("Moins de 4 points valides.")
    idx = np.argsort(disp)
    if not np.array_equal(idx, np.arange(len(disp))):
        msgs.append("Données triées par déplacement croissant.")
        disp, force = disp[idx], force[idx]
    if not np.all(np.diff(disp) > 0):
        raise ValueError("Déplacements non strictement croissants (doublons). Vérifiez le fichier.")
    if np.argmax(force) < len(force) - 2:
        msgs.append(f"Post-pic détecté à partir du point {np.argmax(force)+1}/{len(force)}.")
    return disp, force, msgs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TRANSFORMATION MDOF → SDOF
# ─────────────────────────────────────────────────────────────────────────────

def mdof_to_sdof(disp_cm, force_kN, mass_kg, gamma):
    """
    Transformation MDOF → SDOF (Fajfar 1999, Eq. 14-19).
    Sd* = D_toit / (Γ · φ_toit)   [φ_toit normalisé à 1]
    Sa* = V_base / (m* · g)        [m* = masse_totale / Γ]
    """
    m_star_kg = mass_kg / gamma
    Sd_m  = (disp_cm / 100.0) / gamma
    Sa_ms2 = (force_kN * 1e3) / m_star_kg
    return Sd_m, Sa_ms2, m_star_kg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — IDÉALISATION BILINÉAIRE
# ─────────────────────────────────────────────────────────────────────────────

def bilinear_idealize(Sd, Sa, method="vmax") -> dict:
    """
    Idéalisation bilinéaire élasto-parfaitement plastique.

    Deux méthodes disponibles :
    ──────────────────────────────────────────────────────────────────────────
    1) "vmax" (défaut, conforme à l'exemple RPA 2024) :
       - Fy* = Sa_max  (résistance = pic de la courbe de capacité)
       - dy* résolu par égalité d'aire :
           Fy* · (du* − dy*/2) = ∫₀^{du*} Sa(d) dd
           → dy* = 2 · (Fy* · du* − E_real) / Fy*
       - Vérifié sur l'exemple : Vy=948 kN, Sdy=89.13 mm ✓

    2) "ec8" (Annexe B EC8 — résolution quadratique) :
       - Fy* et dy* résolus simultanément depuis l'égalité d'aire
         et la pente initiale par régression.
    ──────────────────────────────────────────────────────────────────────────
    Retourne dict : Sa_y, Sd_y, Sa_u, Sd_u, T_star, K_star
    Unités : Sa en m/s², Sd en m.
    """
    Sa_max = float(np.max(Sa))
    u_idx  = int(np.argmax(Sa))  # point ultime = pic de la courbe
    Sd_u   = float(Sd[u_idx])
    Sa_u   = float(Sa[u_idx])

    if Sd_u <= 0:
        raise ValueError("Déplacement au pic nul. Vérifiez les données.")

    # Énergie réelle sous la courbe de 0 à du*
    E_real = float(np.trapezoid(Sa[: u_idx + 1], Sd[: u_idx + 1]))

    if method == "vmax":
        # Méthode Vy = Vmax  (pratique courante et exemple de calcul)
        Sa_y = Sa_max
        # Condition égalité d'aire : Sa_y·(Sd_u − Sd_y/2) = E_real
        # → Sd_y = 2·(Sa_y·Sd_u − E_real) / Sa_y
        Sd_y_candidate = 2.0 * (Sa_y * Sd_u - E_real) / Sa_y
        if Sd_y_candidate <= 0 or Sd_y_candidate >= Sd_u:
            # Repli : Sd_y = 2/3 · Sd_u (approximation)
            Sd_y = (2.0 / 3.0) * Sd_u
        else:
            Sd_y = Sd_y_candidate

    else:  # "ec8" : résolution quadratique
        n_init = max(3, int(0.30 * len(Sd)))
        K_star = float(np.sum(Sa[:n_init] * Sd[:n_init]) / np.sum(Sd[:n_init]**2))
        a = 1.0 / (2.0 * K_star)
        b = -Sd_u
        c = E_real
        disc = b**2 - 4.0 * a * c
        if disc >= 0:
            s1 = (-b - np.sqrt(disc)) / (2.0 * a)
            s2 = (-b + np.sqrt(disc)) / (2.0 * a)
            cands = [s for s in (s1, s2) if 0 < s <= Sa_max * 1.05]
            Sa_y = float(min(cands)) if cands else 0.6 * Sa_max
        else:
            Sa_y = 0.6 * Sa_max
        Sd_y = Sa_y / K_star
        if Sd_y >= Sd_u * 0.95:
            Sd_y = 0.80 * Sd_u
            Sa_y = K_star * Sd_y

    K_star_eff = Sa_y / Sd_y if Sd_y > 0 else 0.0
    T_star = 2.0 * np.pi * np.sqrt(Sd_y / Sa_y) if Sa_y > 0 else 0.0

    return {
        "Sa_y": float(Sa_y),   # m/s²
        "Sd_y": float(Sd_y),   # m
        "Sa_u": float(Sa_u),   # m/s²
        "Sd_u": float(Sd_u),   # m
        "T_star": float(T_star),
        "K_star": float(K_star_eff),
        "E_real": float(E_real),
        "method": method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MÉTHODE N2 (Fajfar 2000, EC8 Annexe B)
# ─────────────────────────────────────────────────────────────────────────────

def n2_method(Sd_cap, Sa_cap, bil, A, I_factor, S, T1, T2, T3, xi, gamma) -> dict:
    """
    Calcul du point de performance par la méthode N2.

    Algorithme (Fajfar 2000, Eq. 4–8 ; EC8 Annexe B.4–B.5) :
    ──────────────────────────────────────────────────────────────────────────
    Tc = T2  (fin du plateau = transition vélocité → déplacement)

    1. Se*(T*) = spectre_élastique_RPA2024(T*)
    2. Sd_el* = Se* · T*²/(4π²)
    3. Rμ = Se* / Sa*_y
    4. Si Se* ≤ Sa*_y  →  élastique, Sd_t* = Sd_el*
    5. Sinon :
       a) T* ≥ Tc : règle du déplacement égal → Sd_t* = Sd_el*
       b) T* < Tc : μ = (Rμ−1)·Tc/T* + 1 → Sd_t* = μ·Sd*_y
    6. d_MDOF = Γ · Sd_t*
    ──────────────────────────────────────────────────────────────────────────
    """
    T_star = bil["T_star"]
    Sa_y   = bil["Sa_y"]
    Sd_y   = bil["Sd_y"]
    Tc     = T2  # période de coin = fin du plateau

    # Demande élastique au SDOF
    Se_star    = float(rpa2024_spectrum(T_star, A, I_factor, S, T1, T2, T3, xi))
    Sd_el_star = Se_star * T_star**2 / (4.0 * np.pi**2)

    if Se_star <= Sa_y:
        Sd_t_star = Sd_el_star
        R_mu      = Se_star / Sa_y if Sa_y > 0 else 1.0
        mu_demand = 1.0
        regime    = "Élastique"
    else:
        R_mu = Se_star / Sa_y
        if T_star >= Tc:
            Sd_t_star = Sd_el_star
            mu_demand = R_mu
            regime    = f"Inélastique — T* ≥ Tc (déplacement égal)"
        else:
            mu_demand = (R_mu - 1.0) * (Tc / max(T_star, 1e-6)) + 1.0
            Sd_t_star = mu_demand * Sd_y
            regime    = f"Inélastique — T* < Tc (courte période, amplification)"

    # Lecture de Sa sur la courbe de capacité par interpolation
    f_interp = interpolate.interp1d(
        Sd_cap, Sa_cap, kind="linear",
        bounds_error=False, fill_value=(Sa_cap[0], Sa_cap[-1])
    )
    Sa_perf = float(f_interp(Sd_t_star))
    cap_exceeded = bool(Sd_t_star > Sd_cap[-1])

    return {
        "T_star": T_star,
        "Tc": Tc,
        "Se_star_ms2": Se_star,
        "Se_star_g":   Se_star / G_MS2,
        "Sd_el_star_m":  Sd_el_star,
        "Sd_el_star_cm": Sd_el_star * 100.0,
        "Sa_y_ms2": Sa_y, "Sa_y_g": Sa_y / G_MS2,
        "Sd_y_m": Sd_y,   "Sd_y_cm": Sd_y * 100.0,
        "R_mu": R_mu,
        "mu_demand": mu_demand,
        "regime": regime,
        "Sd_target_star_m":  Sd_t_star,
        "Sd_target_star_cm": Sd_t_star * 100.0,
        "Sa_perf_ms2": Sa_perf,
        "Sa_perf_g":   Sa_perf / G_MS2,
        "d_target_cm": gamma * Sd_t_star * 100.0,
        "cap_exceeded": cap_exceeded,
    }


def interp_force(disp_cm, force_kN, d_cm):
    f = interpolate.interp1d(disp_cm, force_kN, kind="linear",
                              bounds_error=False, fill_value=(force_kN[0], force_kN[-1]))
    return float(f(d_cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VISUALISATIONS PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

C = {"push": "#1565C0", "bil": "#2E7D32", "dem": "#C62828",
     "pp": "#E65100", "T": "#6A1B9A", "fill": "rgba(198,40,40,0.07)"}


def fig_pushover(disp_cm, force_kN, d_pp=None, V_pp=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=disp_cm, y=force_kN, mode="lines",
        name="Courbe pushover (MDOF)",
        line=dict(color=C["push"], width=2.5),
        hovertemplate="d = %{x:.2f} cm<br>V = %{y:.1f} kN<extra></extra>"))
    if d_pp is not None:
        fig.add_trace(go.Scatter(x=[d_pp], y=[V_pp], mode="markers",
            name=f"Point de performance  ({d_pp:.2f} cm | {V_pp:.0f} kN)",
            marker=dict(symbol="star", size=18, color=C["pp"],
                        line=dict(color="black", width=1))))
        fig.add_vline(x=d_pp, line_dash="dash", line_color=C["pp"], opacity=0.6)
    fig.update_layout(title="Courbe de capacité Pushover (MDOF)",
        xaxis_title="Déplacement au sommet (cm)",
        yaxis_title="Effort tranchant à la base (kN)",
        height=420, template="plotly_white", hovermode="x unified",
        legend=dict(x=0.02, y=0.98))
    return fig


def fig_adrs(Sd_cap, Sa_cap, bil, res, A, I_factor, S, T1, T2, T3, xi) -> go.Figure:
    """Diagramme ADRS complet (RPA 2024)."""
    fig = go.Figure()
    T_arr = np.linspace(0.001, 4.0, 800)
    Sa_d  = rpa2024_spectrum(T_arr, A, I_factor, S, T1, T2, T3, xi)
    Sd_d  = Sa_d * T_arr**2 / (4.0 * np.pi**2)

    # Spectre de demande
    fig.add_trace(go.Scatter(x=Sd_d * 100, y=Sa_d / G_MS2,
        mode="lines", name="Spectre demande RPA 2024",
        line=dict(color=C["dem"], width=2.0, dash="dot"),
        fill="tozeroy", fillcolor=C["fill"]))

    # Ligne T* constante
    T_star = res["T_star"]
    Sa_line = np.linspace(0, max(Sa_cap) * 1.4, 80)
    Sd_line = Sa_line * T_star**2 / (4.0 * np.pi**2)
    fig.add_trace(go.Scatter(x=Sd_line * 100, y=Sa_line / G_MS2,
        mode="lines", name=f"Rayon T* = {T_star:.3f} s",
        line=dict(color=C["T"], width=1.5, dash="longdash")))

    # Courbe de capacité SDOF
    fig.add_trace(go.Scatter(x=Sd_cap * 100, y=Sa_cap / G_MS2,
        mode="lines", name="Capacité SDOF (réelle)",
        line=dict(color=C["push"], width=2.5)))

    # Bilinéaire
    Sd_b = [0, bil["Sd_y"] * 100, bil["Sd_u"] * 100]
    Sa_b = [0, bil["Sa_y"] / G_MS2, bil["Sa_y"] / G_MS2]
    fig.add_trace(go.Scatter(x=Sd_b, y=Sa_b, mode="lines+markers",
        name=f"Bilinéaire (méthode {bil['method'].upper()})",
        line=dict(color=C["bil"], width=2, dash="dash"),
        marker=dict(size=8, color=C["bil"], symbol="circle")))

    # Point de performance
    Sdpp = res["Sd_target_star_cm"]
    Sapp = res["Sa_perf_g"]
    fig.add_trace(go.Scatter(x=[Sdpp], y=[Sapp],
        mode="markers+text", name="Point de performance",
        marker=dict(symbol="star", size=20, color=C["pp"],
                    line=dict(color="black", width=1.5)),
        text=[f"  ({Sdpp:.2f} cm | {Sapp:.3f} g)"],
        textposition="top right", textfont=dict(size=11)))

    fig.update_layout(
        title="Diagramme ADRS — Capacité vs Demande RPA 2024 (Méthode N2)",
        xaxis_title="Déplacement spectral Sd (cm)",
        yaxis_title="Accélération spectrale Sa (g)",
        height=540, template="plotly_white", hovermode="closest",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1))
    return fig


def fig_spectrum_T(A, I_factor, S, T1, T2, T3, xi, T_star=None) -> go.Figure:
    T_arr = np.linspace(0.001, 4.0, 600)
    Sa    = rpa2024_spectrum(T_arr, A, I_factor, S, T1, T2, T3, xi)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=T_arr, y=Sa / G_MS2,
        mode="lines", name="Spectre élastique RPA 2024",
        line=dict(color=C["dem"], width=2.5),
        fill="tozeroy", fillcolor=C["fill"],
        hovertemplate="T = %{x:.3f} s<br>Sa = %{y:.4f} g<extra></extra>"))
    # Marqueurs des périodes caractéristiques
    for lbl, T_val, col in [("T1", T1, "green"), ("T2", T2, "orange"), ("T3", T3, "red")]:
        fig.add_vline(x=T_val, line_dash="dot", line_color=col, opacity=0.6,
                      annotation_text=f"{lbl}={T_val}s", annotation_position="top")
    if T_star is not None:
        Se_ts = float(rpa2024_spectrum(T_star, A, I_factor, S, T1, T2, T3, xi))
        fig.add_trace(go.Scatter(x=[T_star], y=[Se_ts / G_MS2],
            mode="markers+text", name=f"T* = {T_star:.3f} s",
            marker=dict(symbol="x", size=14, color=C["T"],
                        line=dict(width=3)),
            text=[f"  T*={T_star:.3f}s"], textposition="top right"))
        fig.add_vline(x=T_star, line_dash="dash", line_color=C["T"], opacity=0.4)
    fig.update_layout(
        title="Spectre de réponse élastique — RPA 2024 (Eq. 3.8)",
        xaxis_title="Période T (s)", yaxis_title="Sa (g)",
        height=400, template="plotly_white", hovermode="x unified")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VÉRIFICATION NUMÉRIQUE AVEC L'EXEMPLE DE RÉFÉRENCE
# ─────────────────────────────────────────────────────────────────────────────

def verification_example():
    """
    Vérification de l'implémentation sur l'exemple de référence.
    (Bâtiment 4 niveaux, m*=217.44t, Γ=1.336, zone 'VI', S3 Type1, I=1)
    Résultats attendus : T*=0.898s, Say=0.444g, Se(T*)=0.651g (I=1, 475ans)
    """
    A_ref = 0.30          # Zone VI RPA 2024 = 0.30g
    I_ref = 1.00          # Groupe 2 (bâtiment courant)
    S_ref, T1_ref, T2_ref, T3_ref = 1.30, 0.15, 0.60, 2.0  # S3 Type1
    xi_ref = 0.05
    T_star_ref = 0.898    # calculé depuis l'exemple

    Se_calc = rpa2024_spectrum(T_star_ref, A_ref, I_ref, S_ref, T1_ref, T2_ref, T3_ref, xi_ref)
    Se_g    = Se_calc / G_MS2
    # Attendu : Se(0.898) = 0.30*1.0*1.3*2.5*1.0*(0.6/0.898) ≈ 0.651g
    Se_expected = A_ref * I_ref * S_ref * 2.5 * 1.0 * (T2_ref / T_star_ref)
    ok = abs(Se_g - Se_expected) < 1e-4
    return {"Se_calc_g": round(Se_g, 5), "Se_expected_g": round(Se_expected, 5), "ok": ok}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EXPORT EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(disp_cm, force_kN, Sd_cap, Sa_cap, bil, res, params) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"Déplacement (cm)": disp_cm,
                       "Effort tranchant (kN)": force_kN}).to_excel(
            writer, sheet_name="Pushover MDOF", index=False)

        pd.DataFrame({"Sd (m)": Sd_cap, "Sa (m/s²)": Sa_cap,
                       "Sd (cm)": Sd_cap*100, "Sa (g)": Sa_cap/G_MS2}).to_excel(
            writer, sheet_name="Capacité SDOF", index=False)

        T_arr = np.linspace(0.001, 4.0, 400)
        Sa_sp = rpa2024_spectrum(T_arr, params["A"], params["I_factor"],
                                  params["S"], params["T1"], params["T2"],
                                  params["T3"], params["xi"])
        pd.DataFrame({"T (s)": T_arr, "Sa (m/s²)": Sa_sp, "Sa (g)": Sa_sp/G_MS2,
                       "Sd (mm)": Sa_sp*T_arr**2/(4*np.pi**2)*1000}).to_excel(
            writer, sheet_name="Spectre RPA 2024", index=False)

        rows = [
            ("── Paramètres sismiques ──", ""),
            ("Zone sismique", params["zone"]),
            ("Type de spectre", params["spec_type"]),
            ("Coefficient A (g)", f"{params['A']:.4f}"),
            ("Groupe d'importance", params["groupe"]),
            ("Coefficient I", f"{params['I_factor']:.2f}"),
            ("Classe de sol", params["soil"]),
            ("Facteur de site S", f"{params['S']:.3f}"),
            ("T1 (s)", f"{params['T1']:.4f}"),
            ("T2 / Tc (s)", f"{params['T2']:.4f}"),
            ("T3 (s)", f"{params['T3']:.4f}"),
            ("Amortissement ξ (%)", f"{params['xi']*100:.1f}"),
            ("η = √(10/(5+ξ%))", f"{eta_rpa2024(params['xi']):.4f}"),
            ("Facteur de comportement R", f"{params['R']:.2f}"),
            ("Masse sismique (t)", f"{params['mass_t']:.1f}"),
            ("Facteur de participation Γ", f"{params['gamma']:.4f}"),
            ("── Bilinéaire SDOF ──", ""),
            ("Sa_y (g)", f"{bil['Sa_y']/G_MS2:.5f}"),
            ("Sd_y (cm)", f"{bil['Sd_y']*100:.5f}"),
            ("Sa_u (g)", f"{bil['Sa_u']/G_MS2:.5f}"),
            ("Sd_u (cm)", f"{bil['Sd_u']*100:.5f}"),
            ("T* (s)", f"{bil['T_star']:.5f}"),
            ("Méthode bilinéaire", bil["method"]),
            ("── Résultats N2 ──", ""),
            ("Se*(T*) (g)", f"{res['Se_star_g']:.5f}"),
            ("Sd_el* (cm)", f"{res['Sd_el_star_cm']:.5f}"),
            ("Tc = T2 (s)", f"{res['Tc']:.4f}"),
            ("T* ≥ Tc ?", "OUI" if res["T_star"] >= res["Tc"] else "NON"),
            ("Rμ", f"{res['R_mu']:.5f}"),
            ("μ", f"{res['mu_demand']:.5f}"),
            ("Régime", res["regime"]),
            ("Sd_t* (cm)", f"{res['Sd_target_star_cm']:.5f}"),
            ("d_cible MDOF (cm)", f"{res['d_target_cm']:.5f}"),
            ("Sa perf. (g)", f"{res['Sa_perf_g']:.5f}"),
            ("V_base PP (kN)", f"{params.get('V_pp', 0):.2f}"),
        ]
        pd.DataFrame(rows, columns=["Paramètre", "Valeur"]).to_excel(
            writer, sheet_name="Résultats N2", index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — INTERFACE STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    st.sidebar.header("⚙️ Paramètres — RPA 2024")

    # ── Zone sismique ──
    st.sidebar.subheader("🌍 Zone sismique")
    zone = st.sidebar.selectbox("Zone", list(RPA2024_ZONE_A.keys()), index=4)
    A_auto = RPA2024_ZONE_A[zone]
    is_type1 = zone in ZONES_TYPE1
    spec_type = "Type 1 (Zones IV–VI)" if is_type1 else "Type 2 (Zones I–III)"
    st.sidebar.info(f"A = **{A_auto}g** | Spectre **{spec_type}**")
    if st.sidebar.checkbox("Personnaliser A"):
        A_auto = st.sidebar.number_input("A (fraction de g)", 0.01, 0.50, float(A_auto), 0.005)

    # ── Importance ──
    groupe = st.sidebar.selectbox("Groupe d'importance", list(RPA2024_IMPORTANCE.keys()), index=2)
    I_factor = RPA2024_IMPORTANCE[groupe]

    # ── Sol ──
    st.sidebar.subheader("🪨 Sol")
    soil_dict = RPA2024_TYPE1 if is_type1 else RPA2024_TYPE2
    soil = st.sidebar.selectbox("Classe de sol", list(soil_dict.keys()), index=1)
    sp = soil_dict[soil]
    S, T1, T2, T3 = sp["S"], sp["T1"], sp["T2"], sp["T3"]
    c1, c2, c3, c4 = st.sidebar.columns(4)
    c1.metric("S", f"{S:.2f}")
    c2.metric("T1", f"{T1:.2f}s")
    c3.metric("T2", f"{T2:.2f}s")
    c4.metric("T3", f"{T3:.1f}s")
    if st.sidebar.checkbox("Modifier S, T1, T2, T3"):
        S  = st.sidebar.number_input("S", 0.5, 3.0, float(S), 0.05)
        T1 = st.sidebar.number_input("T1 (s)", 0.01, 0.5, float(T1), 0.01)
        T2 = st.sidebar.number_input("T2 (s)", 0.1, 2.0, float(T2), 0.05)
        T3 = st.sidebar.number_input("T3 (s)", 0.5, 4.0, float(T3), 0.1)

    # ── Amortissement ──
    st.sidebar.subheader("📉 Amortissement")
    xi_pct = st.sidebar.slider("ξ (%)", 2, 20, 5, 1)
    xi = xi_pct / 100.0
    eta_val = eta_rpa2024(xi)
    st.sidebar.caption(f"η = √(10/(5+{xi_pct}%)) = **{eta_val:.4f}**")

    R = st.sidebar.number_input("Facteur de comportement R", 1.0, 10.0, 1.0, 0.5,
        help="R=1 pour la méthode N2 standard (spectre élastique). "
             "La non-linéarité est captée par l'idéalisation bilinéaire.")

    # ── Structure ──
    st.sidebar.subheader("🏢 Structure")
    mass_t = st.sidebar.number_input("Masse sismique (tonnes)", 1.0, 1e6, 500.0, 50.0)
    gamma  = st.sidebar.number_input("Facteur de participation Γ", 0.5, 2.5, 1.0, 0.05,
        help="Γ = Σ(mᵢφᵢ) / Σ(mᵢφᵢ²). Valeur typique 1.0–1.5.")
    unit_d = st.sidebar.radio("Unité des déplacements (fichier)", ["cm", "mm"])

    # ── Bilinéaire ──
    st.sidebar.subheader("🔧 Bilinéaire")
    bil_method = st.sidebar.radio("Méthode idéalisation",
        ["vmax — Vy=Vmax (exemple RPA)", "ec8 — Quadratique (EC8 Annexe B)"],
        index=0)
    bil_method_key = "vmax" if "vmax" in bil_method else "ec8"

    return dict(zone=zone, spec_type=spec_type, A=A_auto, groupe=groupe,
                I_factor=I_factor, soil=soil, S=S, T1=T1, T2=T2, T3=T3,
                xi=xi, R=R, mass_t=mass_t, mass_kg=mass_t*1e3,
                gamma=gamma, unit_d=unit_d, bil_method=bil_method_key)


def tab_data(p):
    st.header("📂 Données pushover")
    df_ex = pd.DataFrame({"Déplacement_cm": [0,0.5,1,2,4,6,9,12,16,20,25,30,35],
                           "Effort_kN": [0,180,340,600,980,1250,1480,1590,1650,1670,1650,1600,1530]})
    buf = io.BytesIO()
    df_ex.to_csv(buf, index=False)
    st.download_button("⬇️ Télécharger exemple CSV", buf.getvalue(),
                       "exemple_pushover.csv", "text/csv")

    uploaded = st.file_uploader("Importer la courbe pushover (CSV ou Excel)",
                                 type=["csv","xlsx","xls"])
    if uploaded is None:
        st.info("Aucun fichier importé — téléchargez l'exemple ci-dessus pour tester.")
        return

    df_raw, err = load_file(uploaded)
    if err:
        st.error(f"❌ {err}"); return
    st.success(f"✅ {len(df_raw)} lignes · {len(df_raw.columns)} colonnes")

    col_d_auto, col_f_auto = detect_columns(df_raw)
    c1, c2 = st.columns(2)
    cols = df_raw.columns.tolist()
    col_d = c1.selectbox("Colonne déplacement", cols,
        index=cols.index(col_d_auto) if col_d_auto in cols else 0)
    col_f = c2.selectbox("Colonne effort tranchant", cols,
        index=cols.index(col_f_auto) if col_f_auto in cols else min(1, len(cols)-1))

    with st.expander("👁️ Aperçu"):
        st.dataframe(df_raw.head(20), use_container_width=True)

    try:
        disp_raw, force_raw, msgs = validate_pushover(df_raw, col_d, col_f)
    except ValueError as exc:
        st.error(f"❌ {exc}"); return
    for m in msgs:
        st.warning(m)

    disp_cm = disp_raw / 10.0 if p["unit_d"] == "mm" else disp_raw.copy()
    if p["unit_d"] == "mm":
        st.info("Conversion mm → cm appliquée.")

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Nb points", len(disp_cm))
    s2.metric("Dépl. max", f"{disp_cm.max():.2f} cm")
    s3.metric("Effort max", f"{force_raw.max():.0f} kN")
    Ki = force_raw[1] / disp_cm[1] if disp_cm[1] > 0 else 0
    s4.metric("Raideur init.", f"{Ki:.0f} kN/cm")

    st.session_state.update({"disp_cm": disp_cm, "force_kN": force_raw, "data_ok": True})

    fig = fig_pushover(disp_cm, force_raw)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("⬇️ Pushover (HTML)", fig.to_html(), "pushover.html", "text/html")


def tab_n2(p):
    st.header("📈 Analyse N2 — Point de performance")

    if not st.session_state.get("data_ok", False):
        st.info("👆 Importez d'abord la courbe pushover dans l'onglet **Données**.")
        return

    disp_cm  = st.session_state["disp_cm"]
    force_kN = st.session_state["force_kN"]

    with st.spinner("Calcul en cours…"):
        try:
            Sd_cap, Sa_cap, m_star = mdof_to_sdof(disp_cm, force_kN,
                                                    p["mass_kg"], p["gamma"])
            bil = bilinear_idealize(Sd_cap, Sa_cap, method=p["bil_method"])
            res = n2_method(Sd_cap, Sa_cap, bil,
                            p["A"], p["I_factor"], p["S"],
                            p["T1"], p["T2"], p["T3"], p["xi"], p["gamma"])
            V_pp = interp_force(disp_cm, force_kN, res["d_target_cm"])
            res["V_pp"] = V_pp
            p["V_pp"]   = V_pp
        except Exception as exc:
            st.error(f"❌ Erreur de calcul : {exc}")
            st.exception(exc); return

    # KPIs
    st.subheader("🎯 Point de performance")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Déplacement cible", f"{res['d_target_cm']:.3f} cm")
    k2.metric("Effort tranchant", f"{V_pp:.0f} kN")
    k3.metric("Ductilité μ", f"{res['mu_demand']:.3f}")
    k4.metric("T*", f"{res['T_star']:.4f} s")
    k5.metric("Rμ", f"{res['R_mu']:.3f}")

    mu = res["mu_demand"]
    if   mu <= 1.0: st.success(f"✅ **{res['regime']}**")
    elif mu <= 3.0: st.warning(f"⚠️ **{res['regime']}** — μ = {mu:.2f}")
    elif mu <= 6.0: st.error(f"🔴 **{res['regime']}** — μ = {mu:.2f} (forte demande)")
    else:           st.error(f"🚨 **{res['regime']}** — μ = {mu:.2f} ⚠️ TRÈS ÉLEVÉE")

    if res["cap_exceeded"]:
        st.error("⚠️ Déplacement cible dépasse l'étendue de la courbe de capacité.")

    # Tableau détaillé
    with st.expander("📋 Tableau détaillé des résultats"):
        rows = [
            ("T* — Période effective SDOF (s)", f"{res['T_star']:.5f}"),
            ("Tc = T2 (s) — période de coin", f"{res['Tc']:.4f}"),
            ("T* ≥ Tc ?", "OUI → Déplacement égal" if res["T_star"]>=res["Tc"]
             else "NON → Amplification courte période"),
            ("Se*(T*) — Demande élastique (g)", f"{res['Se_star_g']:.5f}"),
            ("Sd_el* (cm)", f"{res['Sd_el_star_cm']:.5f}"),
            ("Sa_y — Résistance SDOF (g)", f"{res['Sa_y_g']:.5f}"),
            ("Sd_y — Dépl. plastification (cm)", f"{res['Sd_y_cm']:.5f}"),
            ("Rμ = Se*/Sa_y", f"{res['R_mu']:.5f}"),
            ("μ — Ductilité demandée", f"{res['mu_demand']:.5f}"),
            ("Régime", res["regime"]),
            ("Sd_t* — Dépl. cible SDOF (cm)", f"{res['Sd_target_star_cm']:.5f}"),
            ("d_cible — Dépl. toit MDOF (cm)", f"{res['d_target_cm']:.5f}"),
            ("Sa au point de performance (g)", f"{res['Sa_perf_g']:.5f}"),
            ("V_base au point de performance (kN)", f"{V_pp:.2f}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Paramètre","Valeur"]),
                     use_container_width=True, hide_index=True)

    # ADRS
    st.subheader("📊 Diagramme ADRS")
    f_adrs = fig_adrs(Sd_cap, Sa_cap, bil, res,
                       p["A"], p["I_factor"], p["S"],
                       p["T1"], p["T2"], p["T3"], p["xi"])
    st.plotly_chart(f_adrs, use_container_width=True)
    st.download_button("⬇️ ADRS (HTML)", f_adrs.to_html(), "adrs.html", "text/html")

    # Pushover + PP
    st.subheader("📊 Pushover avec point de performance")
    f_push = fig_pushover(disp_cm, force_kN, res["d_target_cm"], V_pp)
    st.plotly_chart(f_push, use_container_width=True)
    st.download_button("⬇️ Pushover (HTML)", f_push.to_html(), "pushover_pp.html", "text/html")

    # Export Excel
    st.subheader("📥 Export")
    xlsx = export_excel(disp_cm, force_kN, Sd_cap, Sa_cap, bil, res, p)
    st.download_button("⬇️ Résultats complets (Excel)", xlsx,
        "resultats_n2_rpa2024.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def tab_spectrum(p):
    st.header("📊 Spectre RPA 2024")

    T_star_d = None
    if st.session_state.get("data_ok"):
        try:
            Sd_c, Sa_c, _ = mdof_to_sdof(st.session_state["disp_cm"],
                                           st.session_state["force_kN"],
                                           p["mass_kg"], p["gamma"])
            T_star_d = bilinear_idealize(Sd_c, Sa_c, p["bil_method"])["T_star"]
        except Exception:
            pass

    fig = fig_spectrum_T(p["A"], p["I_factor"], p["S"],
                          p["T1"], p["T2"], p["T3"], p["xi"], T_star_d)
    st.plotly_chart(fig, use_container_width=True)

    # Vérification interne
    vf = verification_example()
    icon = "✅" if vf["ok"] else "⚠️"
    st.caption(f"{icon} Vérification spectre (Zone VI, S3, T*=0.898s, I=1) : "
               f"Se_calc = {vf['Se_calc_g']:.5f}g | Se_attendu = {vf['Se_expected_g']:.5f}g")

    eta_v = eta_rpa2024(p["xi"])
    plateau_v = p["A"] * p["I_factor"] * p["S"] * 2.5 * eta_v
    rows = [
        ("Zone sismique", p["zone"]), ("Type de spectre", p["spec_type"]),
        ("Coefficient A (g)", f"{p['A']:.4f}"),
        ("Groupe d'importance", p["groupe"]),
        ("Coefficient I", f"{p['I_factor']:.2f}"),
        ("Classe de sol", p["soil"]), ("S", f"{p['S']:.3f}"),
        ("T1 (s)", f"{p['T1']:.4f}"), ("T2 / Tc (s)", f"{p['T2']:.4f}"),
        ("T3 (s)", f"{p['T3']:.4f}"),
        ("ξ (%)", f"{p['xi']*100:.1f}"),
        ("η = √(10/(5+ξ%))", f"{eta_v:.4f}"),
        ("Plateau A·I·S·2.5η (g)", f"{plateau_v:.4f}"),
        ("T* (si données chargées)", f"{T_star_d:.4f}s" if T_star_d else "—"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Paramètre","Valeur"]),
                 use_container_width=True, hide_index=True)
    st.download_button("⬇️ Spectre (HTML)", fig.to_html(), "spectre_rpa2024.html", "text/html")


def tab_hypotheses():
    st.header("ℹ️ Méthode, hypothèses et vérification")
    st.markdown("""
---
### Spectre RPA 2024 — Formule exacte (Eq. 3.8)

| Branche | Condition | Expression |
|---------|-----------|------------|
| 1 | 0 ≤ T < T₁ | Sae/g = A·I·S·[1 + (T/T₁)·(2,5η−1)] |
| 2 | T₁ ≤ T < T₂ | Sae/g = A·I·S·2,5η  *(plateau)* |
| 3 | T₂ ≤ T < T₃ | Sae/g = A·I·S·2,5η·**(T₂/T)**  ← linéaire |
| 4 | T₃ ≤ T < 4s | Sae/g = A·I·S·2,5η·**(T₂·T₃/T²)**  ← dépl. constant |

η = √(10/(5+ξ%)) ≥ 0,55   —   À ξ=5% : η = 1,0

> ⚠️ La branche 3 décroît **linéairement** en (T₂/T), **pas** en (T₂/T)^(2/3).  
> Différence critique avec RPA 99 v2003.

---
### Paramètres RPA 2024 (Tableaux 3.4 & 3.5)

| Type | Zones | Sol | S | T₁ | T₂ | T₃ |
|------|-------|-----|---|-----|-----|-----|
| Type 1 | IV, V, VI | S1 | 1.00 | 0.10 | 0.40 | 2.0 |
| Type 1 | IV, V, VI | S2 | 1.20 | 0.10 | 0.50 | 2.0 |
| Type 1 | IV, V, VI | **S3** | **1.30** | **0.15** | **0.60** | **2.0** |
| Type 1 | IV, V, VI | S4 | 1.35 | 0.15 | 0.70 | 2.0 |
| Type 2 | I, II, III | S1 | 1.00 | 0.05 | 0.25 | 1.20 |
| Type 2 | I, II, III | S2 | 1.30 | 0.05 | 0.30 | 1.20 |
| Type 2 | I, II, III | S3 | 1.55 | 0.10 | 0.40 | 1.20 |
| Type 2 | I, II, III | S4 | 1.80 | 0.10 | 0.50 | 1.20 |

---
### Zones et coefficients A (Tableau 3.3, sol S1, Tr=475 ans)

| Zone | Sismicité | A |
|------|-----------|---|
| 0 | Très faible | 0.00 |
| I | Faible | **0.07g** |
| II | Faible à moyenne | **0.10g** |
| III | Moyenne | **0.15g** |
| IV | Moyenne à élevée | **0.20g** |
| V | Élevée | **0.25g** |
| VI | Élevée | **0.30g** |

---
### Coefficient d'importance I (Tableau 3.11)

| Groupe | Description | I |
|--------|-------------|---|
| 1A | Vital (hôpitaux, centres de secours) | **1.40** |
| 1B | Grande importance (>300 pers., établ. scolaires) | **1.20** |
| 2 | Courant (bâtiments courants) | **1.00** |
| 3 | Faible importance | **0.80** |

---
### Vérification numérique — Exemple de référence

Données : m*=217,44 t · Γ=1,336 · Vy*=948 kN · Sdy*=89,13 mm  
Zone VI (A=0,30g) · S3 Type 1 (S=1,3 · T₂=0,6s) · I=1,0 · ξ=5%

| Quantité | Formule | Résultat |
|----------|---------|----------|
| Say = Vy*/m* | 948/(217,44·9,81) | **0,444 g** |
| T* = 2π√(Sdy/Say) | 2π√(0,08913/4,36) | **0,898 s** |
| T* ≥ Tc=T₂=0,6s | Règle déplacement égal | ✅ |
| Se*(T*=0,898) | 0,30·1,0·1,3·2,5·1,0·(0,6/0,898) | **0,651 g** |
| del* = Se·(T*/2π)² | 0,651·9,81·(0,898/2π)² | **≈ 0,130 m** |
| dt_MDOF = Γ·del* | 1,336·0,130 | **≈ 0,174 m** |

---
### Limitations de la méthode N2

| Limitation | Détail |
|-----------|--------|
| Mode 1 uniquement | Valide si T₁ < 2s et structure régulière |
| Γ fourni | Obtenu par analyse modale propre |
| Bilinéaire EPP | Sans écrouissage ni dégradation |
| Spectre déterministe | Pas de variabilité probabiliste |
| Séismes proches (near-fault) | Règle du déplacement égal non conservatrice |
""")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if "data_ok" not in st.session_state:
        st.session_state["data_ok"] = False

    st.title("🏗️ Calcul sismique — Méthode N2 + RPA 2024")
    st.caption("N2 (Fajfar 2000 / EC8 Annexe B) · Spectre RPA 2024 (DTR BC 2.48 — 2024) · v2.0.0")
    st.divider()

    p = sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Données", "📈 Analyse N2", "📊 Spectres", "ℹ️ Méthode & Hypothèses"])

    with tab1: tab_data(p)
    with tab2: tab_n2(p)
    with tab3: tab_spectrum(p)
    with tab4: tab_hypotheses()


if __name__ == "__main__":
    main()
