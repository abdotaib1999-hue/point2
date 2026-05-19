import io
import math
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import interp1d

G = 9.81


@dataclass
class SpectrumParams:
    zone: float
    soil: str
    damping: float
    t1: float
    q: float
    weight_kn: float
    mass_t: float
    ag_ratio: float
    s_factor: float
    t_b: float
    t_c: float
    t_d: float
    eta: float


def set_page():
    st.set_page_config(page_title="Calcul sismique N2", layout="wide")
    st.title("Application de calcul sismique par méthode N2")
    st.caption("Import pushover, bilinéarisation, point de performance, spectre RPA 2024, exports.")


def read_input_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Aucun fichier importé.")
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format non supporté. Utilisez CSV, XLS ou XLSX.")
    if df.empty:
        raise ValueError("Le fichier est vide.")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols).copy()
    return df


def detect_columns(df: pd.DataFrame):
    disp_candidates = ["deplacement", "déplacement", "disp", "displacement", "drift", "u", "delta"]
    shear_candidates = ["effort tranchant", "effort lateral", "effort latéral", "base shear", "shear", "vb", "force", "base", "f"]
    disp_col = next((c for c in df.columns if any(k in c for k in disp_candidates)), None)
    shear_col = next((c for c in df.columns if any(k in c for k in shear_candidates)), None)
    return disp_col, shear_col


def validate_pushover(df: pd.DataFrame, disp_col: str, shear_col: str):
    if disp_col is None or shear_col is None:
        raise ValueError("Colonnes introuvables. Le fichier doit contenir le déplacement et l'effort tranchant/base.")
    work = df[[disp_col, shear_col]].copy()
    work.columns = ["disp", "shear"]
    work["disp"] = pd.to_numeric(work["disp"], errors="coerce")
    work["shear"] = pd.to_numeric(work["shear"], errors="coerce")
    work = work.dropna()
    if len(work) < 3:
        raise ValueError("Données insuffisantes après nettoyage.")
    if (work["disp"].diff().dropna() < 0).all():
        work = work.sort_values("disp")
    if not work["disp"].is_monotonic_increasing:
        work = work.sort_values("disp").reset_index(drop=True)
    if work["disp"].duplicated().any():
        work = work.groupby("disp", as_index=False)["shear"].mean()
    if work["disp"].nunique() < 3:
        raise ValueError("La courbe doit contenir au moins 3 abscisses distinctes.")
    if (work["disp"].diff().dropna() <= 0).any():
        raise ValueError("La courbe n'est pas monotone en déplacement. Corrigez les données importées.")
    return work.reset_index(drop=True)


def unit_to_m(value, unit):
    return value / 1000.0 if unit == "mm" else value / 100.0


def compute_area(x, y):
    return np.trapz(y, x)


def bilinearize_capacity(sd, sa):
    sd = np.asarray(sd, dtype=float)
    sa = np.asarray(sa, dtype=float)
    if sd[0] != 0:
        sd = np.insert(sd, 0, 0.0)
        sa = np.insert(sa, 0, 0.0)
    if sa[0] != 0:
        sa[0] = 0.0
    k0 = (sa[1] - sa[0]) / max(sd[1] - sd[0], 1e-9)
    best = None
    for i in range(2, len(sd)):
        sdy = sd[i]
        say = k0 * sdy
        if say <= 0:
            continue
        area_orig = compute_area(sd[:i+1], sa[:i+1])
        area_bi = 0.5 * sdy * say + (sd[i] - sdy) * sa[i]
        err = abs(area_bi - area_orig)
        cand = {"sdy": sdy, "say": say, "error": err, "area_orig": area_orig}
        if best is None or err < best["error"]:
            best = cand
    if best is None:
        raise ValueError("Bilinéarisation impossible.")
    return best["sdy"], best["say"]


def hysteretic_reduction_factor(q, ductility):
    return min(1.0, 1.0 / max(q, 1e-9) * (1.0 + 0.1 * max(ductility - 1.0, 0.0)))


def rpa2024_parameters(zone, soil, damping):
    soil = soil.upper()
    soil_map = {
        "S1": {"s": 1.0, "tb": 0.15, "tc": 0.40, "td": 2.0},
        "S2": {"s": 1.2, "tb": 0.20, "tc": 0.60, "td": 2.0},
        "S3": {"s": 1.35, "tb": 0.25, "tc": 0.80, "td": 2.5},
        "S4": {"s": 1.50, "tb": 0.30, "tc": 1.00, "td": 3.0},
    }
    if soil not in soil_map:
        raise ValueError("Type de sol invalide. Choisissez S1, S2, S3 ou S4.")
    if zone <= 0:
        raise ValueError("Zone sismique invalide.")
    if damping <= 0 or damping >= 100:
        raise ValueError("Amortissement incohérent.")
    params = soil_map[soil]
    ag_ratio = 0.10 * zone
    eta = math.sqrt(10.0 / (5.0 + damping))
    return ag_ratio, params["s"], params["tb"], params["tc"], params["td"], eta


def rpa2024_spectrum_sa_sd(p: SpectrumParams, n=500):
    t = np.linspace(0.01, max(4.0, 1.5 * p.t1), n)
    sa = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= p.t_b:
            sa[i] = p.ag_ratio * p.s_factor * (1 + ti / p.t_b * 2)
        elif ti <= p.t_c:
            sa[i] = p.ag_ratio * p.s_factor * 2.0
        elif ti <= p.t_d:
            sa[i] = p.ag_ratio * p.s_factor * 2.0 * (p.t_c / ti)
        else:
            sa[i] = p.ag_ratio * p.s_factor * 2.0 * (p.t_c * p.t_d / (ti ** 2))
    sa = sa * p.eta
    sd = sa * G * (t ** 2) / (4 * math.pi ** 2)
    return t, sa, sd


def capacity_to_adrs(disp_m, shear_kn, mass_t, weight_kn):
    if mass_t <= 0 and weight_kn <= 0:
        raise ValueError("Masse ou poids total requis.")
    if mass_t <= 0:
        mass_t = weight_kn / G
    m_eff = mass_t * 1000.0
    sd = np.asarray(disp_m, dtype=float)
    sa = (np.asarray(shear_kn, dtype=float) * 1000.0) / (m_eff * G)
    return sd, sa


def target_displacement_n2(sd_cap, sa_cap, t1, q, ag_ratio, s_factor, eta):
    if t1 <= 0:
        raise ValueError("T1 doit être positif.")
    k = (4 * math.pi ** 2) / (t1 ** 2)
    sdy, say = bilinearize_capacity(sd_cap, sa_cap)
    mu = max(sd_cap[-1] / max(sdy, 1e-9), 1.0)
    red = hysteretic_reduction_factor(q, mu)
    se_t1 = ag_ratio * s_factor * 2.0 * eta / max(red, 1e-9)
    sd_el = se_t1 * G * (t1 ** 2) / (4 * math.pi ** 2)
    if t1 <= 0.5:
        cd = 1.0
    else:
        cd = 1.0
    sd_target = cd * sd_el
    sa_target = k * sd_target
    return {"sdy": sdy, "say": say, "sd_target": sd_target, "sa_target": sa_target, "mu": mu, "red": red}


def interpolate_capacity(sd, sa):
    f = interp1d(sd, sa, kind="linear", fill_value="extrapolate", bounds_error=False)
    return f


def find_intersection(sd_cap, sa_cap, t1, target_sa):
    fcap = interpolate_capacity(sd_cap, sa_cap)
    sa_at_t = float(fcap(target_sa[0])) if hasattr(target_sa, "__len__") else None
    return sa_at_t


def build_plots(sd_cap, sa_cap, sdy, say, sd_target, sa_target, t, sa_demand, sd_demand):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sd_cap, y=sa_cap, mode="lines+markers", name="Capacité brute"))
    fig1.add_trace(go.Scatter(x=[0, sdy, sd_cap[-1]], y=[0, say, say], mode="lines", name="Bilinéaire"))
    fig1.add_trace(go.Scatter(x=[sd_target], y=[sa_target], mode="markers", marker=dict(size=12), name="Point de performance"))
    fig1.update_layout(xaxis_title="Sd (m)", yaxis_title="Sa (g)", template="plotly_white")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sd_demand, y=sa_demand, mode="lines", name="Demande RPA 2024"))
    fig2.add_trace(go.Scatter(x=sd_cap, y=sa_cap, mode="lines+markers", name="Capacité"))
    fig2.add_trace(go.Scatter(x=[sd_target], y=[sa_target], mode="markers", marker=dict(size=12), name="Performance"))
    fig2.update_layout(xaxis_title="Sd (m)", yaxis_title="Sa (g)", template="plotly_white")
    return fig1, fig2


def main():
    set_page()

    with st.sidebar:
        st.header("Entrée des données")
        uploaded = st.file_uploader("Importer CSV ou Excel", type=["csv", "xlsx", "xls"])
        unit = st.selectbox("Unité déplacement", ["mm", "cm"])
        weight_kn = st.number_input("Poids total W (kN)", min_value=0.0, value=0.0, step=10.0)
        mass_t = st.number_input("Masse sismique m (t)", min_value=0.0, value=0.0, step=10.0)
        q = st.number_input("Facteur de comportement q", min_value=0.1, value=3.0, step=0.1)
        zone = st.number_input("Zone sismique", min_value=0.1, value=3.0, step=0.1)
        soil = st.selectbox("Type de sol", ["S1", "S2", "S3", "S4"])
        damping = st.number_input("Amortissement (%)", min_value=0.1, value=5.0, step=0.5)
        t1 = st.number_input("Période fondamentale T1 (s)", min_value=0.01, value=0.5, step=0.05)

    try:
        if uploaded is None:
            st.info("Importez un fichier de courbe pushover pour commencer.")
            return

        raw = read_input_file(uploaded)
        raw = normalize_columns(raw)
        disp_col, shear_col = detect_columns(raw)
        data = validate_pushover(raw, disp_col, shear_col)

        disp_m = data["disp"].apply(lambda x: unit_to_m(x, unit)).to_numpy()
        shear_kn = data["shear"].to_numpy()

        ag_ratio, s_factor, t_b, t_c, t_d, eta = rpa2024_parameters(zone, soil, damping)
        spec = SpectrumParams(zone, soil, damping, t1, q, weight_kn, mass_t, ag_ratio, s_factor, t_b, t_c, t_d, eta)

        sd_cap, sa_cap = capacity_to_adrs(disp_m, shear_kn, mass_t, weight_kn)
        res = target_displacement_n2(sd_cap, sa_cap, t1, q, ag_ratio, s_factor, eta)

        t, sa_dem, sd_dem = rpa2024_spectrum_sa_sd(spec)

        fig1, fig2 = build_plots(sd_cap, sa_cap, res["sdy"], res["say"], res["sd_target"], res["sa_target"], t, sa_dem, sd_dem)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Courbe de capacité")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.subheader("Capacité et demande")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Résultats")
        out = pd.DataFrame([
            ["Sd,y", res["sdy"], "m"],
            ["Sa,y", res["say"], "g"],
            ["Déplacement cible Sd,target", res["sd_target"], "m"],
            ["Accélération cible Sa,target", res["sa_target"], "g"],
            ["Ductilité mu", res["mu"], "-"],
            ["Facteur de réduction", res["red"], "-"],
            ["Zone", zone, "-"],
            ["Sol", soil, "-"],
            ["Amortissement", damping, "%"],
            ["T1", t1, "s"],
        ], columns=["Paramètre", "Valeur", "Unité"])
        st.dataframe(out, use_container_width=True, hide_index=True)

        result_csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les résultats CSV", result_csv, "resultats_n2.csv", "text/csv")

        fig1_json = fig1.to_json().encode("utf-8")
        fig2_json = fig2.to_json().encode("utf-8")
        st.download_button("Télécharger la figure capacité", fig1_json, "figure_capacite.json", "application/json")
        st.download_button("Télécharger la figure capacité-demande", fig2_json, "figure_demande.json", "application/json")

        with st.expander("Hypothèses de calcul"):
            st.write("Méthode N2 avec bilinéarisation par énergie équivalente.")
            st.write("Le spectre RPA 2024 est paramétré via variables explicites à aligner sur le texte réglementaire officiel.")
            st.write(f"ag/g = {ag_ratio:.3f}, S = {s_factor:.3f}, Tb = {t_b:.3f}, Tc = {t_c:.3f}, Td = {t_d:.3f}, eta = {eta:.3f}")

    except Exception as e:
        st.error(str(e))
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
