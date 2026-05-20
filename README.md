# 🏗️ Calcul Sismique — Méthode N2 + RPA 2024

Application Streamlit de calcul du point de performance sismique par la **méthode N2**
(Fajfar 1999 / EC8 Annexe B) avec le **spectre de demande RPA 2024**.

---

## Fonctionnalités

- Import de courbe pushover (CSV ou Excel)
- Spectre RPA 2024 paramétrable (zone, sol, groupe, amortissement)
- Idéalisation bilinéaire par énergie égale
- Calcul N2 complet (régime élastique et inélastique, règle du déplacement égal)
- Graphiques ADRS interactifs (Plotly)
- Tableau récapitulatif des résultats
- Export Excel multi-feuilles + téléchargement des figures

---

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Structure du projet

```
seismic_n2/
├── app.py               ← Application principale
├── requirements.txt     ← Dépendances Python
├── README.md
└── data/
    └── exemple_pushover.csv   ← (optionnel) données d'exemple
```

---

## Déploiement sur Streamlit Cloud

Voir section dédiée dans la réponse de génération.

---

## Note réglementaire

Les coefficients du spectre sont basés sur **RPA 99 v2003**. Vérifier les valeurs
exactes contre le document officiel **RPA 2024 (DTR BC 2.48 — 2024)**.

---

## Références

- Fajfar P. (1999). EESD 28(9), 979–993.
- EN 1998-1:2004, Annexe B.
- RPA 99 version 2003.
