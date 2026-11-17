"""Domains and variable descriptions of bnlearn benchmark networks (for LLM prior queries)."""

NETWORKS = {
    "asia": {
        "domain": "pulmonary medicine (lung diseases and their risk factors)",
        "descriptions": {"asia": "patient recently visited Asia", "tub": "patient has tuberculosis", "smoke": "patient smokes",
                         "lung": "patient has lung cancer", "bronc": "patient has bronchitis", "either": "patient has tuberculosis or lung cancer",
                         "xray": "chest X-ray result is abnormal", "dysp": "patient has dyspnoea (shortness of breath)"},
    },
    "sachs": {
        "domain": "protein signaling in human immune T cells (phosphorylated proteins and phospholipids)",
        "descriptions": {"Raf": "Raf kinase (MAPKKK)", "Mek": "Mek kinase (MAPKK)", "Erk": "Erk kinase (p44/42 MAPK)", "Plcg": "phospholipase C gamma",
                         "PIP2": "phosphatidylinositol 4,5-bisphosphate", "PIP3": "phosphatidylinositol 3,4,5-trisphosphate", "Akt": "Akt kinase",
                         "PKA": "protein kinase A", "PKC": "protein kinase C", "P38": "p38 MAPK", "Jnk": "c-Jun N-terminal kinase"},
    },
    "insurance": {"domain": "car insurance risk assessment (driver, vehicle, accident and claim costs)", "descriptions": {}},
    "alarm": {"domain": "patient monitoring in anesthesia and intensive care (the ALARM monitoring network)", "descriptions": {}},
    # Held-out for validation: networks not used when choosing hybrid-mode thresholds
    "child": {"domain": "diagnosis of congenital heart disease in newborn babies (the CHILD network)", "descriptions": {}},
    "hepar2": {"domain": "diagnosis of liver disorders from risk factors, symptoms and lab tests (the HEPAR II network)", "descriptions": {}},
    "win95pts": {"domain": "troubleshooting printing problems in Windows 95 (the WIN95PTS network)", "descriptions": {}},
    "hailfinder": {"domain": "forecasting severe summer hail in northeastern Colorado (the HAILFINDER network)", "descriptions": {}},
}
DEV, HELDOUT = ["asia", "sachs", "insurance", "alarm"], ["child", "hepar2", "win95pts", "hailfinder"]
