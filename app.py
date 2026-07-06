# -*- coding: utf-8 -*-
# ============================================================================
#  Dashboard interactivo (Componente 2) — FIFA World Cup 2026
#  Carga el modelo MLP entrenado en el notebook (mlp_model_best.keras, scaler.pkl,
#  feature_metadata.json), reconstruye las 28 features de cada partido desde
#  model_df_features.csv y predice con la MISMA red y normalizador del notebook.
#
#  Mejoras temáticas incluidas:
#   · Banderas de cada selección (flagcdn.com) en grupos, bracket y campeones.
#   · Bracket de DOS LADOS (llave tradicional) que converge a la Final + trofeo.
#   · Confederación de cada selección y marca de país anfitrión (localía real).
#   · Desempate de grupos por diferencia de goles y goles a favor.
#   · % de clasificación por simulación Monte Carlo del grupo (no un softmax fijo).
#   · Historial de enfrentamientos (H2H) real calculado desde el histórico.
#   · Nueva vista "Predecir partido" (Componente 1) con fichas de scouting.
#   · Marcador esperado por cruce, botones de descarga y feedback de carga.
# ============================================================================
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="World Cup 2026 · Inteligencia Deportiva",
                   page_icon="⚽", layout="wide")

# ── Paleta del sistema de diseño (para los estilos en línea del HTML) ─────────
NIGHT, PANEL, PANELHI = "#0a1628", "#0f1f38", "#16294a"
LINE, LIME, AMBER = "#1e3459", "#c8f04a", "#f5a623"
INK, MUTED, LOSS = "#eaf1fb", "#7d93b8", "#3a5f8a"

# ── Hoja de estilos: tema, fuentes, animaciones y ajustes de widgets ─────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@700;800;900&family=Inter:wght@400;500;600;700&display=swap');

.stApp { background: #0a1628; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }

/* Cromo de Streamlit fuera: consola a pantalla completa */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

@keyframes riseIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes barGrow { from { width: 0; } }
.wc-rise { animation: riseIn .4s ease; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

::-webkit-scrollbar { height: 9px; width: 9px; }
::-webkit-scrollbar-thumb { background: #1e3459; border-radius: 5px; }
::-webkit-scrollbar-track { background: transparent; }

/* Cabecera */
.wc-header { border-bottom: 1px solid #1e3459; padding: 6px 0 16px; margin-bottom: 8px;
  background: linear-gradient(180deg, #0f1f38 0%, rgba(10,22,40,0) 100%); }
.wc-title { font-family: 'Archivo', sans-serif; font-weight: 900; font-size: 34px;
  letter-spacing: -.5px; color: #eaf1fb; line-height: 1; }
.wc-sub { display: inline-block; font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
  color: #f5a623; font-weight: 700; margin: 10px 0 0; }
.wc-meta { color: #7d93b8; font-size: 13px; margin: 6px 0 0; }
.wc-meta img { height: 13px; border-radius: 2px; vertical-align: -1px; margin: 0 2px; }
.wc-lead { color: #7d93b8; font-size: 14px; max-width: 760px; margin: 2px 0 18px; }
.wc-lead strong { color: #c8f04a; }

/* Selector de vista (segmented control) como barra de pestañas */
[data-testid="stSegmentedControl"] button { border: none !important; background: transparent !important;
  color: #7d93b8 !important; font-weight: 600 !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important; padding: 8px 16px !important; }
[data-testid="stSegmentedControl"] button:hover { color: #eaf1fb !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
  color: #c8f04a !important; border-bottom: 2px solid #c8f04a !important; background: transparent !important; }

/* Tarjetas de grupo (contenedores con borde) */
[data-testid="stVerticalBlockBorderWrapper"] { background: #0f1f38; border-color: #1e3459 !important;
  border-radius: 10px; }

/* Insignia de letra del grupo */
.wc-badge { display: inline-flex; align-items: center; gap: 10px; }
.wc-badge b { font-family: 'Archivo', sans-serif; font-weight: 900; font-size: 20px; color: #0a1628;
  background: #c8f04a; width: 32px; height: 32px; border-radius: 6px; display: grid; place-items: center; }
.wc-badge span { font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 14px; color: #eaf1fb;
  letter-spacing: 1px; text-transform: uppercase; }
.wc-cols { display: flex; justify-content: space-between; color: #7d93b8; font-size: 10px;
  text-transform: uppercase; letter-spacing: 1px; padding: 4px 6px 2px; font-weight: 600; }

/* Filas-selección: botones de intercambio con aspecto de fila de tabla.
   La bandera se inyecta como ::before con la URL de flagcdn (ver flag_css). */
div[class*="st-key-swap_"] button { width: 100%; justify-content: flex-start; text-align: left;
  background: transparent; border: 1px solid transparent; border-left: 3px solid transparent;
  border-radius: 6px; color: #eaf1fb; font-weight: 500; padding: 7px 10px;
  font-feature-settings: 'tnum'; transition: background .15s; }
div[class*="st-key-swap_"] button:hover { background: #16294a; }
div[class*="st-key-swap_"] button[kind="primary"] { background: #c8f04a; color: #0a1628;
  border-left-color: #c8f04a; font-weight: 700; }

/* ── Bracket de DOS LADOS (llave tradicional del Mundial) ──────────────────
   La mitad izquierda avanza hacia la derecha, la derecha hacia la izquierda,
   y la Final + el trofeo quedan en el centro. En pantallas pequeñas hace
   scroll horizontal como respaldo. */
/* 'safe center' centra cuando cabe pero NO recorta la 1ª columna al desbordar
   (evita que los 16vos de la izquierda queden ocultos); si el navegador no lo
   soporta, cae a flex-start, que también es desplazable. */
.wc-bracket2 { display: flex; align-items: stretch; justify-content: safe center; gap: 8px;
  overflow-x: auto; padding: 4px 2px 14px; }
.wc-round { flex: 1 1 0; min-width: 132px; display: flex; flex-direction: column; }
.wc-round-h { font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 12px; letter-spacing: 1.5px;
  text-transform: uppercase; color: #f5a623; text-align: center; padding-bottom: 5px; margin-bottom: 10px;
  border-bottom: 1px solid #1e3459; }
.wc-round-body { display: flex; flex-direction: column; justify-content: space-around; flex: 1; gap: 10px; }
.wc-match { background: #0f1f38; border: 1px solid #1e3459; border-radius: 8px; overflow: hidden;
  flex: 0 0 auto; position: relative; }
/* Conector: pequeña línea horizontal que une cada cruce con la ronda siguiente */
.wc-round .wc-match::after { content: ""; position: absolute; top: 50%; left: 100%;
  width: 12px; height: 1px; background: #1e3459; }
.wc-round--right .wc-match::after { left: auto; right: 100%; }   /* espejo para la mitad derecha */
.wc-scoreline { text-align: center; font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 10px;
  letter-spacing: 1px; color: #7d93b8; padding: 3px 0; background: #0a1628; border-top: 1px solid #1e3459; }

/* Columna central: trofeo + partido de la Final */
.wc-center { display: flex; flex-direction: column; justify-content: center; align-items: center;
  min-width: 150px; flex: 1 1 0; }
.wc-trophy { font-size: 30px; margin-bottom: 8px; filter: drop-shadow(0 0 8px rgba(200,240,74,.45)); }
.wc-center .wc-round-h { width: 100%; }

.wc-champ { display: inline-flex; align-items: center; gap: 14px; margin: 4px 0 22px;
  background: #16294a; border: 1px solid #c8f04a; border-radius: 10px; padding: 12px 22px; }
.wc-champ img { height: 26px; border-radius: 3px; box-shadow: 0 0 0 1px rgba(0,0,0,.35); }
.wc-champ .lbl { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #7d93b8; font-weight: 700; }
.wc-champ .name { font-family: 'Archivo', sans-serif; font-weight: 900; font-size: 26px; color: #c8f04a; }

/* Ficha de scouting (vista Predecir partido) */
.wc-scout { background: #0f1f38; border: 1px solid #1e3459; border-radius: 10px; padding: 16px 18px; }
.wc-scout .nm { font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 19px; color: #eaf1fb;
  display: flex; align-items: center; gap: 10px; }
.wc-scout .nm img { height: 22px; border-radius: 3px; box-shadow: 0 0 0 1px rgba(0,0,0,.35); }
.wc-scout .cf { font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: #f5a623; font-weight: 700; margin: 4px 0 12px; }
.wc-stat { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; border-bottom: 1px solid #16294a; }
.wc-stat span:first-child { color: #7d93b8; }
.wc-stat span:last-child { color: #eaf1fb; font-weight: 600; font-feature-settings: 'tnum'; }

/* Barra de probabilidad tri-color (vista Predecir partido) */
.wc-tribar { display: flex; height: 30px; border-radius: 7px; overflow: hidden; border: 1px solid #1e3459; margin: 6px 0 4px; }
.wc-tribar div { display: grid; place-items: center; font-size: 12px; font-weight: 700;
  font-family: 'Archivo', sans-serif; color: #0a1628; }

/* Pie de transparencia */
.wc-foot { color: #7d93b8; font-size: 11px; line-height: 1.6; max-width: 900px;
  border-top: 1px solid #1e3459; padding-top: 14px; margin-top: 22px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
#  METADATOS DE PAÍS — ISO2 para la bandera (flagcdn) y confederación
# ============================================================================
# Código ISO 3166-1 alfa-2 en minúsculas: flagcdn sirve la bandera en
# https://flagcdn.com/w40/<iso>.png. Las selecciones británicas usan los
# códigos especiales gb-eng / gb-sct / gb-wls / gb-nir y Kosovo el código xk.
FLAG_ISO = {
    "Afghanistan": "af", "Albania": "al", "Algeria": "dz", "American Samoa": "as",
    "Andorra": "ad", "Angola": "ao", "Anguilla": "ai", "Antigua and Barbuda": "ag",
    "Argentina": "ar", "Armenia": "am", "Aruba": "aw", "Australia": "au", "Austria": "at",
    "Azerbaijan": "az", "Bahamas": "bs", "Bahrain": "bh", "Bangladesh": "bd", "Barbados": "bb",
    "Belarus": "by", "Belgium": "be", "Belize": "bz", "Benin": "bj", "Bermuda": "bm",
    "Bhutan": "bt", "Bolivia": "bo", "Bosnia and Herzegovina": "ba", "Botswana": "bw",
    "Brazil": "br", "British Virgin Islands": "vg", "Bulgaria": "bg", "Burkina Faso": "bf",
    "Burundi": "bi", "Cambodia": "kh", "Cameroon": "cm", "Canada": "ca", "Cape Verde": "cv",
    "Cayman Islands": "ky", "Central African Republic": "cf", "Chad": "td", "Chile": "cl",
    "China": "cn", "Colombia": "co", "Comoros": "km", "Congo": "cg", "Cook Islands": "ck",
    "Costa Rica": "cr", "Croatia": "hr", "Cuba": "cu", "Curaçao": "cw", "Cyprus": "cy",
    "Czechia": "cz", "Côte d'Ivoire": "ci", "DR Congo": "cd", "Denmark": "dk", "Djibouti": "dj",
    "Dominica": "dm", "Dominican Republic": "do", "East Timor": "tl", "Ecuador": "ec",
    "Egypt": "eg", "El Salvador": "sv", "England": "gb-eng", "Equatorial Guinea": "gq",
    "Eritrea": "er", "Estonia": "ee", "Eswatini": "sz", "Ethiopia": "et", "Faroe Islands": "fo",
    "Fiji": "fj", "Finland": "fi", "France": "fr", "Gabon": "ga", "Gambia": "gm",
    "Georgia": "ge", "Germany": "de", "Ghana": "gh", "Gibraltar": "gi", "Greece": "gr",
    "Grenada": "gd", "Guam": "gu", "Guatemala": "gt", "Guinea": "gn", "Guinea-Bissau": "gw",
    "Guyana": "gy", "Haiti": "ht", "Honduras": "hn", "Hong Kong": "hk", "Hungary": "hu",
    "Iceland": "is", "India": "in", "Indonesia": "id", "Iran": "ir", "Iraq": "iq",
    "Israel": "il", "Italy": "it", "Jamaica": "jm", "Japan": "jp", "Jordan": "jo",
    "Kazakhstan": "kz", "Kenya": "ke", "Kosovo": "xk", "Kuwait": "kw", "Kyrgyzstan": "kg",
    "Laos": "la", "Latvia": "lv", "Lebanon": "lb", "Lesotho": "ls", "Liberia": "lr",
    "Libya": "ly", "Liechtenstein": "li", "Lithuania": "lt", "Luxembourg": "lu", "Macau": "mo",
    "Madagascar": "mg", "Malawi": "mw", "Malaysia": "my", "Maldives": "mv", "Mali": "ml",
    "Malta": "mt", "Mauritania": "mr", "Mauritius": "mu", "Mexico": "mx", "Moldova": "md",
    "Mongolia": "mn", "Montenegro": "me", "Montserrat": "ms", "Morocco": "ma", "Mozambique": "mz",
    "Myanmar": "mm", "Namibia": "na", "Nepal": "np", "Netherlands": "nl", "New Caledonia": "nc",
    "New Zealand": "nz", "Nicaragua": "ni", "Niger": "ne", "Nigeria": "ng", "North Korea": "kp",
    "North Macedonia": "mk", "Northern Ireland": "gb-nir", "Norway": "no", "Oman": "om",
    "Pakistan": "pk", "Palestine": "ps", "Panama": "pa", "Papua New Guinea": "pg",
    "Paraguay": "py", "Peru": "pe", "Philippines": "ph", "Poland": "pl", "Portugal": "pt",
    "Puerto Rico": "pr", "Qatar": "qa", "Republic of Ireland": "ie", "Romania": "ro",
    "Russia": "ru", "Rwanda": "rw", "Samoa": "ws", "San Marino": "sm", "Saudi Arabia": "sa",
    "Scotland": "gb-sct", "Senegal": "sn", "Serbia": "rs", "Seychelles": "sc",
    "Sierra Leone": "sl", "Singapore": "sg", "Slovakia": "sk", "Slovenia": "si",
    "Solomon Islands": "sb", "Somalia": "so", "South Africa": "za", "South Korea": "kr",
    "South Sudan": "ss", "Spain": "es", "Sri Lanka": "lk", "Sudan": "sd", "Suriname": "sr",
    "Sweden": "se", "Switzerland": "ch", "Syria": "sy", "São Tomé and Príncipe": "st",
    "Tahiti": "pf", "Taiwan": "tw", "Tajikistan": "tj", "Tanzania": "tz", "Thailand": "th",
    "Togo": "tg", "Tonga": "to", "Trinidad and Tobago": "tt", "Tunisia": "tn", "Turkey": "tr",
    "Turkmenistan": "tm", "Turks and Caicos Islands": "tc", "Uganda": "ug", "Ukraine": "ua",
    "United Arab Emirates": "ae", "United States": "us", "Uruguay": "uy", "Uzbekistan": "uz",
    "Vanuatu": "vu", "Venezuela": "ve", "Vietnam": "vn", "Wales": "gb-wls", "Yemen": "ye",
    "Zambia": "zm", "Zimbabwe": "zw",
}

# Confederación de cada selección (para etiquetar el continente que representa).
CONFED = {
    # CONMEBOL (Sudamérica)
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Colombia": "CONMEBOL", "Peru": "CONMEBOL", "Chile": "CONMEBOL",
    "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
    # CONCACAF (Norte/Centroamérica y Caribe)
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "United States": "CONCACAF", "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF", "Panama": "CONCACAF", "Costa Rica": "CONCACAF", "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF", "El Salvador": "CONCACAF",
    # UEFA (Europa)
    "Germany": "UEFA", "Spain": "UEFA", "France": "UEFA", "England": "UEFA", "Portugal": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA", "Switzerland": "UEFA",
    "Sweden": "UEFA", "Norway": "UEFA", "Austria": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Czechia": "UEFA", "Bosnia and Herzegovina": "UEFA", "Italy": "UEFA", "Poland": "UEFA",
    "Denmark": "UEFA", "Serbia": "UEFA", "Wales": "UEFA", "Ukraine": "UEFA", "Greece": "UEFA",
    "Hungary": "UEFA", "Romania": "UEFA", "Republic of Ireland": "UEFA",
    # CAF (África)
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Tunisia": "CAF", "Algeria": "CAF",
    "Ghana": "CAF", "Côte d'Ivoire": "CAF", "Cape Verde": "CAF", "DR Congo": "CAF",
    "South Africa": "CAF", "Nigeria": "CAF", "Cameroon": "CAF", "Mali": "CAF",
    # AFC (Asia y Australia)
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC", "Australia": "AFC",
    "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC", "Uzbekistan": "AFC", "United Arab Emirates": "AFC",
    # OFC (Oceanía)
    "New Zealand": "OFC", "Fiji": "OFC", "Tahiti": "OFC",
}

# Países anfitriones del Mundial 2026: juegan de local (venue NO neutral).
ANFITRIONES = {"United States", "Mexico", "Canada"}


def flag_url(team, w=40):
    """URL de la bandera en flagcdn; None si no tenemos el código ISO."""
    iso = FLAG_ISO.get(team)                      # Código ISO2 de la selección
    return f"https://flagcdn.com/w{w}/{iso}.png" if iso else None

def flag_img(team, h=14):
    """Etiqueta <img> con la bandera lista para incrustar en HTML."""
    u = flag_url(team)                            # Buscar la URL de la bandera
    if not u:
        return ""                                 # Sin bandera conocida: no romper el layout
    return (f"<img src='{u}' alt='' style='height:{h}px;width:auto;border-radius:2px;"
            f"vertical-align:middle;margin-right:7px;box-shadow:0 0 0 1px rgba(0,0,0,.3)'>")


# ── Carga de artefactos del notebook (cacheada para no recargar en cada clic) ─
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("outputs/models/mlp_model_best.keras")   # Red neuronal entrenada
    with open("outputs/scalers/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)                                   # Normalizador del entrenamiento
    with open("outputs/metadata/feature_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)                                       # Orden de features + clases
    return modelo, scaler, meta

@st.cache_data
def cargar_datos():
    # Matriz plana de partidos con todas las features ya calculadas por el pipeline
    df = pd.read_csv("outputs/tables/model_df_features.csv", parse_dates=["date"])
    return df

modelo, scaler, meta = cargar_modelo()
FEATURE_COLS = meta["feature_cols"]          # 28 columnas en el orden que la red espera
df = cargar_datos()

# ── Perfil por selección: extraemos su registro más reciente ─────────────────
@st.cache_data
def construir_perfiles(_df):
    perfiles = {}
    for _, fila in _df.sort_values("date").iterrows():
        for lado in ("home", "away"):
            equipo = fila[f"{lado}_team"]
            perfiles[equipo] = {
                "rank":            fila[f"rank_{lado}"],
                "goals_scored":    fila[f"{lado}_avg_goals_scored_last10"],
                "goals_conceded":  fila[f"{lado}_avg_goals_conceded_last10"],
                "win":             fila[f"{lado}_avg_win_last10"],
                "draw":            fila[f"{lado}_avg_draw_last10"],
                "loss":            fila[f"{lado}_avg_loss_last10"],
                "overall":         fila[f"{lado}_attr_overall"],
                "pace":            fila[f"{lado}_attr_pace"],
                "shooting":        fila[f"{lado}_attr_shooting"],
                "defending":       fila[f"{lado}_attr_defending"],
                "physic":          fila[f"{lado}_attr_physic"],
            }
    return perfiles

PERFILES = construir_perfiles(df)
EQUIPOS_DISPONIBLES = sorted(PERFILES.keys())

# ── Historial de enfrentamientos directos (H2H) real desde el histórico ──────
# El modelo se entrenó con features H2H; recuperamos los totales reales de cada
# emparejamiento para no alimentarlo con ceros. Se cachea (se calcula una vez).
@st.cache_data
def construir_h2h(_df):
    # Clave canónica (equipo alfabéticamente menor, mayor) →
    # [victorias_del_menor, victorias_del_mayor, empates, dif_goles_del_menor, n_partidos]
    tab = {}
    for r in _df[["home_team", "away_team", "home_score", "away_score"]].itertuples(index=False):
        h, a, hs, as_ = r.home_team, r.away_team, r.home_score, r.away_score
        if pd.isna(hs) or pd.isna(as_):
            continue                                       # Saltar partidos sin marcador
        first, second = (h, a) if h <= a else (a, h)       # Orden canónico del par
        rec = tab.setdefault((first, second), [0, 0, 0, 0.0, 0])
        gd = (hs - as_) if h == first else (as_ - hs)      # Dif. de goles desde la vista del "menor"
        rec[3] += gd
        rec[4] += 1
        if hs == as_:
            rec[2] += 1                                    # Empate
        elif (hs > as_) == (h == first):
            rec[0] += 1                                    # Ganó el equipo "menor"
        else:
            rec[1] += 1                                    # Ganó el equipo "mayor"
    return tab

H2H_TABLE = construir_h2h(df)

def h2h_feats(equipo_a, equipo_b):
    """Features H2H reales desde la perspectiva de 'equipo_a' como local."""
    first, second = (equipo_a, equipo_b) if equipo_a <= equipo_b else (equipo_b, equipo_a)
    rec = H2H_TABLE.get((first, second))
    if not rec:
        return {"h2h_wins_home": 0, "h2h_draws": 0, "h2h_wins_away": 0, "h2h_goal_diff": 0, "h2h_n": 0}
    fw, sw, dr, gdf, n = rec
    if equipo_a == first:                                  # a es el "menor": sus victorias son fw
        return {"h2h_wins_home": fw, "h2h_draws": dr, "h2h_wins_away": sw, "h2h_goal_diff": gdf, "h2h_n": n}
    return {"h2h_wins_home": sw, "h2h_draws": dr, "h2h_wins_away": fw, "h2h_goal_diff": -gdf, "h2h_n": n}

# ── Reconstrucción del vector de 28 features para un partido A vs B ──────────
def construir_features(equipo_a, equipo_b, neutral=1):
    """Devuelve un DataFrame de 1 fila con las 28 columnas en el orden correcto."""
    pa, pb = PERFILES.get(equipo_a), PERFILES.get(equipo_b)
    if pa is None or pb is None:
        return None
    h2h = h2h_feats(equipo_a, equipo_b)                    # Historial directo real
    fila = {
        "rank_home": pa["rank"], "rank_away": pb["rank"],
        "home_avg_goals_scored_last10": pa["goals_scored"],
        "home_avg_goals_conceded_last10": pa["goals_conceded"],
        "home_avg_win_last10": pa["win"],
        "home_avg_draw_last10": pa["draw"],
        "home_avg_loss_last10": pa["loss"],
        "away_avg_goals_scored_last10": pb["goals_scored"],
        "away_avg_goals_conceded_last10": pb["goals_conceded"],
        "away_avg_win_last10": pb["win"],
        "away_avg_draw_last10": pb["draw"],
        "away_avg_loss_last10": pb["loss"],
        "h2h_wins_home": h2h["h2h_wins_home"], "h2h_draws": h2h["h2h_draws"],
        "h2h_wins_away": h2h["h2h_wins_away"], "h2h_goal_diff": h2h["h2h_goal_diff"],
        "h2h_n": h2h["h2h_n"],
        "home_attr_overall": pa["overall"], "home_attr_pace": pa["pace"],
        "home_attr_shooting": pa["shooting"], "home_attr_defending": pa["defending"],
        "home_attr_physic": pa["physic"],
        "away_attr_overall": pb["overall"], "away_attr_pace": pb["pace"],
        "away_attr_shooting": pb["shooting"], "away_attr_defending": pb["defending"],
        "away_attr_physic": pb["physic"],
        "neutral": neutral,   # 1 = sede neutral; 0 cuando el local es anfitrión
    }
    return pd.DataFrame([fila])[FEATURE_COLS]   # Reordenamos exactamente como la red espera

# ── Predicción de un partido usando el MODELO REAL ───────────────────────────
@st.cache_data
def predecir(equipo_a, equipo_b, neutral=1):
    X = construir_features(equipo_a, equipo_b, neutral)
    if X is None:
        return None
    X_s = scaler.transform(X.values.astype(np.float32))   # Mismo scaler del entrenamiento
    probs, gh, ga = modelo.predict(X_s, verbose=0)         # 3 cabezas: clases + goles
    p = probs[0]
    return {
        "pA": float(p[0]), "pDraw": float(p[1]), "pB": float(p[2]),
        "goalsA": float(gh[0][0]), "goalsB": float(ga[0][0]),
    }

def pred_local_neutral(equipo_a, equipo_b):
    """Predicción para un partido de torneo teniendo en cuenta la localía real.

    Si uno de los dos es país anfitrión (EE. UU., México o Canadá), juega de
    LOCAL (neutral=0). Si ninguno lo es, la sede es neutral. Devuelve las
    probabilidades ya alineadas al orden (equipo_a, equipo_b)."""
    host_a = equipo_a in ANFITRIONES
    host_b = equipo_b in ANFITRIONES
    if host_b and not host_a:
        # El anfitrión es 'b': lo tratamos como local y luego reordenamos la salida
        m = predecir(equipo_b, equipo_a, neutral=0)
        if m is None:
            return None
        return {"pA": m["pB"], "pDraw": m["pDraw"], "pB": m["pA"],
                "goalsA": m["goalsB"], "goalsB": m["goalsA"]}
    neutral = 0 if host_a else 1                            # 'a' local si es anfitrión
    return predecir(equipo_a, equipo_b, neutral=neutral)

# ── Grupos oficiales del sorteo. Nombres EXACTOS del CSV (en inglés) para que
#    PERFILES los encuentre; con nombres en español las proyecciones fallan. ──
GRUPOS_OFICIALES = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
# Veces campeón del mundo (contexto histórico para la vista de campeones)
TITULOS = {"Brazil": 5, "Germany": 4, "Argentina": 3, "France": 2,
           "Uruguay": 2, "Spain": 1, "England": 1}

# Etiquetas en español SOLO para mostrar; el modelo sigue usando el nombre del
# CSV (inglés). Las selecciones no listadas se muestran tal cual.
NOMBRES_ES = {
    "Mexico": "México", "South Africa": "Sudáfrica", "South Korea": "Corea del Sur",
    "Czechia": "Chequia", "Canada": "Canadá", "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Qatar": "Qatar", "Switzerland": "Suiza", "Brazil": "Brasil", "Morocco": "Marruecos",
    "Haiti": "Haití", "Scotland": "Escocia", "United States": "Estados Unidos",
    "Paraguay": "Paraguay", "Australia": "Australia", "Turkey": "Turquía", "Germany": "Alemania",
    "Curaçao": "Curazao", "Côte d'Ivoire": "Costa de Marfil", "Ecuador": "Ecuador",
    "Netherlands": "Países Bajos", "Japan": "Japón", "Sweden": "Suecia", "Tunisia": "Túnez",
    "Belgium": "Bélgica", "Egypt": "Egipto", "Iran": "Irán", "New Zealand": "Nueva Zelanda",
    "Spain": "España", "Cape Verde": "Cabo Verde", "Saudi Arabia": "Arabia Saudita",
    "Uruguay": "Uruguay", "France": "Francia", "Senegal": "Senegal", "Iraq": "Iraq",
    "Norway": "Noruega", "Argentina": "Argentina", "Algeria": "Argelia", "Austria": "Austria",
    "Jordan": "Jordania", "Portugal": "Portugal", "DR Congo": "RD Congo",
    "Uzbekistan": "Uzbekistán", "Colombia": "Colombia", "England": "Inglaterra",
    "Croatia": "Croacia", "Ghana": "Ghana", "Panama": "Panamá",
    # Selecciones frecuentes fuera de los grupos (por si el usuario las trae)
    "Peru": "Perú", "Chile": "Chile", "Bolivia": "Bolivia", "Venezuela": "Venezuela",
    "Italy": "Italia", "Poland": "Polonia", "Denmark": "Dinamarca", "Serbia": "Serbia",
    "Wales": "Gales", "Ukraine": "Ucrania", "Greece": "Grecia", "Nigeria": "Nigeria",
    "Cameroon": "Camerún", "United Arab Emirates": "Emiratos Árabes Unidos",
    "Costa Rica": "Costa Rica", "Honduras": "Honduras", "Jamaica": "Jamaica",
    "Republic of Ireland": "Irlanda", "Hungary": "Hungría", "Romania": "Rumanía",
}
def es(nombre):
    """Etiqueta en español para mostrar; si no está mapeada, se deja en inglés."""
    return NOMBRES_ES.get(nombre, nombre)


# ── Proyección de un grupo (round-robin) con desempates y % Monte Carlo ───────
@st.cache_data(show_spinner=False)
def proyectar_grupo(equipos):
    equipos = list(equipos)
    n = len(equipos)
    n_matches_per_team = n - 1                    # En un grupo de 4, cada uno juega 3 partidos
    filas = [{"Selección": e, "pts": 0.0, "gf": 0.0, "ga": 0.0, "pj": float(n_matches_per_team)}
             for e in equipos]

    # Guardamos las probabilidades de cada cruce para la simulación Monte Carlo
    partidos = []
    for i in range(n):
        for j in range(i + 1, n):
            m = pred_local_neutral(equipos[i], equipos[j])
            if m is None:
                partidos.append((i, j, 0.4, 0.25, 0.35, 1.2, 1.0))   # Respaldo si falta un perfil
                continue
            filas[i]["pts"] += 3 * m["pA"] + m["pDraw"]
            filas[j]["pts"] += 3 * m["pB"] + m["pDraw"]
            filas[i]["gf"] += m["goalsA"]; filas[i]["ga"] += m["goalsB"]
            filas[j]["gf"] += m["goalsB"]; filas[j]["ga"] += m["goalsA"]
            partidos.append((i, j, m["pA"], m["pDraw"], m["pB"], m["goalsA"], m["goalsB"]))

    # Valores esperados (crudos para ordenar; redondeados para mostrar)
    for f in filas:
        f["gd"] = f["gf"] - f["ga"]
        f["pts_raw"], f["gd_raw"], f["gf_raw"] = f["pts"], f["gd"], f["gf"]
        f["pts"] = round(f["pts"]); f["gf"] = round(f["gf"])
        f["ga"] = round(f["ga"]); f["gd"] = round(f["gd"])

    # ── Probabilidad de clasificar (top-2) por simulación Monte Carlo ─────────
    # Simulamos el grupo N veces muestreando cada partido de las probabilidades
    # del modelo; contamos cuántas veces cada selección termina entre las 2 primeras.
    N = 3000
    rng = np.random.default_rng(20260101)          # Semilla fija: % estable entre recargas
    pts_sim = np.zeros((N, n)); gd_sim = np.zeros((N, n))
    for (i, j, pa, pdr, pb, ga_, gb_) in partidos:
        r = rng.random(N)                          # Un número aleatorio por simulación
        a_win = r < pa                             # Gana i
        draw = (r >= pa) & (r < pa + pdr)          # Empate
        b_win = r >= pa + pdr                      # Gana j
        pts_sim[:, i] += np.where(a_win, 3, np.where(draw, 1, 0))
        pts_sim[:, j] += np.where(b_win, 3, np.where(draw, 1, 0))
        gd_sim[:, i] += (ga_ - gb_); gd_sim[:, j] += (gb_ - ga_)   # Dif. de goles (desempate)
    key = pts_sim + gd_sim * 1e-3                   # Ordenar por puntos y, si empatan, por dif. de goles
    top2 = np.argsort(-key, axis=1)[:, :2]          # Índices de los 2 primeros en cada simulación
    clasif = np.array([(top2 == t).sum() for t in range(n)]) / N
    for t, f in enumerate(filas):
        f["Clasif."] = float(clasif[t])

    # Orden final de la tabla: puntos → diferencia de goles → goles a favor
    filas.sort(key=lambda x: (x["pts_raw"], x["gd_raw"], x["gf_raw"]), reverse=True)
    return filas

# ── Clasificados: 2 primeros de cada grupo + 8 mejores terceros (32) ─────────
def clasificados_desde(grupos):
    primeros, segundos, terceros = [], [], []
    for equipos in grupos.values():
        tabla = proyectar_grupo(tuple(equipos))
        primeros.append(tabla[0]["Selección"])
        segundos.append(tabla[1]["Selección"])
        terceros.append({"eq": tabla[2]["Selección"], "pts": tabla[2]["pts_raw"], "gd": tabla[2]["gd_raw"]})
    # Los 8 mejores terceros por puntos y luego por diferencia de goles
    terceros = [t["eq"] for t in sorted(terceros, key=lambda x: (x["pts"], x["gd"]), reverse=True)[:8]]
    clas = primeros + segundos + terceros
    # Sembrado: más fuerte (menor ranking FIFA) primero, para cruzar fuertes vs débiles
    clas.sort(key=lambda e: PERFILES.get(e, {}).get("rank", 999))
    return clas

# Constants for knockout rounds
RONDAS_NOMBRES = ["16vos", "Octavos", "Cuartos", "Semifinal", "Final"]
RONDAS_MAP = {name: i for i, name in enumerate(RONDAS_NOMBRES)}

# ── Simulación del cuadro por favorito (con opción de forzar un avance) ───────
def construir_bracket(clasificados, forzados_team=None, forzados_round_idx=-1):
    ronda = [(clasificados[i], clasificados[31 - i]) for i in range(16)]
    rondas_data, ri = [], 0
    while len(ronda) >= 1:
        partidos, ganadores = [], []
        for a, b in ronda:
            m = pred_local_neutral(a, b)
            if m is None:
                gan = a  # Default to team A if no prediction data
                pa, pb, pdr, gA, gB = 0.5, 0.5, 0.0, 0.0, 0.0
            else:
                # ¿Hay un ganador forzado en esta ronda?
                if forzados_team and ri <= forzados_round_idx:
                    if forzados_team == a:
                        gan = a
                    elif forzados_team == b:
                        gan = b
                    else:
                        gan = (a if m["pA"] >= m["pB"] else b)     # Equipo forzado no está en este cruce
                else:
                    gan = (a if m["pA"] >= m["pB"] else b)         # Sin forzar: gana el favorito
                pa, pb, pdr = m["pA"], m["pB"], m["pDraw"]
                gA, gB = m["goalsA"], m["goalsB"]

            partidos.append({"a": a, "b": b, "pa": pa, "pb": pb, "pd": pdr,
                             "gan": gan, "gA": gA, "gB": gB})
            ganadores.append(gan)

        rondas_data.append({"nombre": RONDAS_NOMBRES[ri] if ri < len(RONDAS_NOMBRES) else f"Ronda {ri+1}",
                       "partidos": partidos})
        if len(ronda) == 1:
            break
        ronda = [(ganadores[i], ganadores[i + 1]) for i in range(0, len(ganadores), 2)]
        ri += 1
    return rondas_data

# ── Propagación probabilística por todo el cuadro → top campeones ────────────
def campeones_probables(clasificados):
    dists = []
    for i in range(16):
        a, b = clasificados[i], clasificados[31 - i]
        m = pred_local_neutral(a, b)
        if m:
            dists.append({a: m["pA"] + m["pDraw"] / 2, b: m["pB"] + m["pDraw"] / 2})
        else:
            dists.append({a: 0.5, b: 0.5})
    while len(dists) > 1:
        siguiente = []
        for i in range(0, len(dists), 2):
            dA, dB = dists[i], dists[i + 1]
            fusion = {}
            for na, pa in dA.items():
                for nb, pb in dB.items():
                    m = pred_local_neutral(na, nb)
                    if not m:
                        continue
                    fusion[na] = fusion.get(na, 0) + pa * pb * (m["pA"] + m["pDraw"] / 2)
                    fusion[nb] = fusion.get(nb, 0) + pa * pb * (m["pB"] + m["pDraw"] / 2)
            siguiente.append(fusion)
        dists = siguiente
    return sorted(dists[0].items(), key=lambda x: x[1], reverse=True)[:10]


# ============================================================================
#  RENDER — generadores de HTML para las piezas custom del diseño
# ============================================================================
def _match_card(p, forced_team_name=None):
    """Tarjeta HTML de un cruce: fila A, franja de empate, fila B y marcador esperado."""
    def fila(nombre, prob, gana):
        col = LIME if gana else MUTED
        estrella = (f"<span style='color:{AMBER};font-size:10px;margin-left:5px'>★</span>"
                    if nombre == forced_team_name else "")
        return (
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:7px 10px;"
            f"background:{PANELHI if gana else 'transparent'};border-left:3px solid "
            f"{AMBER if nombre == forced_team_name else (LIME if gana else 'transparent')}'>"
            f"<span style='font-size:12px;font-weight:{700 if gana else 500};display:flex;align-items:center;"
            f"color:{INK if gana else MUTED}'>{flag_img(nombre, 12)}{es(nombre)}{estrella}</span>"
            f"<span style='font-size:11px;font-weight:700;font-family:Archivo,sans-serif;color:{col}'>"
            f"{round(prob * 100)}%</span></div>"
        )
    # Marcador esperado del cruce (goles A – goles B redondeados)
    score = f"{es(p['a'])[:3].upper()} {round(p['gA'])} – {round(p['gB'])} {es(p['b'])[:3].upper()}"
    return (
        f"<div class='wc-match'>{fila(p['a'], p['pa'], p['gan'] == p['a'])}"
        f"<div style='display:flex;justify-content:center;align-items:center;height:16px;"
        f"background:{PANEL};border-top:1px solid {LINE};border-bottom:1px solid {LINE}'>"
        f"<span style='font-size:9px;letter-spacing:1px;text-transform:uppercase;color:{MUTED}'>"
        f"empate {round(p['pd'] * 100)}%</span></div>"
        f"{fila(p['b'], p['pb'], p['gan'] == p['b'])}"
        f"<div class='wc-scoreline'>≈ {score}</div></div>"
    )

def _half(partidos, side):
    """Mitad superior (izquierda) o inferior (derecha) de los cruces de una ronda."""
    h = len(partidos) // 2
    return partidos[:h] if side == "L" else partidos[h:]

def _round_col(nombre, partidos, side, forced_team_name):
    cards = "".join(_match_card(p, forced_team_name) for p in partidos)
    cls = "wc-round wc-round--right" if side == "R" else "wc-round"
    return (f"<div class='{cls}'><div class='wc-round-h'>{nombre}</div>"
            f"<div class='wc-round-body'>{cards}</div></div>")

def html_bracket(rondas, forced_team_name=None):
    """Bracket de dos lados: mitad izquierda →, mitad derecha ←, Final al centro."""
    intermedias = rondas[:-1]                       # Todas las rondas menos la Final
    final = rondas[-1]["partidos"][0]               # Único cruce de la Final
    # Mitad izquierda: rondas en orden 16vos → Semifinal, con sus cruces superiores
    izq = "".join(_round_col(r["nombre"], _half(r["partidos"], "L"), "L", forced_team_name)
                  for r in intermedias)
    # Mitad derecha: rondas en orden inverso (16vos al extremo), con sus cruces inferiores
    der = "".join(_round_col(r["nombre"], _half(r["partidos"], "R"), "R", forced_team_name)
                  for r in reversed(intermedias))
    # Centro: trofeo + cruce de la Final
    centro = (f"<div class='wc-center'><div class='wc-trophy'>🏆</div>"
              f"<div class='wc-round-h'>Final</div>{_match_card(final, forced_team_name)}</div>")
    return f"<div class='wc-bracket2 wc-rise'>{izq}{centro}{der}</div>"

def html_campeones(top):
    maxp = top[0][1] if top else 1
    filas = []
    for pos, (eq, p) in enumerate(top):
        n_tit = TITULOS.get(eq, 0)
        tit = (f"<span style='color:{AMBER};font-size:11px;margin-left:8px;font-weight:600'>"
               f"{'★' * n_tit} {n_tit} {'título' if n_tit == 1 else 'títulos'}</span>") if n_tit else ""
        barra = LIME if pos == 0 else f"linear-gradient(90deg,{LOSS},{LIME})"
        filas.append(
            f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:11px'>"
            f"<span style='font-family:Archivo,sans-serif;font-weight:900;font-size:18px;"
            f"color:{LIME if pos == 0 else MUTED};width:26px;text-align:right'>{pos + 1}</span>"
            f"<div style='flex:1'>"
            f"<div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px'>"
            f"<span style='font-weight:700;font-size:15px;color:{INK};display:flex;align-items:center'>"
            f"{flag_img(eq, 15)}{es(eq)}{tit}</span>"
            f"<span style='font-family:Archivo,sans-serif;font-weight:800;font-size:16px;color:{LIME}'>"
            f"{p * 100:.1f}%</span></div>"
            f"<div style='background:{PANEL};border:1px solid {LINE};border-radius:4px;height:12px;overflow:hidden'>"
            f"<div style='width:{max(2, p / maxp * 100):.1f}%;height:100%;background:{barra};border-radius:3px;"
            f"animation:barGrow .6s ease'></div></div></div></div>")
    return f"<div class='wc-rise' style='max-width:800px'>{''.join(filas)}</div>"

def html_scout(equipo):
    """Ficha de scouting de una selección: ranking, atributos FIFA y forma reciente."""
    pf = PERFILES.get(equipo, {})
    conf = CONFED.get(equipo, "")
    host = " · Anfitrión 🏟️" if equipo in ANFITRIONES else ""
    def stat(lbl, val):
        return f"<div class='wc-stat'><span>{lbl}</span><span>{val}</span></div>"
    # Pre-calculamos cada valor para no anidar comillas dentro del f-string
    v_rank = f"#{int(pf.get('rank', 0))}"
    v_ovr = f"{pf.get('overall', 0):.1f}"
    v_atk = f"{pf.get('shooting', 0):.0f} / {pf.get('pace', 0):.0f}"
    v_def = f"{pf.get('defending', 0):.0f} / {pf.get('physic', 0):.0f}"
    v_gf = f"{pf.get('goals_scored', 0):.2f}"
    v_gc = f"{pf.get('goals_conceded', 0):.2f}"
    forma = (f"{pf.get('win', 0) * 100:.0f}% V · {pf.get('draw', 0) * 100:.0f}% E · "
             f"{pf.get('loss', 0) * 100:.0f}% D")
    return (
        f"<div class='wc-scout'>"
        f"<div class='nm'>{flag_img(equipo, 22)}{es(equipo)}</div>"
        f"<div class='cf'>{conf}{host}</div>"
        f"{stat('Ranking FIFA', v_rank)}"
        f"{stat('Media general (overall)', v_ovr)}"
        f"{stat('Ataque (shooting / pace)', v_atk)}"
        f"{stat('Defensa / físico', v_def)}"
        f"{stat('Goles anotados (prom. últ. 10)', v_gf)}"
        f"{stat('Goles recibidos (prom. últ. 10)', v_gc)}"
        f"{stat('Forma reciente', forma)}"
        f"</div>"
    )


# ============================================================================
#  ESTADO Y CALLBACKS
#  Fuente única de verdad = las claves sel_<grupo>_<slot> del session_state.
#  El intercambio (swap) y el desplegable escriben ahí; el resto lo lee.
# ============================================================================
for _g, _eqs in GRUPOS_OFICIALES.items():
    for _s, _e in enumerate(_eqs):
        st.session_state.setdefault(f"sel_{_g}_{_s}", _e)
        st.session_state.setdefault(f"shadow_{_g}_{_s}", _e)   # valor previo, para reubicar
st.session_state.setdefault("swap_sel", None)   # (grupo, slot) marcado para intercambiar
st.session_state.setdefault("forzados_team", None)
st.session_state.setdefault("forzados_round_idx", -1) # -1 means no round forced
st.session_state.setdefault("_aviso", None)     # mensaje pendiente para el toast

def grupos_actuales():
    return {g: [st.session_state[f"sel_{g}_{s}"] for s in range(4)] for g in GRUPOS_OFICIALES}

def restaurar_sorteo():
    for g, eqs in GRUPOS_OFICIALES.items():
        for s, e in enumerate(eqs):
            st.session_state[f"sel_{g}_{s}"] = e
    st.session_state.swap_sel = None
    st.session_state.forzados_team = None
    st.session_state.forzados_round_idx = -1

def tap_swap(g, slot):
    sel = st.session_state.swap_sel
    if sel is None:
        st.session_state.swap_sel = (g, slot)
    elif sel == (g, slot):
        st.session_state.swap_sel = None          # deseleccionar
    else:
        k0, k1 = f"sel_{sel[0]}_{sel[1]}", f"sel_{g}_{slot}"
        st.session_state[k0], st.session_state[k1] = st.session_state[k1], st.session_state[k0]
        st.session_state.swap_sel = None

def reubicar(g, slot):
    """Al elegir una selección en el desplegable: si ya estaba en otro puesto,
    intercambian posiciones (la desplazada va al puesto original de la elegida)."""
    nuevo = st.session_state[f"sel_{g}_{slot}"]
    viejo = st.session_state[f"shadow_{g}_{slot}"]
    if nuevo == viejo:
        return
    for g2 in GRUPOS_OFICIALES:
        for s2 in range(4):
            if (g2, s2) != (g, slot) and st.session_state[f"sel_{g2}_{s2}"] == nuevo:
                st.session_state[f"sel_{g2}_{s2}"] = viejo   # intercambio de posiciones
                st.session_state["_aviso"] = (
                    f"Selección repetida: {es(nuevo)} ya estaba en el Grupo {g2}. "
                    f"Se intercambió con {es(viejo)}.")
                return

def forzar_equipo():
    team = st.session_state.get("ff_team")
    round_name = st.session_state.get("ff_round")

    if team and round_name:
        st.session_state.forzados_team = team
        st.session_state.forzados_round_idx = RONDAS_MAP.get(round_name, -1)
    else:
        st.session_state.forzados_team = None
        st.session_state.forzados_round_idx = -1

def limpiar_forzados():
    st.session_state.forzados_team = None
    st.session_state.forzados_round_idx = -1


# ============================================================================
#  INTERFAZ
# ============================================================================
# Banderas de los tres países anfitriones para la cabecera
_host_flags = "".join(flag_img(h, 13) for h in ["United States", "Canada", "Mexico"])
st.markdown(
    "<div class='wc-header'>"
    "<div class='wc-title'>WORLD CUP <span style='color:#c8f04a'>2026</span></div>"
    "<div class='wc-sub'>Consola de Inteligencia Deportiva</div>"
    "<div class='wc-meta'>Simulador predictivo con el modelo MLP entrenado · "
    f"{_host_flags} EE. UU. · Canadá · México · 48 selecciones</div>"
    "</div>", unsafe_allow_html=True)

nav, reset = st.columns([4, 1])
with nav:
    vista = st.segmented_control(
        "Vista", ["Predecir partido", "Fase de grupos", "Eliminación directa", "Campeones probables"],
        default="Fase de grupos", key="vista", label_visibility="collapsed")
with reset:
    st.button("↺ Restaurar sorteo", on_click=restaurar_sorteo, width="stretch")

grupos = grupos_actuales()

# Popup de advertencia si un cambio provocó (y resolvió) una selección repetida
if st.session_state["_aviso"]:
    st.toast(st.session_state["_aviso"], icon="⚠️")
    st.session_state["_aviso"] = None

# ── VISTA 0: PREDECIR UN PARTIDO (Componente 1 expuesto directamente) ─────────
if vista == "Predecir partido":
    st.markdown(
        "<p class='wc-lead'>Elige <strong>dos selecciones</strong> y el modelo predice las "
        "probabilidades de victoria, empate y derrota, además del marcador esperado. Es el "
        "<strong>modelo predictivo</strong> del sistema, accesible de forma directa.</p>",
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        eq_a = st.selectbox("Selección A (local)", EQUIPOS_DISPONIBLES,
                            index=EQUIPOS_DISPONIBLES.index("Argentina"), format_func=es)
    with c2:
        eq_b = st.selectbox("Selección B (visitante)", EQUIPOS_DISPONIBLES,
                            index=EQUIPOS_DISPONIBLES.index("France"), format_func=es)
    with c3:
        sede_neutral = st.toggle("Sede neutral", value=True,
                                 help="Desactívalo para dar localía a la Selección A")

    if eq_a == eq_b:
        st.warning("Elige dos selecciones distintas.")
    else:
        m = predecir(eq_a, eq_b, neutral=1 if sede_neutral else 0)
        if m is None:
            st.error("No hay datos suficientes para una de las selecciones.")
        else:
            # Barra tri-color: victoria A · empate · victoria B
            pa, pd_, pb = m["pA"], m["pDraw"], m["pB"]
            tribar = (
                f"<div class='wc-tribar'>"
                f"<div style='width:{pa*100:.1f}%;background:{LIME}'>{pa*100:.0f}%</div>"
                f"<div style='width:{pd_*100:.1f}%;background:{MUTED}'>{pd_*100:.0f}%</div>"
                f"<div style='width:{pb*100:.1f}%;background:{LOSS};color:{INK}'>{pb*100:.0f}%</div>"
                f"</div>"
                f"<div style='display:flex;justify-content:space-between;font-size:12px;color:{MUTED};margin-bottom:14px'>"
                f"<span>Gana {es(eq_a)}</span><span>Empate</span><span>Gana {es(eq_b)}</span></div>"
                f"<div style='text-align:center;font-family:Archivo,sans-serif;font-weight:900;font-size:30px;"
                f"color:{INK};margin:6px 0 20px'>{flag_img(eq_a,24)} {round(m['goalsA'])} "
                f"<span style='color:{MUTED};font-size:20px'>–</span> {round(m['goalsB'])} {flag_img(eq_b,24)}"
                f"<div style='font-size:11px;color:{MUTED};letter-spacing:2px;margin-top:4px'>MARCADOR ESPERADO</div></div>"
            )
            st.markdown(tribar, unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(html_scout(eq_a), unsafe_allow_html=True)
            with sc2:
                st.markdown(html_scout(eq_b), unsafe_allow_html=True)

# ── VISTA 1: FASE DE GRUPOS ──────────────────────────────────────────────────
elif vista == "Fase de grupos" or vista is None:
    st.markdown(
        "<p class='wc-lead'>Los 12 grupos del sorteo oficial. Toca una selección y luego otra "
        "para <strong>intercambiarlas</strong> (incluso entre grupos distintos); usa "
        "<strong>✎ Editar</strong> para traer cualquiera de las 206 selecciones. Las tablas y "
        "los porcentajes de clasificación (simulados 3000 veces) se recalculan al instante.</p>",
        unsafe_allow_html=True)

    with st.spinner("Recalculando los 12 grupos con el modelo…"):
        proyecciones = {g: proyectar_grupo(tuple(grupos[g])) for g in GRUPOS_OFICIALES}

    # CSS dirigido: bandera (::before) + borde de cal en los 2 primeros de cada grupo
    reglas = []
    for g, tabla in proyecciones.items():
        for pos, fila in enumerate(tabla):
            nombre = fila["Selección"]
            slot = grupos[g].index(nombre)
            u = flag_url(nombre)
            if u:
                reglas.append(
                    f".st-key-swap_{g}_{slot} button::before{{content:'';display:inline-block;"
                    f"width:21px;height:14px;margin-right:9px;background:url('{u}') center/cover;"
                    f"border-radius:2px;vertical-align:-2px;box-shadow:0 0 0 1px rgba(0,0,0,.35)}}")
            if pos < 2:   # Los 2 clasificados: borde de cal
                reglas.append(f".st-key-swap_{g}_{slot} button{{border-left-color:#c8f04a}}")
    st.markdown("<style>" + "".join(reglas) + "</style>", unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    for idx, g in enumerate(GRUPOS_OFICIALES):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"<div class='wc-badge'><b>{g}</b><span>Grupo {g}</span></div>"
                            "<div class='wc-cols'><span>Selección · PJ · GF · GC · DG · Pts</span><span>Clasif.</span></div>",
                            unsafe_allow_html=True)
                for pos, fila in enumerate(proyecciones[g]):
                    nombre = fila["Selección"]
                    slot = grupos[g].index(nombre)
                    marcado = st.session_state.swap_sel == (g, slot)
                    # Etiqueta: puesto · nombre · PJ · GF · GC · DG · Pts · % clasificación
                    etiqueta = (f"{pos + 1}.  {es(nombre)}   ·   {int(fila['pj'])}   ·   {int(fila['gf'])}   ·   "
                                f"{int(fila['ga'])}   ·   {int(fila['gd'])}   ·   {int(fila['pts'])} pts"
                                f"      {round(fila['Clasif.'] * 100)}%")
                    st.button(etiqueta, key=f"swap_{g}_{slot}", on_click=tap_swap, args=(g, slot),
                              width="stretch", type="primary" if marcado else "secondary")
                with st.expander("✎ Editar selecciones"):
                    for s in range(4):
                        st.selectbox(f"Puesto {s + 1}", EQUIPOS_DISPONIBLES,
                                     key=f"sel_{g}_{s}", on_change=reubicar, args=(g, s),
                                     format_func=es, label_visibility="visible")

    # Botón de descarga de todas las tablas de grupos (para la entrega/defensa)
    filas_export = []
    for g, tabla in proyecciones.items():
        for pos, fila in enumerate(tabla):
            filas_export.append({"Grupo": g, "Pos": pos + 1, "Seleccion": fila["Selección"],
                                 "PJ": int(fila["pj"]), "GF": int(fila["gf"]), "GC": int(fila["ga"]),
                                 "DG": int(fila["gd"]), "Pts": int(fila["pts"]),
                                 "Clasif_%": round(fila["Clasif."] * 100, 1)})
    csv_grupos = pd.DataFrame(filas_export).to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar tablas de grupos (CSV)", csv_grupos,
                       file_name="fase_de_grupos_2026.csv", mime="text/csv")

    # Guardamos el estado actual como "previo" para el próximo cambio (reubicar)
    for g in GRUPOS_OFICIALES:
        for s in range(4):
            st.session_state[f"shadow_{g}_{s}"] = st.session_state[f"sel_{g}_{s}"]

# ── VISTA 2: ELIMINACIÓN DIRECTA ─────────────────────────────────────────────
elif vista == "Eliminación directa":
    st.markdown(
        "<p class='wc-lead'>Cuadro de eliminación simulado con el modelo, en formato de "
        "<strong>llave de dos lados</strong> que converge a la Final. Usa el panel para "
        "<strong>forzar el avance</strong> de una selección hasta una ronda y ver cómo se "
        "reconfigura el cuadro desde ahí.</p>", unsafe_allow_html=True)

    with st.spinner("Simulando el cuadro de eliminación…"):
        clasificados = clasificados_desde(grupos)
        rondas = construir_bracket(clasificados, st.session_state.forzados_team,
                                   st.session_state.forzados_round_idx)
    campeon = rondas[-1]["partidos"][0]["gan"]
    forced_team_name = st.session_state.forzados_team

    st.markdown(f"<div class='wc-champ'><span class='lbl'>Campeón proyectado</span>"
                f"{flag_img(campeon, 26)}<span class='name'>{es(campeon)}</span></div>",
                unsafe_allow_html=True)

    with st.expander("Forzar el avance de una selección"):
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1:
            st.selectbox("Selección que avanza", clasificados,
                         key="ff_team", format_func=es, label_visibility="collapsed")
        with c2:
            st.selectbox("Hasta la ronda", RONDAS_NOMBRES,
                         key="ff_round", label_visibility="collapsed")
        with c3:
            st.button("Forzar", on_click=forzar_equipo, width="stretch")
        with c4:
            st.button("Limpiar", on_click=limpiar_forzados, width="stretch")

    st.markdown(html_bracket(rondas, forced_team_name), unsafe_allow_html=True)
    st.caption("El favorito de cada cruce se decide por la mayor probabilidad de victoria del modelo "
               "(el empate se resolvería por penales). El “Campeón proyectado” sigue el camino de "
               "favoritos; la vista de Campeones probables lo calcula de forma probabilística, por lo "
               "que ambos pueden diferir.")

# ── VISTA 3: CAMPEONES PROBABLES ─────────────────────────────────────────────
else:
    st.markdown(
        "<p class='wc-lead'>Las 10 selecciones con mayor probabilidad de levantar el trofeo, "
        "propagando el modelo por todo el cuadro. Se recalcula al modificar los grupos. "
        "La marca ★ indica títulos mundiales previos.</p>", unsafe_allow_html=True)

    with st.spinner("Propagando el modelo por todo el cuadro…"):
        clasificados = clasificados_desde(grupos)
        top = campeones_probables(clasificados)
    st.markdown(html_campeones(top), unsafe_allow_html=True)

st.markdown(
    "<p class='wc-foot'>El dashboard carga el modelo real (mlp_model_best.keras) y reconstruye las "
    "28 features con el mismo pipeline del notebook (incluye historial H2H real y localía de los "
    "países anfitriones). Banderas: flagcdn.com.</p>", unsafe_allow_html=True)
