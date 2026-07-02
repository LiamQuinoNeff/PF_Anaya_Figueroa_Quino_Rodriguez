# -*- coding: utf-8 -*-
# ============================================================================
#  Dashboard interactivo (Componente 2) — FIFA World Cup 2026
#  Carga el modelo MLP entrenado en el notebook (mlp_model_best.keras, scaler.pkl,
#  feature_metadata.json), reconstruye las 28 features de cada partido desde
#  model_df_features.csv y predice con la MISMA red y normalizador del notebook.
#  Interfaz "Tactical Command Console": tema oscuro con HTML/CSS a medida.
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

/* Filas-selección: botones de intercambio con aspecto de fila de tabla */
div[class*="st-key-swap_"] button { width: 100%; justify-content: flex-start; text-align: left;
  background: transparent; border: 1px solid transparent; border-left: 3px solid transparent;
  border-radius: 6px; color: #eaf1fb; font-weight: 500; padding: 7px 10px;
  font-feature-settings: 'tnum'; transition: background .15s; }
div[class*="st-key-swap_"] button:hover { background: #16294a; }
div[class*="st-key-swap_"] button p { font-size: 13px; margin: 0; }
/* Slot seleccionado para intercambiar (botón primario) */
div[class*="st-key-swap_"] button[kind="primary"] { background: #c8f04a; color: #0a1628;
  border-left-color: #c8f04a; font-weight: 700; }

/* Bracket (árbol). Rondas flexibles: llenan el ancho en pantallas grandes,
   con scroll horizontal solo como respaldo en pantallas pequeñas. */
.wc-bracket { display: flex; gap: 18px; overflow-x: auto; align-items: stretch; padding: 4px 2px 14px; }
.wc-round { flex: 1 1 0; min-width: 168px; display: flex; flex-direction: column; }
.wc-round-h { font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 13px; letter-spacing: 1.5px;
  text-transform: uppercase; color: #f5a623; text-align: center; padding-bottom: 5px; margin-bottom: 10px;
  border-bottom: 1px solid #1e3459; }
.wc-round-body { display: flex; flex-direction: column; justify-content: space-around; flex: 1; gap: 10px; }
.wc-match { background: #0f1f38; border: 1px solid #1e3459; border-radius: 8px; overflow: hidden;
  flex: 0 0 auto; position: relative; }
/* Conector: pequeña línea que une cada cruce con la ronda siguiente */
.wc-round:not(:last-child) .wc-match::after { content: ""; position: absolute; top: 50%; left: 100%;
  width: 18px; height: 1px; background: #1e3459; }
.wc-champ { display: inline-flex; align-items: center; gap: 14px; margin: 4px 0 22px;
  background: #16294a; border: 1px solid #c8f04a; border-radius: 10px; padding: 12px 22px; }
.wc-champ .lbl { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #7d93b8; font-weight: 700; }
.wc-champ .name { font-family: 'Archivo', sans-serif; font-weight: 900; font-size: 26px; color: #c8f04a; }

/* Pie de transparencia */
.wc-foot { color: #7d93b8; font-size: 11px; line-height: 1.6; max-width: 900px;
  border-top: 1px solid #1e3459; padding-top: 14px; margin-top: 22px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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

# ── Reconstrucción del vector de 28 features para un partido A vs B ──────────
def construir_features(equipo_a, equipo_b, neutral=1):
    """Devuelve un DataFrame de 1 fila con las 28 columnas en el orden correcto."""
    pa, pb = PERFILES.get(equipo_a), PERFILES.get(equipo_b)
    if pa is None or pb is None:
        return None
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
        "h2h_wins_home": 0, "h2h_draws": 0, "h2h_wins_away": 0,
        "h2h_goal_diff": 0, "h2h_n": 0,
        "home_attr_overall": pa["overall"], "home_attr_pace": pa["pace"],
        "home_attr_shooting": pa["shooting"], "home_attr_defending": pa["defending"],
        "home_attr_physic": pa["physic"],
        "away_attr_overall": pb["overall"], "away_attr_pace": pb["pace"],
        "away_attr_shooting": pb["shooting"], "away_attr_defending": pb["defending"],
        "away_attr_physic": pb["physic"],
        "neutral": neutral,   # Los partidos del Mundial se juegan en sede neutral
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
}
def es(nombre):
    """Etiqueta en español para mostrar; si no está mapeada, se deja en inglés."""
    return NOMBRES_ES.get(nombre, nombre)


# ── Proyección de un grupo (round-robin). Acumula por índice de puesto para que
#    siempre devuelva una fila por selección (evita IndexError aguas abajo). ───
def proyectar_grupo(equipos):
    filas = [{"Selección": e, "pts": 0.0, "gf": 0.0, "ga": 0.0} for e in equipos]
    for i in range(len(equipos)):
        for j in range(i + 1, len(equipos)):
            m = predecir(equipos[i], equipos[j])
            if m is None:
                continue
            filas[i]["pts"] += 3 * m["pA"] + m["pDraw"]
            filas[j]["pts"] += 3 * m["pB"] + m["pDraw"]
            filas[i]["gf"] += m["goalsA"]; filas[i]["ga"] += m["goalsB"]
            filas[j]["gf"] += m["goalsB"]; filas[j]["ga"] += m["goalsA"]
    # Probabilidad de clasificar (top-2) vía softmax sobre puntos, escalada a 2 cupos
    pts = np.array([f["pts"] for f in filas])
    exps = np.exp(pts / 1.15); soft = exps / exps.sum()
    for f, s in zip(filas, soft):
        f["Clasif."] = min(0.99, s / soft.sum() * 2)
    filas.sort(key=lambda x: x["pts"], reverse=True)
    return filas

# ── Clasificados: 2 primeros de cada grupo + 8 mejores terceros (32) ─────────
def clasificados_desde(grupos):
    primeros, segundos, terceros = [], [], []
    for equipos in grupos.values():
        tabla = proyectar_grupo(equipos)
        primeros.append(tabla[0]["Selección"])
        segundos.append(tabla[1]["Selección"])
        terceros.append({"eq": tabla[2]["Selección"], "pts": tabla[2]["pts"]})
    terceros = [t["eq"] for t in sorted(terceros, key=lambda x: x["pts"], reverse=True)[:8]]
    clas = primeros + segundos + terceros
    # Sembrado: más fuerte (menor ranking FIFA) primero, para cruzar fuertes vs débiles
    clas.sort(key=lambda e: PERFILES.get(e, {}).get("rank", 999))
    return clas

# ── Simulación del cuadro por favorito (con opción de forzar un avance) ───────
def construir_bracket(clasificados, forzados):
    ronda = [(clasificados[i], clasificados[31 - i]) for i in range(16)]
    nombres = ["32avos", "Octavos", "Cuartos", "Semifinal", "Final"]
    rondas, ri = [], 0
    while len(ronda) >= 1:
        partidos, ganadores = [], []
        for a, b in ronda:
            m = predecir(a, b)
            if m is None:
                gan, pa, pb, pd = a, 0.5, 0.5, 0.0   # sin datos: reparto neutro, empate 0%
            else:
                forzado = forzados.get("0-forzado")
                gan = forzado if forzado in (a, b) else (a if m["pA"] >= m["pB"] else b)
                pa, pb, pd = m["pA"], m["pB"], m["pDraw"]   # 3 probabilidades: victoria A, victoria B y empate
            partidos.append({"a": a, "b": b, "pa": pa, "pb": pb, "pd": pd, "gan": gan})
            ganadores.append(gan)
        rondas.append({"nombre": nombres[ri] if ri < len(nombres) else f"Ronda {ri+1}",
                       "partidos": partidos})
        if len(ronda) == 1:
            break
        ronda = [(ganadores[i], ganadores[i + 1]) for i in range(0, len(ganadores), 2)]
        ri += 1
    return rondas

# ── Propagación probabilística por todo el cuadro → top campeones ────────────
def campeones_probables(clasificados):
    dists = []
    for i in range(16):
        a, b = clasificados[i], clasificados[31 - i]
        m = predecir(a, b)
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
                    m = predecir(na, nb)
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
def html_bracket(rondas, forzado):
    def fila(nombre, prob, gana):
        col = LIME if gana else MUTED
        estrella = f"<span style='color:{AMBER};font-size:10px;margin-left:5px'>★</span>" if nombre == forzado else ""
        return (f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 12px;"
                f"background:{PANELHI if gana else 'transparent'};border-left:3px solid "
                f"{AMBER if nombre == forzado else (LIME if gana else 'transparent')}'>"
                f"<span style='font-size:13px;font-weight:{700 if gana else 500};"
                f"color:{INK if gana else MUTED}'>{es(nombre)}{estrella}</span>"
                f"<span style='font-size:11px;font-weight:700;font-family:Archivo,sans-serif;color:{col}'>"
                f"{round(prob * 100)}%</span></div>")
    cols = []
    for rnd in rondas:
        # Cada cruce muestra las TRES probabilidades del enunciado: victoria A (fila superior),
        # empate (franja central) y victoria B (fila inferior).
        cards = "".join(
            f"<div class='wc-match'>{fila(p['a'], p['pa'], p['gan'] == p['a'])}"
            f"<div style='display:flex;justify-content:center;align-items:center;height:17px;"
            f"background:{PANEL};border-top:1px solid {LINE};border-bottom:1px solid {LINE}'>"
            f"<span style='font-size:9px;letter-spacing:1px;text-transform:uppercase;color:{MUTED}'>"
            f"empate {round(p['pd'] * 100)}%</span></div>"
            f"{fila(p['b'], p['pb'], p['gan'] == p['b'])}</div>"
            for p in rnd["partidos"])
        cols.append(f"<div class='wc-round'><div class='wc-round-h'>{rnd['nombre']}</div>"
                    f"<div class='wc-round-body'>{cards}</div></div>")
    return f"<div class='wc-bracket wc-rise'>{''.join(cols)}</div>"

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
            f"<span style='font-weight:700;font-size:15px;color:{INK}'>{es(eq)}{tit}</span>"
            f"<span style='font-family:Archivo,sans-serif;font-weight:800;font-size:16px;color:{LIME}'>"
            f"{p * 100:.1f}%</span></div>"
            f"<div style='background:{PANEL};border:1px solid {LINE};border-radius:4px;height:12px;overflow:hidden'>"
            f"<div style='width:{max(2, p / maxp * 100):.1f}%;height:100%;background:{barra};border-radius:3px;"
            f"animation:barGrow .6s ease'></div></div></div></div>")
    return f"<div class='wc-rise' style='max-width:800px'>{''.join(filas)}</div>"


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
st.session_state.setdefault("forzados", {})
st.session_state.setdefault("_aviso", None)     # mensaje pendiente para el toast

def grupos_actuales():
    return {g: [st.session_state[f"sel_{g}_{s}"] for s in range(4)] for g in GRUPOS_OFICIALES}

def restaurar_sorteo():
    for g, eqs in GRUPOS_OFICIALES.items():
        for s, e in enumerate(eqs):
            st.session_state[f"sel_{g}_{s}"] = e
    st.session_state.swap_sel = None
    st.session_state.forzados = {}

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
    eq = st.session_state.get("ff_team")
    if eq:
        st.session_state.forzados = {"0-forzado": eq}

def limpiar_forzados():
    st.session_state.forzados = {}


# ============================================================================
#  INTERFAZ
# ============================================================================
st.markdown(
    "<div class='wc-header'>"
    "<div class='wc-title'>WORLD CUP <span style='color:#c8f04a'>2026</span></div>"
    "<div class='wc-sub'>Consola de Inteligencia Deportiva</div>"
    "<div class='wc-meta'>Simulador predictivo con el modelo MLP entrenado · "
    "EE. UU. · Canadá · México · 48 selecciones</div>"
    "</div>", unsafe_allow_html=True)

nav, reset = st.columns([4, 1])
with nav:
    vista = st.segmented_control(
        "Vista", ["Fase de grupos", "Eliminación directa", "Campeones probables"],
        default="Fase de grupos", key="vista", label_visibility="collapsed")
with reset:
    st.button("↺ Restaurar sorteo", on_click=restaurar_sorteo, width="stretch")

grupos = grupos_actuales()

# Popup de advertencia si un cambio provocó (y resolvió) una selección repetida
if st.session_state["_aviso"]:
    st.toast(st.session_state["_aviso"], icon="⚠️")
    st.session_state["_aviso"] = None

# ── VISTA 1: FASE DE GRUPOS ──────────────────────────────────────────────────
if vista == "Fase de grupos" or vista is None:
    st.markdown(
        "<p class='wc-lead'>Los 12 grupos del sorteo oficial. Toca una selección y luego otra "
        "para <strong>intercambiarlas</strong> (incluso entre grupos distintos); usa "
        "<strong>✎ Editar</strong> para traer cualquiera de las 206 selecciones. Las tablas y "
        "los porcentajes de clasificación se recalculan con el modelo al instante.</p>",
        unsafe_allow_html=True)

    proyecciones = {g: proyectar_grupo(grupos[g]) for g in GRUPOS_OFICIALES}

    # CSS dirigido: borde de cal en los botones de los 2 primeros de cada grupo
    top2 = []
    for g, tabla in proyecciones.items():
        for fila in tabla[:2]:
            top2.append(f".st-key-swap_{g}_{grupos[g].index(fila['Selección'])} button"
                        "{border-left-color:#c8f04a}")
    st.markdown("<style>" + "".join(top2) + "</style>", unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    for idx, g in enumerate(GRUPOS_OFICIALES):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"<div class='wc-badge'><b>{g}</b><span>Grupo {g}</span></div>"
                            "<div class='wc-cols'><span>Selección · Pts · Goles esp.</span><span>Clasif.</span></div>",
                            unsafe_allow_html=True)
                for pos, fila in enumerate(proyecciones[g]):
                    nombre = fila["Selección"]
                    slot = grupos[g].index(nombre)
                    marcado = st.session_state.swap_sel == (g, slot)
                    # Etiqueta de la fila: puesto · selección · puntos proyectados ·
                    # goles esperados (a favor–en contra) · % de clasificación a la siguiente fase
                    etiqueta = (f"{pos + 1}.  {es(nombre)}   ·   {fila['pts']:.1f} pts   "
                                f"·   {fila['gf']:.1f}–{fila['ga']:.1f} goles   ·   {round(fila['Clasif.'] * 100)}%")
                    st.button(etiqueta, key=f"swap_{g}_{slot}", on_click=tap_swap, args=(g, slot),
                              width="stretch", type="primary" if marcado else "secondary")
                with st.expander("✎ Editar selecciones"):
                    for s in range(4):
                        st.selectbox(f"Puesto {s + 1}", EQUIPOS_DISPONIBLES,
                                     key=f"sel_{g}_{s}", on_change=reubicar, args=(g, s),
                                     format_func=es, label_visibility="visible")

    # Guardamos el estado actual como "previo" para el próximo cambio (reubicar)
    for g in GRUPOS_OFICIALES:
        for s in range(4):
            st.session_state[f"shadow_{g}_{s}"] = st.session_state[f"sel_{g}_{s}"]

# ── VISTA 2: ELIMINACIÓN DIRECTA ─────────────────────────────────────────────
elif vista == "Eliminación directa":
    st.markdown(
        "<p class='wc-lead'>Cuadro de eliminación simulado con el modelo, desde 32avos hasta la "
        "final. Usa el panel para <strong>forzar el avance</strong> de una selección y ver cómo se "
        "reconfigura el cuadro desde ahí.</p>", unsafe_allow_html=True)

    clasificados = clasificados_desde(grupos)
    rondas = construir_bracket(clasificados, st.session_state.forzados)
    campeon = rondas[-1]["partidos"][0]["gan"]
    forzado = st.session_state.forzados.get("0-forzado")

    st.markdown(f"<div class='wc-champ'><span class='lbl'>Campeón proyectado</span>"
                f"<span class='name'>{es(campeon)}</span></div>", unsafe_allow_html=True)

    with st.expander("Forzar el avance de una selección"):
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.selectbox("Selección que avanza", clasificados, key="ff_team",
                     format_func=es, label_visibility="collapsed")
        c2.button("Forzar", on_click=forzar_equipo, width="stretch")
        c3.button("Limpiar", on_click=limpiar_forzados, width="stretch")

    st.markdown(html_bracket(rondas, forzado), unsafe_allow_html=True)

# ── VISTA 3: CAMPEONES PROBABLES ─────────────────────────────────────────────
else:
    st.markdown(
        "<p class='wc-lead'>Las 10 selecciones con mayor probabilidad de levantar el trofeo, "
        "propagando el modelo por todo el cuadro. Se recalcula al modificar los grupos. "
        "La marca ★ indica títulos mundiales previos.</p>", unsafe_allow_html=True)

    clasificados = clasificados_desde(grupos)
    top = campeones_probables(clasificados)
    st.markdown(html_campeones(top), unsafe_allow_html=True)

st.markdown(
    "<p class='wc-foot'>El dashboard carga el modelo real (mlp_model_best.keras) y reconstruye las "
    "28 features con el mismo pipeline del notebook.</p>", unsafe_allow_html=True)
