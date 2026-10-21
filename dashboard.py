"""
Tablero de decisión: pitcheo en el Estadio Harp Helú (Diablos Rojos del México).

Cinco secciones, pensadas para que un entrenador, un gerente deportivo o un jurado puedan usarlas sin leer código:

  Resumen                 la pregunta, la respuesta y cómo leer el tablero
  Candidatos              a quién contratar: ranking, incertidumbre y perfil de cada pitcher (contratacion.py)
  Laboratorio de pitches  el mismo pitch a nivel del mar y en Harp Helú (simulador_altitud.py)
  Efecto de la altura     qué tipos de pitch aguantan mejor el aire delgado
  Metodología             glosario, validación y límites

Corre en local con las salidas de ../salidas/ (datos del ISAC: no se suben a git).

Uso:
    streamlit run dashboard.py
"""
import json
from math import erf, sqrt

import altair as alt
import pandas as pd
import streamlit as st

from altitud_movimiento import RHO_HARP, RHO_MAR, ajustar_movimiento
from arsenal import RECTAS
from modelo_multiclase import CARPETA_SALIDAS
from simulador_altitud import TIPOS, calificar_pitch, cargar_recursos

RUTA_MUESTRA = CARPETA_SALIDAS / "muestra_movimiento.parquet"
COLUMNAS_FISICA = ["RelSpeed", "SpinRate", "IVB_n", "HB_n", "Extension", "RelHeight", "RelSide", "SpinAxis", "pitch_usage_pct"]

# Paleta Diablos Rojos sobre fondo oscuro: rojo del club, plata (valores neutros o favorables), grises y casi blanco para el texto.
# (AZUL se conserva como nombre de la variable; ahora es la plata.)
ROJO, AZUL, GRIS, GRIS_CLARO, TINTA = "#E4002B", "#D5DAE0", "#6B7280", "#2E2E35", "#F4F4F5"

NOMBRES_TIPO = {"Four-Seam": "Recta de cuatro costuras", "Sinker": "Sinker", "Cutter": "Cutter", "Slider": "Slider",
                "Sweeper": "Sweeper", "Curveball": "Curva", "Changeup": "Cambio de velocidad", "Splitter": "Splitter"}

st.set_page_config(page_title="Harp Helú · Evaluación de pitcheo", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
    #MainMenu, footer, header [data-testid="stToolbar"] {{visibility: hidden;}}
    .block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px;}}
    html, body, [class*="css"] {{font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;}}
    .encabezado {{background: linear-gradient(100deg, #000000 0%, #14080A 55%, #6E0A18 100%); border-bottom: 4px solid {ROJO};
                  padding: 1.3rem 1.6rem 1.1rem 1.6rem; margin-bottom: 1.4rem; border-radius: 2px;}}
    .encabezado .marca {{color: {ROJO}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase;}}
    .encabezado .titulo {{color: #FFFFFF; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.15; margin: 0.25rem 0 0.3rem 0;}}
    .encabezado .sub {{color: #B8BEC6; font-size: 0.98rem;}}
    h2, h3 {{font-weight: 650; color: {TINTA}; letter-spacing: -0.005em;}}
    h2 {{border-bottom: 2px solid {ROJO}; padding-bottom: 0.35rem; margin-top: 1.8rem; font-size: 1.3rem; text-transform: uppercase; letter-spacing: 0.04em;}}
    h3 {{font-size: 1.08rem;}}
    [data-testid="stMetric"] {{background: #16161A; border: 1px solid {GRIS_CLARO}; border-left: 4px solid {ROJO};
                               padding: 0.8rem 0.95rem; border-radius: 3px;}}
    [data-testid="stMetricLabel"] p {{font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: #9AA1AB;}}
    [data-testid="stMetricValue"] {{font-size: 1.75rem; font-weight: 700; color: #FFFFFF;}}
    section[data-testid="stSidebar"] {{background: #000000; border-right: 2px solid {ROJO};}}
    section[data-testid="stSidebar"] h3 {{color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.95rem;}}
    .nota {{background: #16161A; border-left: 4px solid {AZUL}; padding: 0.75rem 1rem; margin: 0.6rem 0 1rem 0;
            font-size: 0.93rem; color: #D6DAE0; border-radius: 2px;}}
    .alerta {{background: #2A0E13; border-left: 4px solid {ROJO}; padding: 0.75rem 1rem; margin: 0.6rem 0 1rem 0;
              font-size: 0.93rem; color: #F1D9DC; border-radius: 2px;}}
    .pie {{color: #7B8490; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid {GRIS_CLARO}; padding-top: 0.6rem;}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- datos
@st.cache_resource
def recursos():
    return cargar_recursos()


@st.cache_data
def muestra_liga():
    if not RUTA_MUESTRA.exists():
        from modelo_stuff_arsenal import cargar_entrenamiento_arsenal
        m = cargar_entrenamiento_arsenal().sample(20000, random_state=42)[["AutoPitchType"] + COLUMNAS_FISICA + ["fb_tipo"]]
        m.to_parquet(RUTA_MUESTRA, index=False)
    return pd.read_parquet(RUTA_MUESTRA)


@st.cache_data
def cargar_tablas():
    ranking = pd.read_csv(CARPETA_SALIDAS / "ranking_contratacion.csv")
    validacion = {}
    for nombre in ["stuff_arsenal", "location", "pitching"]:
        v = pd.read_csv(CARPETA_SALIDAS / f"validacion_{nombre}.csv")
        validacion[nombre] = float(v["mejora_%"].mean())
    return (pd.read_csv(CARPETA_SALIDAS / "delta_altitud_tipo.csv"), ranking,
            pd.read_csv(CARPETA_SALIDAS / "delta_altitud_pitcher_tipo.csv"),
            json.loads((CARPETA_SALIDAS / "prueba_retrospectiva.json").read_text(encoding="utf-8")), validacion,
            json.loads((CARPETA_SALIDAS / "validacion_test_2026.json").read_text(encoding="utf-8")))


def prob_mayor(dif, sd):
    """P(X > 0) para X normal con media `dif` y desviación `sd`."""
    return 0.5 * (1 + erf(dif / (sd * sqrt(2)))) if sd > 0 else 1.0


def nivel_confianza(r):
    return "Alta" if r >= 0.85 else "Media" if r >= 0.7 else "Baja"


def comparar_r(r_ranking, r_pasado, tolerancia=0.05):
    return "igual de bien" if abs(r_ranking - r_pasado) < tolerancia else "mejor" if r_ranking > r_pasado else "peor"


def nota(texto, clase="nota"):
    st.markdown(f'<div class="{clase}">{texto}</div>', unsafe_allow_html=True)


def tipo_es(t):
    return NOMBRES_TIPO.get(t, t)


rec, liga = recursos(), muestra_liga()
por_tipo, ranking, por_pitcher_tipo, prueba, validacion, val26 = cargar_tablas()
modelos26 = {m["modelo"]: m for m in val26["modelos"]}
rk26 = val26["ranking"]
calib = rec["calib"]
mediana = liga.groupby("AutoPitchType")[COLUMNAS_FISICA].median()
perdida_ivb = abs(float(ajustar_movimiento(mediana.loc["Four-Seam", "IVB_n"], mediana.loc["Four-Seam", "HB_n"], RHO_MAR, RHO_HARP, tipo="Four-Seam")[0])
                  - float(mediana.loc["Four-Seam", "IVB_n"]))
precio_batazo = abs(calib["delta_in_play"]) * (RHO_MAR - RHO_HARP)
externos = ranking[~ranking["diablos"]].sort_values("rating_100", ascending=False)
por_tipo = por_tipo.assign(tipo_es=por_tipo["AutoPitchType"].map(tipo_es))

# ---------------------------------------------------------------- navegación
SECCIONES = ["Resumen", "Candidatos", "Laboratorio de pitches", "Efecto de la altura", "Metodología y glosario"]
with st.sidebar:
    st.markdown("### Harp Helú")
    st.caption("Evaluación de pitcheo para los Diablos Rojos del México")
    seccion = st.radio("Sección", SECCIONES, key="seccion", label_visibility="collapsed")
    st.markdown("---")
    st.caption("**Qué hay en cada sección**")
    st.caption("Resumen: la pregunta y la respuesta.  \nCandidatos: a quién contratar.  \n"
               "Laboratorio: probar un pitch en dos aires.  \nEfecto de la altura: qué pitches la aguantan.  \n"
               "Metodología: definiciones, validación y límites.")
    st.markdown("---")
    st.caption("Datos de la LMB, temporadas 2024, 2025 y 2026. Pitchers y bateadores anónimos. Estadio Harp Helú a 2,232 m sobre el nivel del mar.")

st.markdown('<div class="encabezado"><div class="marca">Diablos Rojos del México</div>'
            '<div class="titulo">Harp Helú: quién lanza mejor a 2,232 metros</div>'
            '<div class="sub">Herramienta de decisión para pitcheo en altura. Hackathon ISAC 2026.</div></div>', unsafe_allow_html=True)


# ================================================================ RESUMEN
if seccion == "Resumen":
    st.header("La pregunta")
    st.markdown(
        "En Harp Helú el aire es **17 % menos denso** que a nivel del mar. La pelota frena menos, los pitches se mueven menos y un batazo "
        "en juego vale más carreras. Entonces: **¿a qué pitchers conviene contratar para lanzar ahí, y qué tipos de pitch resisten la altura?**")

    st.header("La respuesta en cuatro números")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pérdida promedio por la altura", f"{abs(calib['delta_total_medio']):.0f} puntos",
              help="Puntos de Stuff+ que pierde un pitch promedio al pasar de nivel del mar a Harp Helú (100 = pitch promedio, 10 puntos = una desviación estándar). "
                   "Casi todo es el valor extra de los batazos en juego: afecta a todos por igual.")
    c2.metric("Movimiento perdido por la recta", f"{perdida_ivb:.1f} pulgadas",
              help="Cuánto menos sube una recta de cuatro costuras (movimiento vertical inducido) en Harp Helú que a nivel del mar. Medido en la LMB, mismo pitcher en estadios distintos.")
    c3.metric("Costo extra de un batazo en juego", f"+{precio_batazo:.3f} carreras",
              help="Carreras adicionales que vale un batazo en juego en Harp Helú frente a nivel del mar, comparando al mismo bateador.")
    c4.metric("Acierto del ranking en 2026 (prueba externa)", f"r = {rk26['kbb']['r_rating']:.2f}",
              help=f"Correlación entre el ranking hecho solo con 2024 y 2025 y el K−BB% real de 2026, una temporada que ningún modelo vio. 0 = nada, 1 = perfecto. "
                   f"Es una relación débil a moderada (mirar el K−BB% pasado da r = {rk26['kbb']['r_resultado_pasado']:.2f}): sirve para una lista corta, no para garantizar un nombre.")

    st.header("Lo que encontramos")
    mejores = por_tipo.sort_values("delta_relativo", ascending=False).head(3)["tipo_es"].tolist()
    peores = por_tipo.sort_values("delta_relativo").head(2)["tipo_es"].tolist()
    st.markdown(
        f"1. **La altura castiga más a las rectas.** Los pitches que viven de subir pierden más que el promedio; "
        f"{', '.join(peores)} son los más afectados.\n"
        f"2. **Los pitches con quiebre resisten mejor.** {', '.join(mejores)} pierden menos que el promedio, porque dependen menos del movimiento vertical y su valor cae menos con el aire delgado.\n"
        f"3. **El parque cambia el valor de la pelota en juego, no solo el movimiento.** De los {abs(calib['delta_total_medio']):.0f} puntos que pierde un pitch promedio, "
        f"casi todos vienen de que los batazos valen más carreras; por eso comparamos siempre contra el pitch promedio.\n"
        f"4. **El ranking es una lista corta, no un veredicto.** Los primeros candidatos están separados por menos que su margen de incertidumbre. En la prueba externa de 2026 "
        f"los 20 primeros del ranking tuvieron un K−BB% de {100 * rk26['kbb']['top20']:.1f} contra {100 * rk26['kbb']['peor20']:.1f} de los 20 últimos (promedio {100 * rk26['kbb']['todos']:.1f}), "
        f"pero la correlación es débil (r = {rk26['kbb']['r_rating']:.2f}) y el solo resultado pasado predice mejor el K−BB% (r = {rk26['kbb']['r_resultado_pasado']:.2f}). "
        f"El valor del ranking está en el ajuste por altura y en separar ejecución de azar, no en reemplazar al criterio de los entrenadores.")

    st.header("Lista corta recomendada")
    corta = externos[externos["n_pitches"] >= 300].head(5)
    vista = pd.DataFrame({"Puesto": range(1, len(corta) + 1), "Pitcher": corta["pitcher_anon_id"].to_numpy(),
                          "Rating": corta["rating_100"].to_numpy(), "Margen (± 1 desv. est.)": corta["incertidumbre_100"].to_numpy(),
                          "Pitches analizados": corta["n_pitches"].astype(int).to_numpy(),
                          "Confianza": corta["confianza"].map(nivel_confianza).to_numpy(), "Arsenal principal": corta["arsenal"].to_numpy()})
    st.dataframe(vista.set_index("Puesto"), width="stretch", column_config={
        "Rating": st.column_config.NumberColumn(format="%+.2f", help="Carreras que evita por cada 100 pitches frente a un pitcher promedio, ya ajustado por la altura."),
        "Margen (± 1 desv. est.)": st.column_config.NumberColumn(format="±%.2f", help="Qué tanto puede moverse el rating real por la poca muestra.")})
    nota("**Cómo leerla.** El rating mide carreras evitadas por 100 pitches contra un pitcher promedio (más alto es mejor). Si dos candidatos se diferencian menos que su margen, "
         "el modelo no puede separarlos: decidan entre ellos por contrato, edad y salud. Excluye a los pitchers de los Diablos; se pueden incluir en la sección Candidatos.")

    st.header("Cómo usar este tablero")
    st.markdown("- **Candidatos:** filtre por muestra mínima y tipo de pitch, compare el rating con su margen y abra el perfil de cada pitcher.\n"
                "- **Laboratorio de pitches:** capture la velocidad, el spin y el movimiento que midió el radar y vea el mismo pitch en los dos aires.\n"
                "- **Efecto de la altura:** vea qué tipos de pitch pierden más o menos que el promedio.\n"
                "- **Metodología y glosario:** definiciones de cada término, validación del modelo y límites. Cada número del tablero tiene una ayuda al pasar el cursor sobre el signo de interrogación.")


# ================================================================ CANDIDATOS
elif seccion == "Candidatos":
    st.header("Candidatos para lanzar en Harp Helú")
    nota("**Qué muestra esta sección.** Todos los pitchers de la liga ordenados por **rating**: las carreras que evita por cada 100 pitches frente a un pitcher promedio, "
         "sumando (1) qué tan bien ejecuta y (2) cuánto de eso conserva en el aire de Harp Helú. Un rating de +1.0 equivale a unas 30 carreras evitadas en una temporada de abridor.")

    f1, f2, f3, f4 = st.columns([1, 1.1, 1.1, 1.3])
    min_pitches = f1.slider("Muestra mínima (pitches)", 300, 2000, 300, 100,
                            help="Pitches analizados en las tres temporadas (2024 a 2026). Con más muestra el rating es más confiable, pero quedan menos candidatos.")
    n_top = f2.select_slider("Candidatos en pantalla", [5, 10, 15, 20, 30], value=10)
    tipo_busca = f3.selectbox("Pitch principal", ["Cualquiera"] + [tipo_es(t) for t in sorted(ranking["tipo_principal"].dropna().unique())],
                              help="Filtra por el tipo de pitch que más usa el pitcher.")
    sin_diablos = f4.checkbox("Excluir pitchers de los Diablos", value=True,
                              help="Son los que lanzan de local en Harp Helú. Sus números ya traen parte del efecto de la altura, así que no son comparables con los de otros equipos.")

    cand = ranking[ranking["n_pitches"] >= min_pitches]
    if sin_diablos:
        cand = cand[~cand["diablos"]]
    if tipo_busca != "Cualquiera":
        cand = cand[cand["tipo_principal"].map(tipo_es) == tipo_busca]
    cand = cand.sort_values("rating_100", ascending=False).reset_index(drop=True)
    cand["puesto"] = cand.index + 1

    if len(cand) < 2:
        nota("Con esos filtros quedan menos de dos candidatos. Baje la muestra mínima o quite un filtro.", "alerta")
    else:
        a, b = cand.iloc[0], cand.iloc[1]
        p12 = prob_mayor(a["rating_100"] - b["rating_100"], (a["incertidumbre_100"] ** 2 + b["incertidumbre_100"] ** 2) ** 0.5)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidatos que cumplen los filtros", f"{len(cand)}")
        m2.metric("Mejor rating", f"{a['rating_100']:+.2f}", help=f"{a['pitcher_anon_id']}. Carreras evitadas por 100 pitches frente a un pitcher promedio.")
        m3.metric("Probabilidad de que el #1 supere al #2", f"{100 * p12:.0f} %",
                  help="Con la incertidumbre de la muestra de cada uno. 50 % = empate técnico. No incluye la incertidumbre del ajuste por altura.")
        m4.metric("Rating promedio del conjunto", f"{cand['rating_100'].mean():+.2f}", help="El promedio de la liga es cero por construcción.")
        if p12 < 0.7:
            nota(f"**Lectura.** {a['pitcher_anon_id']} (#1) y {b['pitcher_anon_id']} (#2) están en empate técnico ({100 * p12:.0f} % de probabilidad de que el primero sea mejor). "
                 "Trate al grupo de arriba como una lista corta y decida por contrato, edad, salud y rol.")
        elif p12 >= 0.8:
            nota(f"**Lectura.** {a['pitcher_anon_id']} se separa del resto: {100 * p12:.0f} % de probabilidad de ser mejor que el #2.")

        top = cand.head(n_top).copy()
        st.subheader(f"Rating de los {len(top)} mejores, con su margen de incertidumbre")
        top["inf"], top["sup"] = top["rating_100"] - top["incertidumbre_100"], top["rating_100"] + top["incertidumbre_100"]
        top["grupo"] = top["diablos"].map({True: "Pitcher de los Diablos", False: "Otros equipos"})
        orden = top["pitcher_anon_id"].tolist()
        base = alt.Chart(top).encode(y=alt.Y("pitcher_anon_id:N", sort=orden, title=None))
        barras = base.mark_rule(strokeWidth=3, color=GRIS).encode(
            x=alt.X("inf:Q", title="Rating (carreras evitadas por 100 pitches vs. un pitcher promedio)"), x2="sup:Q")
        puntos = base.mark_circle(size=130, opacity=1).encode(
            x="rating_100:Q",
            color=alt.Color("grupo:N", scale=alt.Scale(domain=["Otros equipos", "Pitcher de los Diablos"], range=[AZUL, ROJO]), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("pitcher_anon_id:N", title="Pitcher"), alt.Tooltip("rating_100:Q", title="Rating", format="+.2f"),
                     alt.Tooltip("incertidumbre_100:Q", title="Margen ±", format=".2f"), alt.Tooltip("n_pitches:Q", title="Pitches"), alt.Tooltip("arsenal:N", title="Arsenal")])
        cero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4], color=TINTA).encode(x="x:Q")
        st.altair_chart((barras + puntos + cero).properties(height=max(220, 34 * len(top))), width="stretch")
        st.caption("El punto es el rating estimado; la línea gris es ± 1 desviación estándar por la poca muestra. Si las líneas de dos pitchers se traslapan mucho, el modelo no los separa. "
                   "La línea punteada es el pitcher promedio (cero).")

        st.subheader("Tabla de candidatos")
        vista = pd.DataFrame({
            "Puesto": top["puesto"], "Pitcher": top["pitcher_anon_id"], "Rating": top["rating_100"], "Margen ±": top["incertidumbre_100"],
            "Ejecución": top["ejecucion_100"], "Ajuste por altura": top["ajuste_altitud_100"], "Confianza": top["confianza"].map(nivel_confianza),
            "Pitches": top["n_pitches"].astype(int), "Pitching+": top["pitching_plus"], "Location+": top["location_plus"],
            "Arsenal (uso)": top["arsenal"], "K−BB% real": 100 * top["kbb"], "Carreras evitadas real": -100 * top["runs_pa"]})
        st.dataframe(vista.set_index("Puesto"), width="stretch", column_config={
            "Rating": st.column_config.NumberColumn(format="%+.2f", help="Ejecución + ajuste por altura, en carreras evitadas por 100 pitches."),
            "Margen ±": st.column_config.NumberColumn(format="%.2f", help="Una desviación estándar de incertidumbre por la muestra."),
            "Ejecución": st.column_config.NumberColumn(format="%+.2f", help="Qué tan bien lanza en general (Pitching+: física más ubicación), ajustada por poca muestra."),
            "Ajuste por altura": st.column_config.NumberColumn(format="%+.2f", help="Positivo: su arsenal pierde menos que el promedio en Harp Helú. Negativo: pierde más. Cálculo del modelo, no validado con resultados reales."),
            "Confianza": st.column_config.TextColumn(help="Alta, Media o Baja según qué tanto pesa su propia muestra frente al promedio de la liga."),
            "Pitching+": st.column_config.NumberColumn(format="%.0f", help="100 = pitcher promedio, 110 = una desviación estándar mejor, entre pitchers."),
            "Location+": st.column_config.NumberColumn(format="%.0f", help="Solo ubicación. Misma escala que Pitching+."),
            "K−BB% real": st.column_config.NumberColumn(format="%.1f", help="Lo que realmente pasó: ponches menos bases por bolas, como porcentaje de turnos al bat."),
            "Carreras evitadas real": st.column_config.NumberColumn(format="%+.1f", help="Lo que realmente pasó: carreras evitadas por cada 100 turnos, ajustado por parque.")})
        st.download_button("Descargar candidatos (CSV)", cand.drop(columns=["indice"], errors="ignore").to_csv(index=False), "candidatos_harp_helu.csv", "text/csv")

        st.header("Perfil de un candidato")
        elegido = st.selectbox("Elija un pitcher de la tabla", top["pitcher_anon_id"].tolist(), index=0, key="perfil")
        fila = cand[cand["pitcher_anon_id"] == elegido].iloc[0]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Rating", f"{fila['rating_100']:+.2f}", help="Ejecución más ajuste por altura.")
        p2.metric("Ejecución", f"{fila['ejecucion_100']:+.2f}", help="Carreras evitadas por 100 pitches por su forma de lanzar, frente a un pitcher promedio.")
        p3.metric("Ajuste por altura", f"{fila['ajuste_altitud_100']:+.2f}", help="Cuánto menos (+) o más (−) pierde su arsenal que un arsenal promedio al pasar a Harp Helú.")
        p4.metric("Confianza", nivel_confianza(fila["confianza"]), f"{int(fila['n_pitches']):,} pitches", delta_color="off")
        st.markdown(
            f"**En palabras.** {elegido} lanza {'mejor' if fila['ejecucion_100'] >= 0 else 'peor'} que el pitcher promedio y su arsenal "
            f"({fila['arsenal']}) aguanta la altura {'mejor' if fila['ajuste_altitud_100'] >= 0 else 'peor'} que el promedio. "
            f"Sus resultados reales fueron K−BB% de {100 * fila['kbb']:.1f} y {-100 * fila['runs_pa']:+.1f} carreras evitadas por 100 turnos.")

        pt = por_pitcher_tipo[por_pitcher_tipo["pitcher_anon_id"] == elegido].copy()
        pt["uso"] = 100 * pt["n"] / pt["n"].sum()
        pt = pt[pt["uso"] >= 5].sort_values("uso", ascending=False)
        if len(pt):
            detalle = pd.DataFrame({"Tipo de pitch": pt["AutoPitchType"].map(tipo_es), "Uso (%)": pt["uso"], "Stuff+ a nivel del mar": pt["stuff_plus_mar"],
                                    "Stuff+ en Harp Helú": pt["stuff_plus_harp_centrado"], "Pierde de más (−) o de menos (+) que el promedio": pt["delta_relativo"]})
            st.dataframe(detalle.set_index("Tipo de pitch"), width="stretch", column_config={
                "Uso (%)": st.column_config.NumberColumn(format="%.0f"),
                "Stuff+ a nivel del mar": st.column_config.NumberColumn(format="%.0f", help="Calidad del pitch por su física, 100 = pitch promedio."),
                "Stuff+ en Harp Helú": st.column_config.NumberColumn(format="%.0f", help="En la escala propia de Harp Helú: 100 = pitch promedio lanzado ahí."),
                "Pierde de más (−) o de menos (+) que el promedio": st.column_config.NumberColumn(format="%+.1f", help="Puntos de Stuff+ frente a lo que pierde un pitch promedio en Harp Helú.")})

        with st.expander("Qué tanto confiar en este ranking (prueba con datos reales)", expanded=False):
            k, c = prueba["kbb"], prueba["runs_pa"]
            st.markdown(
                f"Se armó el ranking **solo con 2024** y se comparó con lo que pasó en **2025** ({prueba['n_pitchers']} pitchers):\n\n"
                f"- Los **20 primeros** tuvieron K−BB% de **{100 * k['top20_2025']:.1f}** en 2025, contra **{100 * k['todos_2025']:.1f}** del promedio y **{100 * k['peor20_2025']:.1f}** de los 20 últimos.\n"
                f"- En carreras por 100 turnos: **{100 * c['top20_2025']:+.1f}** los primeros, **{100 * c['todos_2025']:+.1f}** el promedio y **{100 * c['peor20_2025']:+.1f}** los últimos.\n"
                f"- La relación es **moderada** (r = {k['r_ranking']:.2f} con K−BB%, {c['r_ranking']:.2f} con carreras). En K−BB% predice {comparar_r(k['r_ranking'], k['r_resultado_pasado'])} "
                f"que mirar el resultado pasado (r = {k['r_resultado_pasado']:.2f}) y en carreras {comparar_r(c['r_ranking'], c['r_resultado_pasado'])} (r = {c['r_resultado_pasado']:.2f}).\n"
                f"- **Prueba externa en 2026** (ranking hecho con 2024 y 2025, {rk26['ranking_n'] if 'ranking_n' in rk26 else rk26['n_pitchers']} pitchers): r = {rk26['kbb']['r_rating']:.2f} con K−BB% y {rk26['runs_pa']['r_rating']:.2f} con carreras. "
                f"Los 20 primeros: K−BB% {100 * rk26['kbb']['top20']:.1f} contra {100 * rk26['kbb']['todos']:.1f} del promedio y {100 * rk26['kbb']['peor20']:.1f} de los 20 últimos. "
                f"Aquí el resultado pasado predice mejor que el ranking en K−BB% (r = {rk26['kbb']['r_resultado_pasado']:.2f}) y casi igual en carreras (r = {rk26['runs_pa']['r_resultado_pasado']:.2f}).\n"
                "- El **ajuste por altura** es un cálculo del modelo y **no se pudo validar** con resultados reales.")


# ================================================================ LABORATORIO
elif seccion == "Laboratorio de pitches":
    st.header("Laboratorio de pitches")
    nota("**Para qué sirve.** Capture lo que midió el radar de un pitch y vea cuánto vale a nivel del mar y en Harp Helú. "
         "Sirve para evaluar a un prospecto medido en otro estadio o para probar qué pasaría si un pitcher cambia su velocidad, su spin o su movimiento.  \n"
         "**Cómo se usa.** (1) Elija el pitch. (2) Indique dónde se midió. (3) Ajuste los valores; los iniciales son los típicos de la liga para ese tipo de pitch.")
    izq, der = st.columns([1, 1.45])
    with izq:
        st.subheader("1. El pitch")
        tipo = st.selectbox("Tipo de pitch", TIPOS, format_func=tipo_es, key="t2_tipo")
        d = mediana.loc[tipo]
        zurdo = st.radio("Mano del pitcher", ["Derecho", "Zurdo"], horizontal=True, key="t2_mano",
                         help="El radar mide a los zurdos al revés. Con Zurdo capture el movimiento horizontal, el lado de salida y el eje de giro tal como los dio el radar; el tablero los voltea antes de calificar.") == "Zurdo"
        sg = -1 if zurdo else 1
        k = f"{tipo}_{'z' if zurdo else 'd'}"
        st.subheader("2. Dónde se midió")
        medido = st.radio("Aire de la medición", ["Nivel del mar", "Harp Helú", "Otro"], horizontal=True, key="t2_aire",
                          help="El movimiento depende del aire. Si el radar midió el pitch en otro estadio, indíquelo para que el tablero lo lleve a una base común.")
        rho_medido = {"Nivel del mar": RHO_MAR, "Harp Helú": RHO_HARP}.get(medido)
        if rho_medido is None:
            rho_medido = st.slider("Densidad del aire (1.00 = nivel del mar)", 0.74, 1.15, 0.95, 0.01, key="t2_rho",
                                   help="Harp Helú ≈ 0.83. Un estadio a 1,000 m ≈ 0.89.")
        st.subheader("3. Lo que midió el radar")
        velo = st.slider("Velocidad de salida (mph)", 60.0, 105.0, float(d["RelSpeed"]), 0.5, key=f"v_{k}")
        spin = st.slider("Revoluciones por minuto (rpm)", 800, 3400, int(d["SpinRate"]), 10, key=f"s_{k}")
        ivb0, hb0 = ajustar_movimiento(d["IVB_n"], d["HB_n"], RHO_MAR, rho_medido, tipo=tipo)
        ivb = st.slider("Movimiento vertical, IVB (pulgadas)", -25.0, 30.0, float(round(ivb0, 1)), 0.5, key=f"i_{k}_{rho_medido:.3f}",
                        help="Positivo = la bola cae menos de lo que caería por gravedad.")
        hb = st.slider("Movimiento horizontal, HB (pulgadas; " + ("como lo dio el radar" if zurdo else "negativo = hacia el guante") + ")",
                       -30.0, 30.0, float(round(hb0 * sg, 1)), 0.5, key=f"h_{k}_{rho_medido:.3f}")
        with st.expander("Datos adicionales (opcional)"):
            ext = st.slider("Extensión (pies)", 4.5, 7.5, float(d["Extension"]), 0.1, key=f"e_{k}", help="Qué tanto se acerca a home al soltar la bola.")
            altura = st.slider("Altura de salida (pies)", 4.0, 7.5, float(d["RelHeight"]), 0.1, key=f"a_{k}")
            lado = st.slider("Lado de salida (pies)", -4.0, 4.0, float(d["RelSide"] * sg), 0.1, key=f"l_{k}")
            eje = st.slider("Eje de giro (grados)", 0, 360, int(d["SpinAxis"] if not zurdo else 360 - d["SpinAxis"]), 1, key=f"x_{k}")
            if tipo in RECTAS:
                es_recta = st.checkbox("Este pitch es su recta principal", value=True, key=f"fb_{k}")
            else:
                es_recta = False
                st.caption("Este pitch se compara con la recta principal del pitcher, así se entrenó el modelo.")
            fb = None
            if not es_recta:
                fb_tipo = st.selectbox("Su recta principal", RECTAS, format_func=tipo_es, key="t2_fbtipo")
                fd = mediana.loc[fb_tipo]
                i0, h0 = ajustar_movimiento(fd["IVB_n"], fd["HB_n"], RHO_MAR, rho_medido, tipo=fb_tipo)
                fb = {"tipo": fb_tipo,
                      "RelSpeed": st.slider("Velocidad de su recta (mph)", 70.0, 105.0, float(fd["RelSpeed"]), 0.5, key=f"t2_fbv_{fb_tipo}"),
                      "IVB": st.slider("IVB de su recta (pulgadas)", -5.0, 30.0, float(round(i0, 1)), 0.5, key=f"t2_fbi_{fb_tipo}_{rho_medido:.3f}"),
                      "HB": st.slider("HB de su recta (pulgadas)", -30.0, 30.0, float(round(h0 * sg, 1)), 0.5, key=f"t2_fbh_{fb_tipo}_{rho_medido:.3f}_{sg}"),
                      "RelHeight": st.slider("Altura de salida de su recta (pies)", 4.0, 7.5, float(fd["RelHeight"]), 0.1, key=f"t2_fbr_{fb_tipo}")}
            uso = st.slider("Porcentaje de sus pitches que son de este tipo", 1, 90, int(round(d["pitch_usage_pct"])), key=f"u_{k}")

    fisica = {"RelSpeed": velo, "SpinRate": spin, "Extension": ext, "RelHeight": altura, "RelSide": lado, "SpinAxis": eje}
    r = calificar_pitch(rec, tipo, fisica, ivb, hb, rho_medido=rho_medido, fb=fb, uso_pct=uso, mano="L" if zurdo else "R")
    relativo = r["delta"] - calib["delta_total_medio"]
    with der:
        st.subheader("Resultado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Stuff+ a nivel del mar", f"{r['stuff_mar']:.0f}", help="Calidad del pitch por su física. 100 = pitch promedio de la liga; cada 10 puntos = una desviación estándar.")
        c2.metric("Stuff+ en Harp Helú", f"{r['stuff_harp_centrado']:.0f}", f"{relativo:+.1f} vs. promedio", delta_color="off",
                  help="En la escala de Harp Helú: 100 = un pitch promedio lanzado ahí. El número de abajo es cuánto más (negativo) o menos (positivo) pierde este pitch que el promedio.")
        c3.metric("Pérdida total por la altura", f"{r['delta']:+.0f} puntos",
                  help=f"Un pitch promedio pierde {abs(calib['delta_total_medio']):.0f} puntos, sobre todo porque los batazos valen más carreras en Harp Helú. Lo que distingue a un pitch de otro es lo que pierde de más o de menos que el promedio.")
        if relativo >= 3:
            nota(f"**Lectura.** Este pitch **resiste bien la altura**: pierde {abs(relativo):.0f} puntos menos que un pitch promedio.")
        elif relativo <= -3:
            nota(f"**Lectura.** La altura **afecta de más** a este pitch: pierde {abs(relativo):.0f} puntos más que un pitch promedio.", "alerta")
        else:
            nota("**Lectura.** Este pitch **se comporta como un pitch promedio** en la altura.")

        st.subheader("De dónde viene la pérdida")
        canales = pd.DataFrame({"canal": ["Menos movimiento de la bola", "Resto del modelo (whiffs, fouls, bolas en juego)", "Los batazos en juego valen más carreras"],
                                "puntos": [r["delta_movimiento"], r["delta_resultado"], r["delta_precio"]]})
        st.altair_chart(alt.Chart(canales).mark_bar(size=22).encode(
            x=alt.X("puntos:Q", title="Puntos de Stuff+ (negativo = pierde valor)"), y=alt.Y("canal:N", sort=None, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.condition(alt.datum.puntos < 0, alt.value(ROJO), alt.value(AZUL)),
            tooltip=[alt.Tooltip("canal:N", title="Canal"), alt.Tooltip("puntos:Q", title="Puntos", format="+.2f")]).properties(height=150), width="stretch")
        st.caption("El tercer canal es un efecto de parque casi igual para todos los pitches. Los dos primeros dependen del pitch.")

        st.subheader("Mapa de movimiento")
        nube = liga[liga["AutoPitchType"] == tipo].sample(min(1500, int((liga["AutoPitchType"] == tipo).sum())), random_state=1)
        pts = pd.DataFrame({"HB": [r["hb_mar"], r["hb_harp"]], "IVB": [r["ivb_mar"], r["ivb_harp"]], "aire": ["Nivel del mar", "Harp Helú"]})
        fondo = alt.Chart(nube).mark_circle(size=12, opacity=0.25, color=GRIS).encode(
            x=alt.X("HB_n:Q", title="Movimiento horizontal, HB (pulgadas)"), y=alt.Y("IVB_n:Q", title="Movimiento vertical, IVB (pulgadas)"))
        linea = alt.Chart(pts).mark_line(color=TINTA, strokeDash=[4, 3]).encode(x="HB:Q", y="IVB:Q")
        puntos_ = alt.Chart(pts).mark_circle(size=170).encode(
            x="HB:Q", y="IVB:Q", color=alt.Color("aire:N", scale=alt.Scale(domain=["Nivel del mar", "Harp Helú"], range=[AZUL, ROJO]), legend=alt.Legend(title=None, orient="top")),
            tooltip=["aire", alt.Tooltip("HB:Q", format=".1f"), alt.Tooltip("IVB:Q", format=".1f")])
        st.altair_chart((fondo + linea + puntos_).properties(height=280), width="stretch")
        st.caption(f"Puntos grises: {tipo_es(tipo)} de la liga a nivel del mar. Plata: este pitch a nivel del mar. Rojo: el mismo pitch en Harp Helú, con menos aire y por tanto menos movimiento. "
                   "Convención del modelo: horizontal negativo = hacia el guante, para zurdos y derechos.")


# ================================================================ EFECTO DE LA ALTURA
elif seccion == "Efecto de la altura":
    st.header("Qué tipos de pitch aguantan la altura")
    nota("**Cómo leer la gráfica.** Cada barra compara un tipo de pitch con el **pitch promedio**. Hacia la derecha (plata), el pitch pierde **menos** valor que el promedio en Harp Helú. "
         "Hacia la izquierda (rojo), pierde **más**. Se mide en puntos de Stuff+: 10 puntos son una desviación estándar.")
    t = por_tipo.sort_values("delta_relativo")
    st.altair_chart(alt.Chart(t).mark_bar(size=26).encode(
        x=alt.X("delta_relativo:Q", title="Pierde de más (negativo) o de menos (positivo) que el pitch promedio, en puntos de Stuff+"),
        y=alt.Y("tipo_es:N", sort=None, title=None),
        color=alt.condition(alt.datum.delta_relativo < 0, alt.value(ROJO), alt.value(AZUL)),
        tooltip=[alt.Tooltip("tipo_es:N", title="Tipo"), alt.Tooltip("delta_relativo:Q", title="Frente al promedio", format="+.1f"),
                 alt.Tooltip("delta_movimiento:Q", title="Por movimiento", format="+.1f"), alt.Tooltip("n:Q", title="Pitches analizados", format=",")]).properties(height=330), width="stretch")
    mejores = t.tail(3)["tipo_es"].tolist()[::-1]
    peores = t.head(2)["tipo_es"].tolist()
    c1, c2 = st.columns(2)
    c1.metric("Resisten mejor", ", ".join(mejores))
    c2.metric("Pierden más", ", ".join(peores))

    st.subheader("Detalle por canal")
    detalle = pd.DataFrame({"Tipo de pitch": t["tipo_es"], "Pitches analizados": t["n"], "Stuff+ a nivel del mar": t["stuff_plus_mar"],
                            "Stuff+ en Harp Helú (escala propia)": t["stuff_plus_harp_centrado"],
                            "Por menos movimiento": t["delta_movimiento"], "Por el resto del modelo": t["delta_resultado"],
                            "Por valor de los batazos": t["delta_precio"], "Pérdida total": t["delta_total"], "Frente al promedio": t["delta_relativo"]})
    st.dataframe(detalle.iloc[::-1].set_index("Tipo de pitch"), width="stretch", column_config={
        "Pitches analizados": st.column_config.NumberColumn(format="%d"),
        "Stuff+ a nivel del mar": st.column_config.NumberColumn(format="%.1f"),
        "Stuff+ en Harp Helú (escala propia)": st.column_config.NumberColumn(format="%.1f", help="100 = pitch promedio lanzado en Harp Helú."),
        "Por menos movimiento": st.column_config.NumberColumn(format="%+.1f", help="Puntos que se pierden porque la bola se mueve menos en aire delgado."),
        "Por el resto del modelo": st.column_config.NumberColumn(format="%+.1f", help="Efecto de la densidad sobre whiffs, fouls y bolas en juego, además del movimiento. Es el canal menos validado."),
        "Por valor de los batazos": st.column_config.NumberColumn(format="%+.1f", help="Un batazo en juego vale más carreras en Harp Helú. Efecto de parque, casi igual para todos."),
        "Pérdida total": st.column_config.NumberColumn(format="%+.1f", help="Suma de los tres canales."),
        "Frente al promedio": st.column_config.NumberColumn(format="%+.1f", help="Pérdida total menos la pérdida del pitch promedio: lo útil para decidir.")})
    nota(f"Un pitch promedio pierde {abs(calib['delta_total_medio']):.0f} puntos en Harp Helú, casi todo por el valor de los batazos en juego (un efecto de parque, igual para todos). "
         "Por eso la columna importante es **Frente al promedio**. El sinker, un pitch de rolling, pierde más que el promedio; la hipótesis de que los pitchers de rolling aguantan mejor la altura "
         "**no se pudo probar** porque falta un modelo de calidad de contacto.")


# ================================================================ METODOLOGÍA
else:
    st.header("Metodología y glosario")
    st.subheader("Glosario")
    glosario = pd.DataFrame([
        ("Stuff+", "Calidad intrínseca de un pitch por su física: velocidad, spin, movimiento y punto de salida, más su relación con la recta del pitcher. 100 = pitch promedio; 10 puntos = una desviación estándar."),
        ("Location+", "Qué tan bien ubica: solo la ubicación en home, dado el conteo, el bateador y el tipo de pitch. Misma escala."),
        ("Pitching+", "Ejecución total: física, ubicación, conteo y lado del bateador juntos. Es la base del ranking porque fue la que mejor predijo el año siguiente."),
        ("Rating", "Carreras evitadas por cada 100 pitches frente a un pitcher promedio, en Harp Helú. Rating = ejecución + ajuste por altura."),
        ("Ejecución", "La parte del rating que viene de lanzar bien (Pitching+), en carreras por 100 pitches. Con pocos pitches se acerca al promedio de la liga."),
        ("Ajuste por altura", "La parte del rating que viene de cuánto conserva su arsenal en el aire de Harp Helú, frente a un arsenal promedio."),
        ("Confianza y margen", "Con pocos pitches el número es ruidoso. Confianza Alta, Media o Baja indica cuánto pesa su propia muestra; el margen es una desviación estándar de incertidumbre."),
        ("Densidad del aire", "1.00 = nivel del mar. Harp Helú ≈ 0.83: el aire frena menos la pelota y la bola se mueve menos."),
        ("IVB y HB", "Movimiento vertical inducido y movimiento horizontal, en pulgadas, sin contar la gravedad. En HB, negativo = hacia el guante, para zurdos y derechos."),
        ("K−BB%", "Ponches menos bases por bolas, como porcentaje de turnos al bat. Resultado real, no del modelo."),
        ("Pitcher promedio", "El promedio de la liga en 2024, 2025 y 2026. Un rating de cero significa igual al promedio."),
    ], columns=["Término", "Qué significa"])
    st.dataframe(glosario.set_index("Término"), width="stretch")

    st.subheader("Cómo se construye")
    st.markdown(
        "1. **Limpieza.** Se quitan bunts, pitches sin clasificar y errores del radar (1.5 % de las filas) y se voltean los zurdos para que el movimiento horizontal sea comparable.\n"
        "2. **Modelo de resultado.** Un modelo de aprendizaje automático predice, para cada pitch, la probabilidad de cinco resultados: bola, strike cantado, swing en vacío, foul y bola en juego. Se valida por pitcher: cada pitch lo predice un modelo que nunca vio a su pitcher.\n"
        "3. **Valor en carreras.** Cada resultado se traduce a carreras según el conteo y la situación de corredores y outs, a partir de los datos de la liga.\n"
        "4. **Altura.** Se mide el efecto del aire sobre el movimiento comparando al mismo pitcher en estadios distintos, y el valor extra de los batazos en juego comparando al mismo bateador. "
        "Con eso se evalúa cada pitch en los dos aires y se obtiene la pérdida por altura.\n"
        "5. **Ranking.** Rating = ejecución (Pitching+, ajustada por poca muestra) + ajuste por altura. Se prueba con datos reales: el ranking de 2024 contra los resultados de 2025.")

    st.subheader("Calidad de los modelos")
    tabla_q = pd.DataFrame({
        "Modelo": ["Stuff+ (versión oficial)", "Location+", "Pitching+"],
        "Validación cruzada por pitcher (%)": [validacion["stuff_arsenal"], validacion["location"], validacion["pitching"]],
        "Prueba externa en 2026 (%)": [modelos26["Stuff+ v3 (oficial)"]["mejora_%"], modelos26["Location+"]["mejora_%"], modelos26["Pitching+"]["mejora_%"]],
        "En pitchers nuevos de 2026 (%)": [modelos26["Stuff+ v3 (oficial)"]["mejora_nuevos_%"], modelos26["Location+"]["mejora_nuevos_%"], modelos26["Pitching+"]["mejora_nuevos_%"]]})
    st.dataframe(tabla_q.set_index("Modelo"), width="stretch", column_config={
        "Validación cruzada por pitcher (%)": st.column_config.NumberColumn(format="%.2f", help="Mejora del error de predicción (log-loss) sobre un modelo que solo conoce el conteo. Cada pitch lo predice un modelo que nunca vio a su pitcher."),
        "Prueba externa en 2026 (%)": st.column_config.NumberColumn(format="%.2f", help="La misma medida en 2026, con modelos entrenados solo con 2024 y 2025: temporada que ningún modelo vio."),
        "En pitchers nuevos de 2026 (%)": st.column_config.NumberColumn(format="%.2f", help="Solo pitchers que no aparecían en 2024 ni 2025.")})
    st.caption("Mejora del error de predicción frente a un modelo que solo usa el conteo. La física sola explica poco del resultado (Stuff+); la ubicación explica casi todo lo que separa bola de strike (Location+ y Pitching+). "
               "Los modelos mantienen su calidad en una temporada nueva y con pitchers nuevos. Los scores finales se reentrenaron con las tres temporadas.")

    st.subheader("Validación del ranking")
    k, c = prueba["kbb"], prueba["runs_pa"]
    st.markdown(
        f"Ranking hecho solo con 2024 y comparado con 2025, en {prueba['n_pitchers']} pitchers: correlación **{k['r_ranking']:.2f}** con K−BB% (intervalo 95 %: {k['r_ranking_ic95'][0]:.2f} a {k['r_ranking_ic95'][1]:.2f}) "
        f"y **{c['r_ranking']:.2f}** con carreras por turno. Los 20 primeros superaron al promedio y a los 20 últimos en ambas medidas. "
        "La relación es moderada: el ranking sirve para armar una lista corta confiable.")
    st.markdown(
        f"**Prueba externa en 2026.** Con el ranking hecho solo con 2024 y 2025 y comparado con 2026 ({rk26['n_pitchers']} pitchers), la correlación es **{rk26['kbb']['r_rating']:.2f}** con K−BB% "
        f"(intervalo 95 %: {rk26['kbb']['r_rating_ic95'][0]:.2f} a {rk26['kbb']['r_rating_ic95'][1]:.2f}) y **{rk26['runs_pa']['r_rating']:.2f}** con carreras por turno. "
        f"Los 20 primeros tuvieron K−BB% de {100 * rk26['kbb']['top20']:.1f} contra {100 * rk26['kbb']['todos']:.1f} del promedio y {100 * rk26['kbb']['peor20']:.1f} de los 20 últimos. "
        f"Es más débil que la prueba 2024 a 2025, y el resultado pasado solo predice mejor el K−BB% (r = {rk26['kbb']['r_resultado_pasado']:.2f}). Lo decimos tal cual: el ranking no sustituye a mirar el historial; "
        "aporta el ajuste por altura y una medida de ejecución que separa habilidad de azar.")

    st.subheader("Límites que conviene conocer")
    st.markdown(
        "- **No es una garantía.** El ranking predice de forma moderada el año siguiente; úselo para una lista corta.\n"
        "- **No considera** salud, edad, contrato, rol (abridor o relevo), disponibilidad ni quién es el pitcher: los datos son anónimos.\n"
        "- **El ajuste por altura no está validado** con resultados reales, porque casi no hay pitchers con muestra grande dentro y fuera de Harp Helú. Se calcula con el modelo.\n"
        "- **Los pitchers de los Diablos** ya traen parte de la pérdida de movimiento en sus números (38 % de sus pitches son en Harp Helú) y el ajuste la resta otra vez: su puesto no es comparable con el de otros equipos. Se identifican porque lanzan de local en Harp Helú.\n"
        "- **Las probabilidades** del tablero solo cuentan la incertidumbre de la muestra; no incluyen la del ajuste por altura ni que el talento cambie de un año a otro.\n"
        "- **No hay modelo de calidad de contacto**, así que no se puede decir si un pitcher de rolling aguanta mejor la altura.\n"
        "- **Equipos y estadios están reconstruidos** a partir de los datos (no vienen en el archivo); Harp Helú se identifica de forma sólida, otros estadios con más ruido.\n"
        "- Cada pitch se evalúa con un solo efecto de altura por tipo de pitch y movimiento, no uno por pitcher.")
    st.markdown('<div class="pie">Hackathon ISAC 2026. Código, decisiones y bitácora completos en el repositorio (docs/decisions.md).</div>', unsafe_allow_html=True)
