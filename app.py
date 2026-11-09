"""
Dashboard Gerencial y Operativo - Parque Kyqa Mhuysqa.
Aplicación interactiva para predicción de demanda y toma de decisiones prescriptivas.
"""

from typing import Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from dotenv import load_dotenv
load_dotenv(override=True)
from src.database import fetch_historical_reservations
from src.etl_model import (
    run_etl_pipeline,
    train_decision_tree,
    predict_scenario,
    DIAS_SEMANA_MAP,
    DIAS_ORDENADOS,
)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO CORPORATIVO (IDENTIDAD KYQA MHUYSQA)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kyqa Mhuysqa | BI & Analytics Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta corporativa Ecoturismo & Cultura Muisca (Verde Bosque, Tierra Dorada, Crema)
COLOR_PALETTE = {
    "Alta": "#2D6A4F",   # Verde bosque profundo (Pico operativo positivo)
    "Media": "#D4A373",  # Ocre / Dorado Muisca (Operación sostenible)
    "Baja": "#BC4749",   # Terracota / Rojo alerta (Capacidad ociosa)
}

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .main-header {
            background: linear-gradient(90deg, #1B4332 0%, #2D6A4F 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(27, 67, 50, 0.15);
        }
        .main-header h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .main-header p {
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
            color: #D8F3DC;
        }
        .metric-card {
            background-color: #FFFFFF;
            border-left: 5px solid #2D6A4F;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        .action-box {
            padding: 1.1rem 1.3rem;
            border-radius: 10px;
            background-color: #F8F9FA;
            border-left: 6px solid #1B4332;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS CON CACHÉ Y RESILIENCIA
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_and_prepare_data(seed: int) -> tuple[pd.DataFrame, bool]:
    """
    Carga los datos desde MySQL. Si el servicio local MySQL está detenido,
    notifica el estado y genera un respaldo con la distribución exacta del informe.
    """
    try:
        df_raw = fetch_historical_reservations()
        df_processed = run_etl_pipeline(df_raw, seed=seed)
        return df_processed, True
    except Exception as e:
        st.error(f"Error técnico de conexión: {e}")
        # Respaldo automático con las proporciones exactas del proyecto (202 días históricos)
        np.random.seed(seed)
        fechas = pd.date_range(start="2026-01-01", periods=202, freq="D")
        records = []
        for f in fechas:
            # Conversión de weekday de Python a DAYOFWEEK de MySQL (1=Dom, 2=Lun, ..., 7=Sáb)
            mysql_dow = ((f.weekday() + 1) % 7) + 1
            if mysql_dow == 1:  # Domingo (Pico histórico: 470 reservas)
                vol = int(np.random.choice(range(8, 22), p=np.repeat(1/14, 14)))
            elif mysql_dow == 7:  # Sábado (Segundo pico: 279 reservas)
                vol = int(np.random.choice(range(4, 15), p=np.repeat(1/11, 11)))
            elif mysql_dow == 4:  # Miércoles (Valle crítico: 24 reservas)
                vol = int(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
            else:  # Lunes, Martes, Jueves, Viernes
                vol = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]))
            records.append({"Fecha": f.date(), "volumen_real": vol, "dia_semana": mysql_dow})

        df_fallback = pd.DataFrame(records)
        return run_etl_pipeline(df_fallback, seed=seed), False


# -----------------------------------------------------------------------------
# 3. BARRA LATERAL: CONTROLES GLOBALES Y FILTROS INTERACTIVOS
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🌿 Panel de Control Kyqa")
st.sidebar.caption("Configuración del Modelo y Filtros Globales")

max_depth_param = st.sidebar.slider(
    "Profundidad del Árbol (max_depth)",
    min_value=2,
    max_value=6,
    value=3,
    help="En el informe se definió max_depth=3 para evitar el sobreajuste (overfitting) y preservar la interpretabilidad.",
)

seed_param = st.sidebar.number_input(
    "Semilla Estadística (Seed)",
    min_value=1,
    max_value=9999,
    value=42,
    help="Semilla 42 empleada en el código original (modelo.py) para simular clima y festivos.",
)

df_historico, db_connected = load_and_prepare_data(seed=int(seed_param))
clf_model, model_metrics = train_decision_tree(
    df_historico, max_depth=max_depth_param, seed=int(seed_param)
)

if db_connected:
    st.sidebar.success("✅ Conectado a MySQL (`kyqa_mhuysqa_grupo6`)")
else:
    st.sidebar.warning(
        "⚠️ MySQL no detectado en `localhost`. Ejecutando con réplica local del dataset histórico (202 fechas). Verifica tu archivo `.env`."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Filtros de Exploración Histórica")

min_date = df_historico["Fecha"].min()
max_date = df_historico["Fecha"].max()

date_range = st.sidebar.date_input(
    "Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

selected_categories = st.sidebar.multiselect(
    "Categoría de Demanda Operativa",
    options=["Baja", "Media", "Alta"],
    default=["Baja", "Media", "Alta"],
)

selected_days = st.sidebar.multiselect(
    "Días de la Semana",
    options=DIAS_ORDENADOS,
    default=DIAS_ORDENADOS,
)

# Aplicar filtros al DataFrame histórico
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

df_filtered = df_historico[
    (df_historico["Fecha"] >= start_d)
    & (df_historico["Fecha"] <= end_d)
    & (df_historico["categoria_demanda"].isin(selected_categories))
    & (df_historico["nombre_dia"].isin(selected_days))
].copy()


# -----------------------------------------------------------------------------
# 4. ENCABEZADO CORPORATIVO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 Parque Ecoturístico Kyqa Mhuysqa — Tablero Predictivo y Prescriptivo</h1>
        <p>Sistema de Soporte a la Decisión (DSS) para Optimización de Capacidad Instalada, Nómina Dinámica y Mitigación Climática</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. NAVEGACIÓN POR PESTAÑAS ESTRATÉGICAS
# -----------------------------------------------------------------------------
tab_simulador, tab_descriptivo, tab_modelo = st.tabs(
    [
        "🎯 1. Simulador Predictivo & Toma de Decisiones",
        "📊 2. Diagnóstico de Demanda & Perfil Comercial (Drill-Down)",
        "🌳 3. Arquitectura del Árbol de Decisión & Evaluación",
    ]
)

# =============================================================================
# PESTAÑA 1: SIMULADOR PREDICTIVO Y PRESCRIPTIVO (WHAT-IF)
# =============================================================================
with tab_simulador:
    st.markdown(
        """
        **Pregunta de Negocio:** *¿Qué días futuros presentarán un nivel de ocupación categorizado como **Bajo** (capacidad ociosa: 0-3 reservas), 
        **Medio** (operación sostenible: 4-10 reservas) o **Alto** (pico operativo: >11 reservas), considerando variables temporales y exógenas?*
        """
    )

    col_inputs, col_results = st.columns([1, 1.65], gap="large")

    with col_inputs:
        st.markdown("#### ⚙️ Asignación de Variables Predictoras")
        with st.container(border=True):
            # Selector de Día de la Semana invertido para obtener el código MySQL
            inv_dias_map = {v: k for k, v in DIAS_SEMANA_MAP.items()}
            dia_seleccionado_nombre = st.selectbox(
                "1. Día de la Semana de la Visita (Variable Histórica Raíz)",
                options=DIAS_ORDENADOS,
                index=6,  # Domingo por defecto
            )
            dia_codigo_mysql = inv_dias_map[dia_seleccionado_nombre]

            clima_seleccionado = st.radio(
                "2. Pronóstico Meteorológico (Variable Exógena)",
                options=[0, 1],
                format_func=lambda x: (
                    "☀️ Tiempo Seco / Baja Prob. de Lluvia (0)"
                    if x == 0
                    else "🌧️ Alta Probabilidad de Lluvia (1)"
                ),
                horizontal=False,
            )

            festivo_seleccionado = st.radio(
                "3. Calendario de Festividades y Puentes (Variable Binaria)",
                options=[0, 1],
                format_func=lambda x: (
                    "🏢 Día Hábil / Ordinario (0)"
                    if x == 0
                    else "🎉 Día Festivo / Puente Festivo (1)"
                ),
                horizontal=False,
            )

            st.markdown("---")
            st.caption("**Variables Operativas de Contexto Comercial (Prescripción)**")
            chats_whatsapp = st.slider(
                "Consultas por WhatsApp sin conversión (Demanda Latente)",
                min_value=0,
                max_value=100,
                value=25,
                help="Volumen de interacciones abiertas en Bogotá, Funza y El Rosal pendientes de cierre.",
            )
            lead_time = st.slider(
                "Tiempo de Antelación / Lead Time (Días)",
                min_value=1,
                max_value=30,
                value=4,
                help="Días restantes antes de la fecha operativa evaluada.",
            )

    # Ejecutar predicción en tiempo real
    resultado_pred = predict_scenario(
        model=clf_model,
        dia_semana=dia_codigo_mysql,
        es_festivo=festivo_seleccionado,
        prob_lluvia=clima_seleccionado,
        consultas_whatsapp=chats_whatsapp,
        lead_time_dias=lead_time,
    )

    with col_results:
        st.markdown("#### 📈 Respuesta del Modelo Predictivo y Plan de Acción")

        # Tarjeta de veredicto principal
        cat_pred = resultado_pred["categoria"]
        badge_color = COLOR_PALETTE.get(cat_pred, "#2D6A4F")

        st.markdown(
            f"""
            <div style="background-color: {badge_color}; color: white; padding: 1.25rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    Clasificación Estimada para {dia_seleccionado_nombre}
                </span>
                <h2 style="margin: 0.2rem 0; color: white; font-size: 2rem;">
                    Demanda {cat_pred.upper()} — {resultado_pred['estado_operativo']}
                </h2>
                <p style="margin: 0; font-size: 0.95rem; opacity: 0.95;">
                    Volumen esperado: <b>{resultado_pred['rango_esperado']}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gráfico estadístico de probabilidades por categoría
        df_probs = pd.DataFrame(
            {
                "Categoría": ["Baja (0-3)", "Media (4-10)", "Alta (>11)"],
                "Clase": ["Baja", "Media", "Alta"],
                "Probabilidad": [
                    resultado_pred["probabilidades"]["Baja"] * 100,
                    resultado_pred["probabilidades"]["Media"] * 100,
                    resultado_pred["probabilidades"]["Alta"] * 100,
                ],
            }
        )

        fig_probs = px.bar(
            df_probs,
            x="Probabilidad",
            y="Categoría",
            orientation="h",
            color="Clase",
            color_discrete_map=COLOR_PALETTE,
            text=df_probs["Probabilidad"].apply(lambda x: f"{x:.1f}%"),
            title="Distribución de Probabilidad en el Nodo Hoja del Árbol (%)",
            height=230,
        )
        fig_probs.update_layout(
            xaxis_range=[0, 105],
            showlegend=False,
            margin=dict(l=10, r=20, t=35, b=10),
            xaxis_title="Probabilidad Asignada (%)",
            yaxis_title="",
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        # Prescripciones ejecutables de negocio
        st.markdown(
            f"""
            <div class="action-box">
                <b>🛠️ Decisión Operativa (Nómina e Inventario):</b><br>
                {resultado_pred['accion_operativa']}
            </div>
            <div class="action-box" style="border-left-color: #D4A373;">
                <b>📲 Decisión Comercial (WhatsApp, Clima y Segmentación):</b><br>
                {resultado_pred['accion_comercial']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # SIMULADOR MULTIVARIABLE DE PLANIFICACIÓN SEMANAL (SENSIBILIDAD AL CLIMA)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🗓️ Matriz de Sensibilidad Semanal: Variación de Respuesta según Clima y Festivos")
    st.caption(
        "Compara cómo cambia la clasificación operativa para los 7 días de la semana al variar las condiciones meteorológicas y de calendario."
    )

    sim_records = []
    for d_nombre in DIAS_ORDENADOS:
        d_code = inv_dias_map[d_nombre]
        for fest_val, fest_str in [(0, "Ordinario"), (1, "Festivo")]:
            for lluv_val, lluv_str in [(0, "Seco ☀️"), (1, "Lluvia 🌧️")]:
                res = predict_scenario(clf_model, d_code, fest_val, lluv_val)
                sim_records.append(
                    {
                        "Día": d_nombre,
                        "Escenario": f"{fest_str} | {lluv_str}",
                        "Categoría Predicha": res["categoria"],
                        "Prob. Alta (%)": round(res["probabilidades"]["Alta"] * 100, 1),
                        "Prob. Media (%)": round(res["probabilidades"]["Media"] * 100, 1),
                        "Prob. Baja (%)": round(res["probabilidades"]["Baja"] * 100, 1),
                        "Acción Clave": (
                            "Reforzar Personal + Efectivo/QR"
                            if res["categoria"] == "Alta"
                            else (
                                "Campaña WhatsApp Planes Cubiertos"
                                if lluv_val == 1
                                else "Turnos Descanso + Frenar Perecederos"
                            )
                        ),
                    }
                )

    df_sim_matrix = pd.DataFrame(sim_records)
    fig_heatmap = px.density_heatmap(
        df_sim_matrix,
        x="Día",
        y="Escenario",
        z="Prob. Alta (%)",
        category_orders={"Día": DIAS_ORDENADOS},
        color_continuous_scale="Emrld",
        text_auto=True,
        title="Probabilidad de Pico Operativo (Demanda Alta %) según Escenario Climático y Festivo",
        height=310,
    )
    fig_heatmap.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_heatmap, use_container_width=True)


# =============================================================================
# PESTAÑA 2: DIAGNÓSTICO DESCRIPTIVO Y DRILL-DOWN DE NEGOCIO
# =============================================================================
with tab_descriptivo:
    st.markdown("### 📊 Indicadores Clave de Desempeño (KPIs) y Hallazgos de Negocio")

    # Fila de KPIs estratégicos alineados al informe
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_reservas_filtradas = int(df_filtered["volumen_real"].sum())
    pct_dias_baja = (
        (df_filtered["categoria_demanda"] == "Baja").mean() * 100
        if len(df_filtered) > 0
        else 0.0
    )

    kpi1.metric(
        "Reservas (Filtro Activo)",
        f"{total_reservas_filtradas:,}",
        "Pico Dom: 470 | Sáb: 279",
    )
    kpi2.metric(
        "Días en Capacidad Ociosa",
        f"{pct_dias_baja:.1f}%",
        "Valle crítico: Miércoles (24)",
        delta_color="inverse",
    )
    kpi3.metric(
        "Producto Estrella (B2C)",
        "VIP Picnic",
        "> $20.0M COP facturados",
    )
    kpi4.metric(
        "Concentración B2B",
        "Bienestar Corp SAS",
        "14 reservas (> $1.2M COP)",
    )
    kpi5.metric(
        "Pagos en Efectivo (>50 años)",
        "92 Transacciones",
        "Canal líder: WhatsApp",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        # Gráfico 1: Asimetría Semanal (¿Qué pasó?)
        df_by_day = (
            df_filtered.groupby(["nombre_dia", "categoria_demanda"], observed=False)["volumen_real"]
            .sum()
            .reset_index()
        )
        fig_asimetria = px.bar(
            df_by_day,
            x="nombre_dia",
            y="volumen_real",
            color="categoria_demanda",
            category_orders={"nombre_dia": DIAS_ORDENADOS, "categoria_demanda": ["Baja", "Media", "Alta"]},
            color_discrete_map=COLOR_PALETTE,
            title="A. ¿Qué pasó? Distribución Asimétrica de Reservas por Día",
            labels={
                "nombre_dia": "Día de la Semana",
                "volumen_real": "Volumen Acumulado de Reservas",
                "categoria_demanda": "Nivel de Demanda",
            },
            barmode="stack",
        )
        st.plotly_chart(fig_asimetria, use_container_width=True)

    with col_right:
        # Gráfico 2: Drill-Down Interactivo (Sunburst: Día -> Clima -> Categoría de Demanda)
        if not df_filtered.empty:
            df_sunburst = df_filtered.copy()
            df_sunburst["categoria_demanda"] = df_sunburst["categoria_demanda"].astype(str)

            fig_drill = px.sunburst(
                df_sunburst,
                path=["nombre_dia", "clima_label", "categoria_demanda"],
                values="volumen_real",
                color="categoria_demanda",
                color_discrete_map={
                    "Alta": "#2D6A4F",
                    "Media": "#D4A373",
                    "Baja": "#BC4749",
                    "(?)": "#E9ECEF",
                },
                title="🔍 Drill-Down Interactivo: Día ➔ Clima ➔ Nivel de Ocupación",
            )
            fig_drill.update_layout(margin=dict(t=40, l=0, r=0, b=0))
            st.plotly_chart(fig_drill, use_container_width=True)
        else:
            st.info("Selecciona al menos un rango y categoría en la barra lateral para visualizar el Drill-Down.")
            
    # Evolución temporal interactiva con Range Slider (Regresando al ancho total de la pestaña)
    st.markdown("#### 📈 Evolución Temporal de Ocupación y Umbrales Operativos")
    fig_timeline = px.scatter(
        df_filtered,
        x="Fecha_dt",
        y="volumen_real",
        color="categoria_demanda",
        color_discrete_map=COLOR_PALETTE,
        size="volumen_real",
        hover_data=["nombre_dia", "clima_label", "festivo_label"],
        title="Histórico Diario de Reservas con Límites de Clasificación Operativa",
        labels={"Fecha_dt": "Fecha de Visita", "volumen_real": "Reservas Diarias"},
        height=360,
    )
    fig_timeline.add_hline(
        y=3.5,
        line_dash="dash",
        line_color="#BC4749",
        annotation_text="Umbral Capacidad Ociosa (≤ 3 reservas)",
    )
    fig_timeline.add_hline(
        y=10.5,
        line_dash="dash",
        line_color="#2D6A4F",
        annotation_text="Umbral Pico Operativo (≥ 11 reservas)",
    )
    fig_timeline.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Sección de Contexto Comercial: ¿A quién le pasó? y ¿Dónde y cómo pasó?
    st.markdown("#### 👥 Radiografía Financiera, Demográfica y de Canales (Informe Base)")
    col_b2c, col_pagos = st.columns(2, gap="medium")

    with col_b2c:
        df_segmentos = pd.DataFrame(
            {
                "Segmento": [
                    "B2C: Plan VIP Picnic (36 a >50 años)",
                    "B2B: Bienestar Corp SAS (14 reservas)",
                    "B2B: Empresas Esporádicas (1 visita)",
                ],
                "Inversión Estimada (COP)": [20500000, 1250000, 850000],
                "Tipo": ["Mercado B2C", "Mercado B2B Fidelizado", "Mercado B2B En Riesgo"],
            }
        )
        fig_b2c = px.bar(
            df_segmentos,
            x="Inversión Estimada (COP)",
            y="Segmento",
            color="Tipo",
            orientation="h",
            color_discrete_sequence=["#1B4332", "#D4A373", "#BC4749"],
            title="B. ¿A quién le pasó? Concentración de Ingresos B2C vs. B2B",
            text_auto=".3s",
        )
        fig_b2c.update_layout(showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_b2c, use_container_width=True)

    with col_pagos:
        df_brecha = pd.DataFrame(
            {
                "Rango de Edad": [
                    "18 - 35 años (Jóvenes)",
                    "36 - 50 años (Adultos)",
                    "> 50 años (Senior - Alto Valor)",
                ],
                "Billeteras Digitales (Nequi/Daviplata/Bancolombia)": [85, 58, 18],
                "Efectivo en Taquilla (92 Transacciones Senior)": [15, 42, 82],
            }
        )
        fig_pagos = px.bar(
            df_brecha,
            x="Rango de Edad",
            y=[
                "Billeteras Digitales (Nequi/Daviplata/Bancolombia)",
                "Efectivo en Taquilla (92 Transacciones Senior)",
            ],
            barmode="group",
            color_discrete_sequence=["#2D6A4F", "#DDA15E"],
            title="C. ¿Dónde y cómo pasó? Brecha Generacional en Medios de Pago (%)",
            labels={"value": "Proporción de Uso (%)", "variable": "Medio de Pago"},
        )
        fig_pagos.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pagos, use_container_width=True)


# =============================================================================
# PESTAÑA 3: ARQUITECTURA DEL ÁRBOL DE DECISIÓN Y EVALUACIÓN TÉCNICA
# =============================================================================
with tab_modelo:
    st.markdown("### 🌳 Documentación Técnica y Rendimiento del Modelo (`DecisionTreeClassifier`)")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(
        "Exactitud Global (Accuracy)",
        f"{model_metrics['accuracy'] * 100:.2f}%",
        "Referencia experimental informe: 45.10%",
    )
    m_col2.metric(
        "Muestras de Entrenamiento (80%)",
        f"{model_metrics['train_samples']} días",
        f"Total histórico: {len(df_historico)} días",
    )
    m_col3.metric(
        "Muestras de Prueba (20%)",
        f"{model_metrics['test_samples']} días",
        f"Profundidad: max_depth={max_depth_param}",
    )

    col_tree_plot, col_imp = st.columns([1.65, 1], gap="medium")

    with col_tree_plot:
        st.markdown("#### Diagrama de Reglas del Árbol de Decisión")
        fig_mpl, ax = plt.subplots(figsize=(14, 6.5), dpi=150)
        plot_tree(
            clf_model,
            feature_names=["Día Semana", "Festivo", "Lluvia"],
            class_names=clf_model.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
        )
        plt.title(f"Estructura del Árbol de Decisión - Parque Kyqa Mhuysqa (max_depth={max_depth_param})")
        st.pyplot(fig_mpl, use_container_width=True)

    with col_imp:
        st.markdown("#### Peso Predictivo de las Variables")
        fig_imp = px.bar(
            model_metrics["feature_importances"],
            x="Importancia",
            y="Variable",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Tealgrn",
            text=model_metrics["feature_importances"]["Importancia"].apply(
                lambda x: f"{x * 100:.1f}%"
            ),
            title="Importancia Relativa (Gini Importance)",
            height=260,
        )
        fig_imp.update_layout(showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Matriz de Confusión (Set de Prueba)")
        fig_cm = px.imshow(
            model_metrics["confusion_matrix"],
            x=["Pred: Baja", "Pred: Media", "Pred: Alta"],
            y=["Real: Baja", "Real: Media", "Real: Alta"],
            text_auto=True,
            color_continuous_scale="Greens",
            height=260,
        )
        fig_cm.update_layout(margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.info(
        "💡 **Nota Metodológica y Hoja de Ruta a Producción:** El nodo raíz (`Día Semana <= 1.5`, correspondiente al domingo en MySQL) "
        "confirma que la estacionalidad semanal es el predictor dominante. El Accuracy experimental (~45.10%) se explica por la naturaleza "
        "sintética (`np.random.choice`) de las variables de lluvia y festivos durante esta fase académica. Para el pase a producción y "
        "elevar el Accuracy por encima del **80%**, el módulo `src/etl_model.py` está preparado para sustituir la simulación aleatoria "
        "por una **API meteorológica real** y el **calendario oficial de festivos de Colombia**."
    )