
# Módulo Principal de Interfaz (app.py)
# Implementa un Dashboard de Business Analytics en Streamlit para la predicción de demanda
# y prescripción de acciones operativas/comerciales.


from typing import Tuple
from datetime import date
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


# Configuración global de la página y variables de diseño.
# Se define la paleta corporativa y se inyecta CSS personalizado para las tarjetas de métricas
# y alertas prescriptivas, asegurando contraste en modos claro y oscuro.

st.set_page_config(
    page_title="Kyqa Mhuysqa | BI & Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_PALETTE = {
    "Alta": "#2D6A4F",
    "Media": "#D4A373",
    "Baja": "#BC4749",
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
            color: #1A1A1A !important; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def load_and_prepare_data(seed: int) -> tuple[pd.DataFrame, bool]:

    # Gestiona la extracción y procesamiento de datos desde MySQL.
    # En caso de fallo de conexión, provee un conjunto de datos sintético
    # que respeta la distribución estadística del análisis descriptivo original.
    
    try:
        df_raw = fetch_historical_reservations()
        df_processed = run_etl_pipeline(df_raw, seed=seed)
        return df_processed, True
    except Exception as e:
        st.error(f"Error técnico de conexión: {e}")
        np.random.seed(seed)
        fechas = pd.date_range(start="2026-01-01", periods=202, freq="D")
        records = []
        for f in fechas:
            mysql_dow = ((f.weekday() + 1) % 7) + 1
            if mysql_dow == 1:
                vol = int(np.random.choice(range(8, 22), p=np.repeat(1/14, 14)))
            elif mysql_dow == 7:
                vol = int(np.random.choice(range(4, 15), p=np.repeat(1/11, 11)))
            elif mysql_dow == 4:
                vol = int(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
            else:
                vol = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]))
            records.append({"Fecha": f.date(), "volumen_real": vol, "dia_semana": mysql_dow})

        df_fallback = pd.DataFrame(records)
        return run_etl_pipeline(df_fallback, seed=seed), False



# Construcción de la barra lateral (Sidebar).
# Contiene los parámetros de entrenamiento del modelo predictivo y 
# los filtros globales de segmentación para el análisis histórico.

st.sidebar.markdown("## Panel de Control")
st.sidebar.caption("Configuración Analítica y Filtros")

max_depth_param = st.sidebar.slider(
    "Profundidad del Algoritmo (max_depth)",
    min_value=2,
    max_value=6,
    value=3,
    help="Limita el crecimiento del árbol de decisión para evitar el sobreajuste y garantizar interpretabilidad.",
)

#seed_param = st.sidebar.number_input(
#    "Semilla de Aleatoriedad (Seed)",
#    min_value=1,
#    max_value=9999,
#    value=42,
#    help="Garantiza la reproducibilidad de la inyección de variables exógenas.",
#)
seed_param = 42
df_historico, db_connected = load_and_prepare_data(seed=int(seed_param))
clf_model, model_metrics = train_decision_tree(
    df_historico, max_depth=max_depth_param, seed=int(seed_param)
)

if db_connected:
    st.sidebar.success("Conectado a MySQL: Activo")
else:
    st.sidebar.warning("Conexión MySQL local fallida. Operando con instancia de respaldo.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros de Exploración")

# Limitación de la fecha máxima al 31 de diciembre de 2026
real_min_date = df_historico["Fecha"].min().date()
real_max_date = df_historico["Fecha"].max().date()
limit_date = date(2026, 12, 31)

display_max_date = limit_date if real_max_date > limit_date else real_max_date
display_min_date = real_min_date if real_min_date <= display_max_date else display_max_date
date_range = st.sidebar.date_input(
    "Rango Temporal",
    value=(display_min_date, display_max_date),
    min_value=display_min_date,
    max_value=display_max_date,
)

selected_categories = st.sidebar.multiselect(
    "Nivel de Demanda",
    options=["Baja", "Media", "Alta"],
    default=["Baja", "Media", "Alta"],
)

selected_days = st.sidebar.multiselect(
    "Días Operativos",
    options=DIAS_ORDENADOS,
    default=DIAS_ORDENADOS,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = display_min_date, display_max_date

# Convertir las fechas simples a Timestamp para poder compararlas con la columna datetime64
start_dt = pd.to_datetime(start_d)
end_dt = pd.to_datetime(end_d)

df_filtered = df_historico[
    (df_historico["Fecha"] >= start_dt)
    & (df_historico["Fecha"] <= end_dt)
    & (df_historico["categoria_demanda"].isin(selected_categories))
    & (df_historico["nombre_dia"].isin(selected_days))
].copy()


 # Estructura principal del Dashboard.
# Encabezado corporativo y división del sistema en tres capas lógicas:
# Predicción, Diagnóstico y Documentación del Modelo.

st.markdown(
    """
    <div class="main-header">
        <h1>🌿 Dashboard Parque Ecoturístico Kyqa Mhuysqa</h1>
        <p>Análisis prescriptivo de ventas en las reservas del parque Kyqa Mhuysqa en el municipio de Facatativá.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_simulador, tab_descriptivo, tab_modelo = st.tabs(
    [
        "Simulador Predictivo & Prescripción",
        "Diagnóstico Operativo & Comercial",
        "Arquitectura del Modelo de Clasificación",
    ]
)


with tab_simulador:
    st.markdown(
        """
        **Objetivo Analítico:** Clasificar el volumen futuro de visitantes en capacidad ociosa (Baja), 
        operación sostenible (Media) o pico de demanda (Alta), integrando variables históricas y condiciones externas.
        """
    )

    col_inputs, col_results = st.columns([1, 1.65], gap="large")

    with col_inputs:
        st.markdown("#### Configuración de Variables Independientes")
        with st.container(border=True):
            inv_dias_map = {v: k for k, v in DIAS_SEMANA_MAP.items()}
            dia_seleccionado_nombre = st.selectbox(
                "Día de la Visita (Factor de Estacionalidad)",
                options=DIAS_ORDENADOS,
                index=6,
            )
            dia_codigo_mysql = inv_dias_map[dia_seleccionado_nombre]

            clima_seleccionado = st.radio(
                "Pronóstico Meteorológico",
                options=[0, 1],
                format_func=lambda x: "Tiempo Seco / Despejado" if x == 0 else "Alta Probabilidad de Lluvia",
                horizontal=False,
            )

            festivo_seleccionado = st.radio(
                "Calendario Oficial",
                options=[0, 1],
                format_func=lambda x: "Día Ordinario" if x == 0 else "Festividad o Puente",
                horizontal=False,
            )

            st.markdown("---")
            st.caption("**Métricas de Demanda Latente**")
            chats_whatsapp = st.slider(
                "Consultas en WhatsApp sin conversión",
                min_value=0,
                max_value=100,
                value=25,
                help="Volumen de prospectos activos en etapa de negociación.",
            )
            lead_time = st.slider(
                "Antelación de Reserva (Lead Time en días)",
                min_value=1,
                max_value=30,
                value=4,
            )

    resultado_pred = predict_scenario(
        model=clf_model,
        dia_semana=dia_codigo_mysql,
        es_festivo=festivo_seleccionado,
        prob_lluvia=clima_seleccionado,
        consultas_whatsapp=chats_whatsapp,
        lead_time_dias=lead_time,
    )

    with col_results:
        st.markdown("#### Salida del Modelo y Acciones Prescriptivas")

        cat_pred = resultado_pred["categoria"]
        badge_color = COLOR_PALETTE.get(cat_pred, "#2D6A4F")

        st.markdown(
            f"""
            <div style="background-color: {badge_color}; color: white; padding: 1.25rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    Proyección para el día {dia_seleccionado_nombre}
                </span>
                <h2 style="margin: 0.2rem 0; color: white; font-size: 2rem;">
                    Nivel {cat_pred.upper()} — {resultado_pred['estado_operativo']}
                </h2>
                <p style="margin: 0; font-size: 0.95rem; opacity: 0.95;">
                    Rango transaccional esperado: <b>{resultado_pred['rango_esperado']}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            title="Distribución de Probabilidad de Ocupación",
            height=230,
        )
        fig_probs.update_layout(
            xaxis_range=[0, 105],
            showlegend=False,
            margin=dict(l=10, r=20, t=35, b=10),
            xaxis_title="Confianza del Modelo (%)",
            yaxis_title="",
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        st.markdown(
            f"""
            <div class="action-box">
                <b>Directriz Operativa (Nómina y Recursos):</b><br>
                {resultado_pred['accion_operativa']}
            </div>
            <div class="action-box" style="border-left-color: #D4A373;">
                <b>Estrategia Comercial (Canales y Segmentación):</b><br>
                {resultado_pred['accion_comercial']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Matriz de Sensibilidad Semanal")
    st.caption(
        "Simulación de escenarios para los siete días de la semana evaluando variaciones meteorológicas y calendario oficial."
    )

    sim_records = []
    for d_nombre in DIAS_ORDENADOS:
        d_code = inv_dias_map[d_nombre]
        for fest_val, fest_str in [(0, "Ordinario"), (1, "Festivo")]:
            for lluv_val, lluv_str in [(0, "Clima Seco"), (1, "Con Lluvia")]:
                res = predict_scenario(clf_model, d_code, fest_val, lluv_val)
                sim_records.append(
                    {
                        "Día": d_nombre,
                        "Escenario": f"{fest_str} | {lluv_str}",
                        "Categoría Predicha": res["categoria"],
                        "Prob. Alta (%)": round(res["probabilidades"]["Alta"] * 100, 1),
                        "Prob. Media (%)": round(res["probabilidades"]["Media"] * 100, 1),
                        "Prob. Baja (%)": round(res["probabilidades"]["Baja"] * 100, 1),
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
        title="Probabilidad de Pico Operativo por Escenario",
        height=310,
    )
    fig_heatmap.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_heatmap, use_container_width=True)


with tab_descriptivo:
    st.markdown("### Indicadores Estratégicos de Desempeño")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_reservas_filtradas = int(df_filtered["volumen_real"].sum())
    pct_dias_baja = (
        (df_filtered["categoria_demanda"] == "Baja").mean() * 100
        if len(df_filtered) > 0
        else 0.0
    )

    kpi1.metric(
        "Reservas Periodo",
        f"{total_reservas_filtradas:,}",
        "Histórico Dom: 470 | Sáb: 279",
    )
    kpi2.metric(
        "Riesgo de Capacidad Ociosa",
        f"{pct_dias_baja:.1f}%",
        "Valle Crítico: Miércoles",
        delta_color="inverse",
    )
    kpi3.metric(
        "Plan B2C Dominante",
        "VIP Picnic",
        "> $20.0M COP acumulados",
    )
    kpi4.metric(
        "Concentración Riesgo B2B",
        "Bienestar Corp SAS",
        "14 transacciones corporativas",
    )
    kpi5.metric(
        "Fricción Digital (>50 años)",
        "92 Pagos",
        "Canal de origen: WhatsApp",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
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
            title="Distribución Asimétrica de la Demanda Semanal",
            labels={
                "nombre_dia": "Día Operativo",
                "volumen_real": "Flujo de Reservas",
                "categoria_demanda": "Clasificación",
            },
            barmode="stack",
        )
        st.plotly_chart(fig_asimetria, use_container_width=True)

    with col_right:
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
                title="Jerarquía de Demanda: Estacionalidad ➔ Condiciones ➔ Nivel",
            )
            fig_drill.update_layout(margin=dict(t=40, l=0, r=0, b=0))
            st.plotly_chart(fig_drill, use_container_width=True)
        else:
            st.info("Ajuste los parámetros de fecha en el panel lateral para cargar el análisis jerárquico.")
            
    st.markdown("#### Serie Temporal Histórica y Límites de Capacidad")
    fig_timeline = px.scatter(
        df_filtered,
        x="Fecha_dt",
        y="volumen_real",
        color="categoria_demanda",
        color_discrete_map=COLOR_PALETTE,
        size="volumen_real",
        hover_data=["nombre_dia", "clima_label", "festivo_label"],
        title="Volumen Transaccional Diario",
        labels={"Fecha_dt": "Línea de Tiempo", "volumen_real": "Transacciones Completadas"},
        height=360,
    )
    fig_timeline.add_hline(
        y=3.5,
        line_dash="dash",
        line_color="#BC4749",
        annotation_text="Límite Inferior (Capacidad Ociosa)",
    )
    fig_timeline.add_hline(
        y=10.5,
        line_dash="dash",
        line_color="#2D6A4F",
        annotation_text="Límite Superior (Saturación Operativa)",
    )
    fig_timeline.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("#### Comportamiento Transaccional y Demográfico")
    col_b2c, col_pagos = st.columns(2, gap="medium")

    with col_b2c:
        df_segmentos = pd.DataFrame(
            {
                "Segmento": [
                    "B2C: Plan VIP Picnic (36 a >50 años)",
                    "B2B: Bienestar Corp SAS (Fidelizado)",
                    "B2B: Empresas Esporádicas",
                ],
                "Inversión Estimada (COP)": [20500000, 1250000, 850000],
                "Tipo": ["Retención Consolidada", "Dependencia B2B", "Riesgo de Fuga"],
            }
        )
        fig_b2c = px.bar(
            df_segmentos,
            x="Inversión Estimada (COP)",
            y="Segmento",
            color="Tipo",
            orientation="h",
            color_discrete_sequence=["#1B4332", "#D4A373", "#BC4749"],
            title="Generación de Ingresos por Segmento de Negocio",
            text_auto=".3s",
        )
        fig_b2c.update_layout(showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_b2c, use_container_width=True)

    with col_pagos:
        df_brecha = pd.DataFrame(
            {
                "Rango Etario": [
                    "18 - 35 años (Nativos Digitales)",
                    "36 - 50 años (Transicional)",
                    "> 50 años (Alto Valor Conservador)",
                ],
                "Adopción Billeteras Digitales": [85, 58, 18],
                "Preferencia Efectivo Físico": [15, 42, 82],
            }
        )
        fig_pagos = px.bar(
            df_brecha,
            x="Rango Etario",
            y=[
                "Adopción Billeteras Digitales",
                "Preferencia Efectivo Físico",
            ],
            barmode="group",
            color_discrete_sequence=["#2D6A4F", "#DDA15E"],
            title="Brecha Tecnológica en Liquidación de Pagos (%)",
            labels={"value": "Proporción de Uso (%)", "variable": "Canal de Recaudo"},
        )
        fig_pagos.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pagos, use_container_width=True)


with tab_modelo:
    st.markdown("### Documentación Analítica del Modelo Predictivo")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(
        "Precisión Global del Modelo (Accuracy)",
        f"{model_metrics['accuracy'] * 100:.2f}%",
        "Métrica validada en set de prueba",
    )
    m_col2.metric(
        "Volumen de Entrenamiento",
        f"{model_metrics['train_samples']} registros",
        "Partición del 80% de datos históricos",
    )
    m_col3.metric(
        "Validación Cruzada",
        f"{model_metrics['test_samples']} registros",
        f"Control de Overfitting: depth={max_depth_param}",
    )

    col_tree_plot, col_imp = st.columns([1.65, 1], gap="medium")

    with col_tree_plot:
        st.markdown("#### Reglas Condicionales del Algoritmo")
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
        plt.title(f"Visualización Jerárquica del Árbol de Decisión (Profundidad Máxima = {max_depth_param})")
        st.pyplot(fig_mpl)

    with col_imp:
        st.markdown("#### Ponderación de Variables (Gini Importance)")
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
            title="Impacto Predictivo por Factor",
            height=260,
        )
        fig_imp.update_layout(showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Matriz de Confusión")
        fig_cm = px.imshow(
            model_metrics["confusion_matrix"],
            x=["Predicción Baja", "Predicción Media", "Predicción Alta"],
            y=["Realidad Baja", "Realidad Media", "Realidad Alta"],
            text_auto=True,
            color_continuous_scale="Greens",
            height=260,
        )
        fig_cm.update_layout(margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.info(
        "**Arquitectura:** El modelo ha sido desplegado en streamlit cloud y se integra con la base de datos MySQL alojada en TiDB Cloud. "
        "El pipeline ETL extrae el histórico meteorológico consumiendo la API de **Open-Meteo** (georreferenciada "
        "en las coordenadas de Facatativá) e integra el calendario oficial de Colombia (Leyes Emiliani) "
        "para ponderar el impacto de los puentes festivos. Esto permite al algoritmo evaluar los escenarios "
        "con datos del mundo real."
    )
