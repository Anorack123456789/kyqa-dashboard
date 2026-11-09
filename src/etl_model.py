"""
Módulo ETL, Entrenamiento del Árbol de Decisión y Motor Prescriptivo.
Implementa la lógica analítica definida para el Parque Kyqa Mhuysqa.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Mapeo oficial de DAYOFWEEK en MySQL (1 = Domingo, 2 = Lunes, ..., 7 = Sábado)
DIAS_SEMANA_MAP = {
    1: "Domingo",
    2: "Lunes",
    3: "Martes",
    4: "Miércoles",
    5: "Jueves",
    6: "Viernes",
    7: "Sábado",
}

DIAS_ORDENADOS = [
    "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
]


def run_etl_pipeline(df_raw: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Procesa los datos históricos de reservas, agrupa por fecha, clasifica los niveles
    de demanda operativa e inyecta las variables exógenas de validación experimental.

    Args:
        df_raw (pd.DataFrame): Datos crudos extraídos de la tabla Reserva.
        seed (int): Semilla aleatoria para reproducibilidad estadística.

    Returns:
        pd.DataFrame: Dataset transformado listo para el modelo y visualización.
    """
    df = df_raw.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.date

    # Consolidación diaria
    df_historico = (
        df.groupby("Fecha")
        .agg({"volumen_real": "sum", "dia_semana": "first"})
        .reset_index()
    )

    # Transformación a Modelo de Clasificación (Rangos ajustados al tráfico diario real)
    # Baja: 0-3 (Capacidad ociosa) | Media: 4-10 (Sostenible) | Alta: >11 (Pico operativo)
    bins = [-1, 3, 10, float("inf")]
    labels = ["Baja", "Media", "Alta"]
    df_historico["categoria_demanda"] = pd.cut(
        df_historico["volumen_real"], bins=bins, labels=labels
    ).astype(str)

    # Inyección controlada de variables predictivas exógenas (Clima y Festivos)
    np.random.seed(seed)
    df_historico["es_festivo"] = np.random.choice(
        [0, 1], size=len(df_historico), p=[0.9, 0.1]
    )
    df_historico["prob_lluvia"] = np.random.choice(
        [0, 1], size=len(df_historico), p=[0.7, 0.3]
    )

    # Etiquetas legibles para análisis y drill-down
    df_historico["nombre_dia"] = df_historico["dia_semana"].map(DIAS_SEMANA_MAP)
    df_historico["clima_label"] = df_historico["prob_lluvia"].map(
        {0: "Tiempo Seco / Despejado", 1: "Alta Prob. de Lluvia"}
    )
    df_historico["festivo_label"] = df_historico["es_festivo"].map(
        {0: "Día Ordinario", 1: "Festivo / Puente"}
    )
    df_historico["Fecha_dt"] = pd.to_datetime(df_historico["Fecha"])
    df_historico["Mes"] = df_historico["Fecha_dt"].dt.strftime("%Y-%m")

    return df_historico


def train_decision_tree(
    df_historico: pd.DataFrame, max_depth: int = 3, seed: int = 42
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Entrena el modelo DecisionTreeClassifier y calcula las métricas de desempeño.

    Args:
        df_historico (pd.DataFrame): Dataset procesado por el pipeline ETL.
        max_depth (int): Profundidad máxima del árbol para evitar sobreajuste.
        seed (int): Semilla para partición train/test reproducible.

    Returns:
        Tuple[DecisionTreeClassifier, Dict[str, Any]]: Modelo entrenado y diccionario de métricas.
    """
    features = ["dia_semana", "es_festivo", "prob_lluvia"]
    X = df_historico[features]
    y = df_historico["categoria_demanda"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth, class_weight="balanced", random_state=seed
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred, labels=["Baja", "Media", "Alta"])

    feature_importances = pd.DataFrame(
        {
            "Variable": ["Día de la Semana", "Día Festivo / Puente", "Pronóstico de Lluvia"],
            "Importancia": clf.feature_importances_,
        }
    ).sort_values(by="Importancia", ascending=True)

    metrics = {
        "accuracy": acc,
        "report": report_dict,
        "confusion_matrix": conf_mat,
        "feature_importances": feature_importances,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": list(clf.classes_),
    }

    return clf, metrics


def predict_scenario(
    model: DecisionTreeClassifier,
    dia_semana: int,
    es_festivo: int,
    prob_lluvia: int,
    consultas_whatsapp: int = 15,
    lead_time_dias: int = 3,
) -> Dict[str, Any]:
    """
    Evalúa un escenario específico ingresando valores de variables predictoras
    y genera la recomendación gerencial/operativa correspondiente.

    Args:
        model (DecisionTreeClassifier): Modelo entrenado.
        dia_semana (int): Código del día (1=Domingo ... 7=Sábado).
        es_festivo (int): 1 si es festivo/puente, 0 si es ordinario.
        prob_lluvia (int): 1 si hay pronóstico de lluvia, 0 si es seco.
        consultas_whatsapp (int): Volumen de chats sin conversión (demanda latente).
        lead_time_dias (int): Días de antelación respecto a la fecha objetivo.

    Returns:
        Dict[str, Any]: Categoría predicha, probabilidades y plan de acción de negocio.
    """
    input_df = pd.DataFrame(
        [[dia_semana, es_festivo, prob_lluvia]],
        columns=["dia_semana", "es_festivo", "prob_lluvia"],
    )

    pred_categoria = model.predict(input_df)[0]
    probas_raw = model.predict_proba(input_df)[0]
    probas_dict = dict(zip(model.classes_, probas_raw))

    es_fin_de_semana = dia_semana in [1, 7]

    # Motor de Reglas Prescriptivas basado en el Informe Ejecutivo
    if pred_categoria == "Baja":
        rango_esperado = "0 a 3 reservas diarias"
        estado_operativo = "Riesgo de Capacidad Ociosa"
        color_alerta = "#D9534F"
        accion_operativa = (
            "Nómina Dinámica e Inventario: Programar obligatoriamente los turnos de descanso "
            "del personal de atención al cliente y frenar órdenes de compra de inventario "
            "perecedero del restaurante para proteger el flujo de caja frente a la capacidad ociosa."
        )
        if es_fin_de_semana and prob_lluvia == 1:
            accion_comercial = (
                f"Mitigación Climática Urgente: El pronóstico de lluvia está reprimiendo un fin de semana. "
                f"Activar campaña masiva por WhatsApp sobre los {consultas_whatsapp} prospectos en espera "
                f"(Bogotá, Funza y El Rosal) ofreciendo venta cruzada hacia planes cubiertos."
            )
        else:
            accion_comercial = (
                f"Activación B2B e Impulso Entre Semana: Con {lead_time_dias} días de antelación, "
                f"desplegar asesores de alto valor hacia empresas de la sabana para reducir la "
                f"dependencia exclusiva de Bienestar Corp SAS y convertir las {consultas_whatsapp} "
                f"consultas latentes de WhatsApp mediante tarifas corporativas."
            )

    elif pred_categoria == "Media":
        rango_esperado = "4 a 10 reservas diarias"
        estado_operativo = "Nivel de Operación Sostenible"
        color_alerta = "#F0AD4E"
        accion_operativa = (
            "Dotación Estándar Balanceada: Mantener plantilla base en restaurante y senderos. "
            "Alinear abastecimiento de insumos según pre-reservas confirmadas."
        )
        if prob_lluvia == 1:
            accion_comercial = (
                f"Blindaje contra Lluvia por WhatsApp: Automatizar envío de mensajes a las "
                f"{consultas_whatsapp} conversaciones activas en Bogotá, Funza y El Rosal ofreciendo "
                f"migración sin recargo a zonas techadas para evitar cancelaciones de última hora."
            )
        else:
            accion_comercial = (
                "Upselling Focalizado: Asignar asesores especialistas en cierre de alto valor "
                "para escalar reservas básicas hacia el plan 'VIP Picnic' en el segmento de 36 a >50 años."
            )

    else:  # Alta
        rango_esperado = "Más de 11 reservas diarias"
        estado_operativo = "Pico Operativo de Demanda"
        color_alerta = "#2D6A4F"
        accion_operativa = (
            "Capacidad Máxima y Gestión de Pagos: Habilitar 100% del personal operativo, garantizar "
            "stock completo para 'VIP Picnic' y disponer cambio suficiente en efectivo en taquilla "
            "para el segmento mayor de 50 años, junto con códigos QR ágiles (Nequi, Daviplata, Bancolombia) "
            "para visitantes de 18 a 35 años."
        )
        if prob_lluvia == 1:
            accion_comercial = (
                "Contingencia Climática en Pico: Reacomodar logística hacia carpas y espacios cubiertos "
                "priorizando a los grupos confirmados por WhatsApp con pago anticipado."
            )
        else:
            accion_comercial = (
                "Maximización de Rentabilidad y Retención: Restringir descuentos promocionales, "
                "priorizar paquetes de alto margen ('VIP Picnic') y capturar datos de visitantes "
                "primerizos para fidelización recurrente."
            )

    return {
        "categoria": pred_categoria,
        "rango_esperado": rango_esperado,
        "estado_operativo": estado_operativo,
        "color": color_alerta,
        "probabilidades": {
            "Baja": probas_dict.get("Baja", 0.0),
            "Media": probas_dict.get("Media", 0.0),
            "Alta": probas_dict.get("Alta", 0.0),
        },
        "accion_operativa": accion_operativa,
        "accion_comercial": accion_comercial,
    }