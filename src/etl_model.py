
# Módulo ETL, Entrenamiento del Árbol de Decisión y Motor Prescriptivo.
# Implementa la lógica analítica definida para el Parque Kyqa Mhuysqa.


import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import holidays
from datetime import datetime, timedelta

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


def run_etl_pipeline(df_raw, seed=42):
    """
    Limpia los datos históricos y enriquece el modelo con el calendario
    oficial de Colombia y datos meteorológicos reales de Facatativá.
    """
    df = df_raw.copy()
    
    # 1. Asegurar formato de fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Fecha_dt'] = df['Fecha'] # Copia para gráficos
    df = df[df['Fecha'].dt.year >= 2021].copy()
    # 2. Ingeniería de Características: Días de la semana
    df['dia_semana'] = ((df['Fecha'].dt.dayofweek + 1) % 7) + 1
    df['nombre_dia'] = df['dia_semana'].map(DIAS_SEMANA_MAP)
    
    # 3. Integración Calendario Oficial (Festivos Colombia)
    co_holidays = holidays.Colombia()
    df['es_festivo'] = df['Fecha'].apply(lambda x: 1 if x in co_holidays else 0)
    df['festivo_label'] = df['es_festivo'].map({0: 'Día Ordinario', 1: 'Festivo / Puente'})

    # 4. Integración API Meteorológica Real (Open-Meteo Histórico - Facatativá)
    # Obtenemos los rangos de fechas del dataset para hacer una sola llamada masiva
    min_date = df['Fecha'].min().strftime('%Y-%m-%d')
    max_date = df['Fecha'].max().strftime('%Y-%m-%d')
    
    # Coordenadas aproximadas Parque Kyqa Mhuysqa / Facatativá
    limite_api = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    max_date_req = min(df['Fecha'].max().strftime('%Y-%m-%d'), limite_api)

    lat, lon = 4.81, -74.35
    url_clima = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={min_date}&end_date={max_date_req}&daily=precipitation_sum&timezone=America%2FBogota"

    try:
        response = requests.get(url_clima, timeout=10)
        data = response.json()
        
        if 'daily' not in data:
            print("Respuesta de la API:", data) # Imprime el error real si lo hay
            raise ValueError("No se encontró el bloque 'daily'.")
            
        weather_df = pd.DataFrame({
            'Fecha': pd.to_datetime(data['daily']['time']),
            'precipitacion_mm': data['daily']['precipitation_sum']
        })
        
        weather_df['prob_lluvia'] = weather_df['precipitacion_mm'].apply(lambda x: 1 if x > 3.0 else 0)
        df = df.merge(weather_df[['Fecha', 'prob_lluvia']], on='Fecha', how='left')
        
        # A los días del futuro que no pasaron por la API, les asignamos clima seco por defecto
        df['prob_lluvia'] = df['prob_lluvia'].fillna(0).astype(int)
        
    except Exception as e:
        print(f"Error técnico de la API del clima: {e}")
        np.random.seed(seed)
        df['prob_lluvia'] = np.random.choice([0, 1], size=len(df), p=[0.65, 0.35])

    df['clima_label'] = df['prob_lluvia'].map({0: 'Seco', 1: 'Lluvia'})

    # 5. Segmentación de la Variable Objetivo (Categoría de Demanda)
    condiciones = [
        (df['volumen_real'] <= 3),
        (df['volumen_real'] > 3) & (df['volumen_real'] <= 10),
        (df['volumen_real'] > 10)
    ]
    etiquetas = ['Baja', 'Media', 'Alta']
    df['categoria_demanda'] = np.select(condiciones, etiquetas, default='Baja')
    print(df[['Fecha', 'es_festivo', 'prob_lluvia']].head(10))
    return df
def train_decision_tree(
    df_historico: pd.DataFrame, max_depth: int = 3, seed: int = 42
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
   
    
    #Entrena el modelo DecisionTreeClassifier y calcula las métricas de desempeño.

   # Args:
       # df_historico (pd.DataFrame): Dataset procesado por el pipeline ETL.
      #  max_depth (int): Profundidad máxima del árbol para evitar sobreajuste.
     #   seed (int): Semilla para partición train/test reproducible.

    
    #Returns:
   #     Tuple[DecisionTreeClassifier, Dict[str, Any]]: Modelo entrenado y diccionario de métricas.
    
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
    
   # Evalúa un escenario específico ingresando valores de variables predictoras
    # y genera la recomendación gerencial/operativa correspondiente.

   # Args:
       # model (DecisionTreeClassifier): Modelo entrenado.
       # dia_semana (int): Código del día (1=Domingo ... 7=Sábado).
        #es_festivo (int): 1 si es festivo/puente, 0 si es ordinario.
       # prob_lluvia (int): 1 si hay pronóstico de lluvia, 0 si es seco.
      #  consultas_whatsapp (int): Volumen de chats sin conversión (demanda latente).
     #   lead_time_dias (int): Días de antelación respecto a la fecha objetivo.

   # Returns:
    #    Dict[str, Any]: Categoría predicha, probabilidades y plan de acción de negocio.
  
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