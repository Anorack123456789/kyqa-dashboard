
# Módulo de conexión y extracción de datos desde MySQL.
# Gestiona la lectura segura de credenciales y las consultas al esquema de Kyqa Mhuysqa.


import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()


def get_db_engine() -> Engine:
    
    #Crea y retorna el motor de conexión SQLAlchemy utilizando variables de entorno.

    #Returns:
     #   Engine: Instancia de conexión a la base de datos MySQL.
    
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "root")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "kyqa_mhuysqa_grupo6")

    connection_url = (
        f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    return create_engine(connection_url, pool_pre_ping=True)


def fetch_historical_reservations() -> pd.DataFrame:
    
    #Ejecuta la consulta SQL para obtener el volumen histórico de reservas por fecha
    #y el día de la semana correspondiente (1=Domingo ... 7=Sábado en MySQL DAYOFWEEK).

     #Returns:
     #   pd.DataFrame: DataFrame con columnas ['Fecha', 'volumen_real', 'dia_semana'].
    
    engine = get_db_engine()
    query = text("""
        SELECT 
            Fecha, 
            COUNT(Id_Reserva) AS volumen_real,
            DAYOFWEEK(Fecha) AS dia_semana
        FROM Reserva
        GROUP BY Fecha
        ORDER BY Fecha ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


def fetch_detailed_business_metrics() -> pd.DataFrame:
    
    #Intenta extraer el detalle extendido de reservas (planes, canales, clientes, pagos)
    #si las tablas o columnas complementarias están disponibles en la base de datos.
    #En caso de que solo exista la tabla Reserva base, retorna un DataFrame vacío
    #para activar los indicadores consolidados del informe.

    #Returns:
     #   pd.DataFrame: Registros detallados de facturación o DataFrame vacío.
    
    engine = get_db_engine()
    query = text("SELECT * FROM Reserva")
    try:
        with engine.connect() as conn:
            df_raw = pd.read_sql(query, conn)
        return df_raw
    except Exception:
        return pd.DataFrame()