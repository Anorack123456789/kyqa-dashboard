# Dashboard Predictivo y Prescriptivo para el Parque Kyqa Mhuysqa

Sistema interactivo de Business Analytics desarrollado en Python (**Streamlit**, **Scikit-Learn**, **Plotly** y **SQLAlchemy**) para el Parque Ecoturístico Kyqa Mhuysqa

* **Integrante 1:** Jeison Sanchez Buitrago
* **Integrante 2:** Jaime Alberto Florez Paez
* **Integrante 3:** Andres Felipe Triana Ospina
* **Integrante 4:** Daniela Gomez Castro
* **Integrante 5:** Juan Nicolas Barreto Sanchez

## Contexto del Negocio

A partir del análisis descriptivo de los datos históricos del parque, se identificaron hallazgos críticos que motivaron el desarrollo de esta herramienta, entre los que se destaca la distribución asimétrica de la demanda, caracterizada por una alta concentración de reservas durante los fines de semana, con picos de hasta 470 reservas los domingos, frente a una baja ocupación entre semana, especialmente los miércoles, con apenas 24 reservas; asimismo, se evidenció una fuerte dependencia del modelo B2C y del plan VIP Picnic como principal fuente de ingresos, junto con un riesgo en el segmento B2B debido a la concentración de clientes corporativos en una sola empresa, Bienestar Corp SAS. 

Por otra parte, se identificó una brecha generacional en los métodos de pago, donde los clientes mayores de 50 años muestran una marcada preferencia por el efectivo, con 92 transacciones registradas, mientras que los jóvenes entre 18 y 35 años utilizan principalmente billeteras digitales como Nequi, Daviplata y Bancolombia; finalmente, se determinó que WhatsApp constituye el principal canal transaccional en las zonas geográficas de mayor actividad, especialmente Bogotá, Funza y El Rosal.

## Propósito de Negocio
Responder de forma interactiva a la pregunta predictiva del proyecto:
> *¿Qué días futuros presentarán un nivel de ocupación categorizado como **Bajo** (0-3 reservas, capacidad ociosa), **Medio** (4-10 reservas, operación sostenible) o **Alto** (>11 reservas, pico operativo), considerando variables temporales como días de la semana y festivos, junto con factores exógenos como el pronóstico del clima?*

A partir de la predicción del modelo `DecisionTreeClassifier`, el tablero automatiza tres líneas de decisión gerencial:
1. **Nómina Dinámica y Control de Inventario Perecedero:** En días clasificados como demanda Baja, programa descansos obligatorios del personal y detiene compras de insumos perecederos del restaurante.
2. **Mitigación Climática por WhatsApp:** Cuando un fin de semana baja su categoría esperada por el nodo de Lluvia, activa campañas por WhatsApp en Bogotá, Funza y El Rosal ofreciendo migración a planes cubiertos.
3. **Gestión de Pico Operativo y Brecha de Pagos:** En días de demanda **Alta**, garantiza stock del plan estrella (*VIP Picnic*) y equilibra la operación entre pagos en efectivo (segmento >50 años) y billeteras digitales Nequi/Daviplata/Bancolombia (segmento 18-35 años).

## Características del Dashboard
La aplicación  está dividida en tres pestañas:
*   **Simulador Predictivo:** Permite ingresar el día de la semana, pronóstico de lluvia, si es festivo, chats de WhatsApp en espera y días de antelación. Retorna la probabilidad de ocupación y emite las decisiones operativas y comerciales automáticas.
*   **Diagnóstico Descriptivo:** Un panel interactivo con los KPIs del parque, gráficos de asimetría semanal, y un diagrama Sunburst que permite profundizar en cómo el clima afecta cada día de la semana.
*   **Arquitectura del Modelo:** Expone la validación técnica del Árbol de Decisión  permitiendo la transparencia y confianza en el algoritmo.
##  Estructura del Proyecto
- `src/database.py`: Conexión segura a MySQL mediante `SQLAlchemy` y variables de entorno (`.env`).
- `src/etl_model.py`: Pipeline ETL, discretización operativa (`pd.cut`), entrenamiento del `DecisionTreeClassifier` y motor prescriptivo.
- `app.py`: Interfaz gráfica con paleta corporativa ecoturística, simulador interactivo *What-If*, matriz de sensibilidad semanal y gráficos *drill-down*.

## Instrucciones de Instalación y Ejecución Local

### 1. Crear y activar el entorno virtual usando PowerShell
```bash
python -m venv .venv
```
```bash
.\.venv\Scripts\Activate.ps1
```
### 2. Instalar las dependencias

```bash
pip install -r requirements.txt

```
### 3. Creacion de Base de datos en MySQL

Descargar y ejecutar en el orden establecido los scripts guardados en la siguiente carpeta de Drive

[Carpeta Drive](https://drive.google.com/drive/folders/1qDQh8nhNeHIb_PfnIB44Q-GIsSA3cTND?usp=drive_link)
### 4. Configurar las credenciales de MySQL

Copia el archivo `.env.example` a `.env` y verifica los parámetros de tu servidor local MySQL:

```ini
DB_USER=root
DB_PASSWORD=root
DB_HOST=localhost
DB_PORT=3306
DB_NAME=kyqa_mhuysqa_grupo6
RANDOM_SEED=42

```

### 5. Ejecutar el Dashboard

```bash
streamlit run app.py

```

El navegador se abrirá automáticamente en `http://localhost:8501`.

## Link de acceso en la Nube
[Enlace Dashboard](https://kyapp-dashboard-rsgz3vg5o2rjlw8m4pcgkk.streamlit.app/)