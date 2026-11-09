# 🌿 Dashboard Predictivo y Prescriptivo — Parque Kyqa Mhuysqa

Sistema interactivo de Business Analytics desarrollado en Python (**Streamlit**, **Scikit-Learn**, **Plotly** y **SQLAlchemy**) para el **Parque Ecoturístico Kyqa Mhuysqa** (Grupo 6 — Universidad de Cundinamarca, Extensión Facatativá).

---

## 🎯 Propósito de Negocio
Responder de forma interactiva a la pregunta predictiva del proyecto:
> *¿Qué días futuros presentarán un nivel de ocupación categorizado como **Bajo** (0-3 reservas, capacidad ociosa), **Medio** (4-10 reservas, operación sostenible) o **Alto** (>11 reservas, pico operativo), considerando variables temporales como días de la semana y festivos, junto con factores exógenos como el pronóstico del clima?*

A partir de la predicción del modelo `DecisionTreeClassifier`, el tablero automatiza tres líneas de decisión gerencial:
1. **Nómina Dinámica y Control de Inventario Perecedero:** En días clasificados como demanda **Baja**, programa descansos obligatorios del personal y detiene compras de insumos perecederos del restaurante.
2. **Mitigación Climática por WhatsApp:** Cuando un fin de semana baja su categoría esperada por el nodo de **Lluvia**, activa campañas por WhatsApp en Bogotá, Funza y El Rosal ofreciendo migración a planes cubiertos.
3. **Gestión de Pico Operativo y Brecha de Pagos:** En días de demanda **Alta**, garantiza stock del plan estrella (*VIP Picnic*) y equilibra la operación entre pagos en efectivo (segmento >50 años) y billeteras digitales Nequi/Daviplata/Bancolombia (segmento 18-35 años).

---

## 🗂️ Estructura del Proyecto
- `src/database.py`: Conexión segura a MySQL mediante `SQLAlchemy` y variables de entorno (`.env`).
- `src/etl_model.py`: Pipeline ETL, discretización operativa (`pd.cut`), entrenamiento del `DecisionTreeClassifier` y motor prescriptivo.
- `app.py`: Interfaz gráfica con paleta corporativa ecoturística, simulador interactivo *What-If*, matriz de sensibilidad semanal y gráficos *drill-down*.

---

## 🚀 Instrucciones de Instalación y Ejecución Local

### 1. Crear y activar el entorno virtual
```bash
python -m venv .venv
# En Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# En Linux / macOS:
source .venv/bin/activate