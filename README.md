# Bank-Marketing

# 🏦 Bank Marketing — Clasificación con PySpark MLlib

> Taller de procesamiento distribuido y Machine Learning sobre el dataset de marketing bancario de la UCI.  
> Pontificia Universidad Javeriana | Procesamiento de Alto Volumen de Datos

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación y Configuración](#instalación-y-configuración)
- [Configuración del Clúster Spark](#configuración-del-clúster-spark)
- [Ejecución del Notebook](#ejecución-del-notebook)
- [Pipeline del Proyecto](#pipeline-del-proyecto)
- [Modelos Implementados](#modelos-implementados)
- [Métricas de Evaluación](#métricas-de-evaluación)
- [Resultados Esperados](#resultados-esperados)
- [Autor](#autor)

---

## 📌 Descripción del Proyecto

Este repositorio contiene el desarrollo completo de un taller de ciencia de datos orientado al **procesamiento distribuido** y la **clasificación binaria** mediante **Apache PySpark**. Se trabaja sobre el dataset `bank-full.csv` del UCI Machine Learning Repository, que recoge datos de campañas de marketing telefónico de una entidad bancaria portuguesa.

El objetivo central es predecir si un cliente suscribirá un depósito a plazo fijo, pasando por todas las etapas del ciclo de vida de un proyecto de Machine Learning en entorno distribuido:

1. Ingesta y almacenamiento en **HDFS**
2. Exploración y análisis descriptivo (**EDA**)
3. Limpieza y balanceo de clases
4. Codificación y vectorización mediante **Spark ML Pipelines**
5. Entrenamiento y evaluación de **5 modelos de clasificación**
6. Comparación cuantitativa mediante métricas múltiples

---

## 📊 Dataset

| Campo | Valor |
|---|---|
| **Fuente** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Archivo** | `bank-full.csv` |
| **Registros** | ~45,211 |
| **Variables** | 17 (16 predictoras + 1 objetivo) |
| **Separador** | `;` (punto y coma) |
| **Variable objetivo** | `y` — suscripción al depósito (binaria: `yes` / `no`) |
| **Desbalance de clases** | ~88% `no` / ~12% `yes` |

### Variables del Dataset

#### Perfil del Cliente

| Variable | Tipo | Descripción |
|---|---|---|
| `age` | Entero | Edad del cliente |
| `job` | Categórica | Tipo de ocupación (12 categorías) |
| `marital` | Categórica | Estado civil (`divorced`, `married`, `single`, `unknown`) |
| `education` | Categórica | Nivel educativo (8 categorías) |
| `default` | Binaria | ¿Tiene crédito en incumplimiento? |
| `balance` | Entero | Saldo promedio anual en euros |
| `housing` | Binaria | ¿Tiene préstamo hipotecario? |
| `loan` | Binaria | ¿Tiene préstamo personal? |

#### Campaña Actual

| Variable | Tipo | Descripción |
|---|---|---|
| `contact` | Categórica | Canal de comunicación (`cellular`, `telephone`) |
| `day_of_week` | Fecha | Día de la semana del último contacto |
| `month` | Fecha | Mes del último contacto |
| `duration` | Entero | Duración del último contacto en segundos ⚠️ |
| `campaign` | Entero | Número de contactos en esta campaña |

> ⚠️ **Nota sobre `duration`**: esta variable solo se conoce *después* de realizar la llamada, por lo que **no debe usarse en modelos de predicción en tiempo real**. Se incluye únicamente con fines de benchmark.

#### Campañas Anteriores

| Variable | Tipo | Descripción |
|---|---|---|
| `pdays` | Entero | Días desde el último contacto previo (`-1` = nunca contactado) |
| `previous` | Entero | Número de contactos antes de esta campaña |
| `poutcome` | Categórica | Resultado de la campaña anterior (`failure`, `nonexistent`, `success`) |

#### Variable Objetivo

| Variable | Tipo | Descripción |
|---|---|---|
| `y` | Binaria | ¿Suscribió el depósito a plazo? (`yes` / `no`) |

---

## 🗂️ Estructura del Repositorio

```
bank-marketing-pyspark/
│
├── TallerClass_Beltran.ipynb   # Notebook principal del taller
├── README.md                            # Este archivo
│
├── data/
    └── bank-full.csv                    # Dataset original (descargar desde UCI)
```

> 📥 El archivo `bank-full.csv` **no está incluido** en el repositorio por su tamaño. Descárgalo directamente desde la [fuente oficial](https://archive.ics.uci.edu/dataset/222/bank+marketing) y colócalo en `data/` o cárgalo directamente en HDFS (ver sección de configuración).

---

## 💻 Requisitos del Sistema

### Software Base

| Componente | Versión recomendada |
|---|---|
| Python | 3.9+ |
| Apache Spark | 3.3+ |
| Hadoop (HDFS) | 3.x |
| Java (JDK) | 11 o 17 |
| Jupyter Notebook / JupyterLab | Cualquier versión reciente |

### Librerías Python

```txt
pyspark>=3.3.0
findspark>=2.0.1
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
```

> Todas se instalan automáticamente con las celdas de `!pip install` al inicio del notebook.

---

## ⚙️ Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/<tu-usuario>/bank-marketing-pyspark.git
cd bank-marketing-pyspark
```

### 2. Descargar el Dataset

```bash
# Opción A: descarga directa desde UCI (requiere ucimlrepo)
pip install ucimlrepo
python -c "
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=222)
ds.data.original.to_csv('data/bank-full.csv', sep=';', index=False)
"

# Opción B: descarga manual
# Ir a https://archive.ics.uci.edu/dataset/222/bank+marketing
# Descargar bank-marketing.zip → extraer bank-full.csv → mover a data/
```

### 3. Instalar Dependencias Python

```bash
pip install pyspark findspark pandas numpy matplotlib seaborn scikit-learn
```

### 4. Cargar el Dataset en HDFS

```bash
# Crear directorio en HDFS (si no existe)
hdfs dfs -mkdir -p /csv

# Subir el archivo al sistema de archivos distribuido
hdfs dfs -put data/bank-full.csv /csv/bank-full.csv

# Verificar que se cargó correctamente
hdfs dfs -ls /csv/
```

---

## 🔧 Configuración del Clúster Spark

El notebook está configurado para conectarse a un clúster Spark con las siguientes especificaciones. **Si usas un entorno diferente, actualiza las siguientes celdas**:

### Cambiar la dirección del nodo maestro

En la celda de configuración del notebook, localiza:

```python
configura.setMaster("spark://10.43.101.26:7077")
```

Reemplaza la IP por la dirección de tu nodo maestro. Para entornos locales:

```python
# Ejecución local (sin clúster) — útil para pruebas
configura.setMaster("local[*]")
```

### Cambiar la ruta del dataset

En la celda de carga de datos, localiza:

```python
df00 = sparkBeltran.read.format("csv") \
    .option("sep", ";") \
    .option("header", "true") \
    .load("file:///Almacen/bank-full.csv")
```

Ajusta la ruta según tu entorno:

```python
# Desde HDFS (clúster)
.load("hdfs:///csv/bank-full.csv")

# Desde sistema de archivos local
.load("file:///ruta/a/tu/data/bank-full.csv")

# Desde S3 (si usas AWS EMR)
.load("s3a://tu-bucket/bank-full.csv")
```

### Modo local sin clúster (para desarrollo)

Si no tienes acceso a un clúster Hadoop/Spark, puedes ejecutar todo en modo local reemplazando el bloque de configuración por:

```python
from pyspark.sql import SparkSession

sparkBeltran = SparkSession.builder \
    .appName("Metricas_Banca_Local") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

---

## ▶️ Ejecución del Notebook

### Con Jupyter

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# O con JupyterLab
jupyter lab
```

Abrir `TallerClass_Beltran_MEJORADO.ipynb` y ejecutar las celdas en orden con `Shift + Enter` o usando `Kernel → Restart & Run All`.

### Con nbconvert (ejecución por línea de comandos)

```bash
jupyter nbconvert --to notebook --execute TallerClass_Beltran_MEJORADO.ipynb \
    --output TallerClass_Beltran_ejecutado.ipynb
```

### Orden de ejecución recomendado

El notebook está diseñado para ejecutarse **de forma secuencial de arriba hacia abajo**. No omitas secciones intermedias, ya que cada sección genera objetos y DataFrames que son utilizados en las secciones posteriores.

| Sección | Descripción | Dependencias |
|---|---|---|
| 1 | Configuración e imports | Ninguna |
| 2 | Carga del dataset | Sección 1 |
| 3 | EDA y visualizaciones | Sección 2 |
| 4 | Calidad de datos | Sección 2 |
| 5 | Preparación y limpieza | Secciones 2–4 |
| 6 | Codificación y pipeline | Sección 5 |
| 7 | Entrenamiento de modelos | Sección 6 |
| 8 | Evaluación comparativa | Sección 7 |
| 9 | Cierre de sesión | Al finalizar |

---

## 🔄 Pipeline del Proyecto

```
Carga CSV (HDFS/local)
        ↓
Casting de tipos (string → int)
        ↓
EDA: histogramas, boxplots, correlaciones, crosstabs
        ↓
Filtrado de outliers (previous > 30)
        ↓
Eliminación de pdays (baja informatividad)
        ↓
Oversampling clase minoritaria (y = 'yes')
        ↓
StringIndexer + OneHotEncoder (variables categóricas)
        ↓
VectorAssembler → columna 'features'
        ↓
Almacenamiento en Parquet (HDFS)
        ↓
Split 80/20 (train/test)
        ↓
Entrenamiento de 5 modelos en paralelo
        ↓
Evaluación: Matriz de Confusión + Curva ROC + Métricas
        ↓
Tabla comparativa y análisis final
```

---

## 🤖 Modelos Implementados

| # | Modelo | Clase Spark ML | Hiperparámetros principales |
|---|---|---|---|
| 1 | Regresión Logística | `LogisticRegression` | `maxIter=10` |
| 2 | Árbol de Decisión | `DecisionTreeClassifier` | Profundidad por defecto |
| 3 | Random Forest | `RandomForestClassifier` | `numTrees=100`, `maxDepth=5` |
| 4 | Gradient Boosted Tree | `GBTClassifier` | `maxIter=10`, `maxDepth=5` |
| 5 | Support Vector Machine | `LinearSVC` | `maxIter=10`, `regParam=0.1` |

---

## 📐 Métricas de Evaluación

Para cada modelo se calculan las siguientes métricas sobre el conjunto de prueba (20%):

| Métrica | Clase Spark ML | Descripción |
|---|---|---|
| **Accuracy** | `MulticlassClassificationEvaluator` | Proporción de predicciones correctas |
| **Precisión (weighted)** | `MulticlassClassificationEvaluator` | Promedio ponderado de precisión por clase |
| **Recall (weighted)** | `MulticlassClassificationEvaluator` | Promedio ponderado de recall por clase |
| **F1-Score (weighted)** | `MulticlassClassificationEvaluator` | Media armónica de precisión y recall |
| **AUC-ROC** | `BinaryClassificationEvaluator` | Área bajo la curva ROC |
| **AUC-PR** | `BinaryClassificationEvaluator` | Área bajo la curva Precisión-Recall |

Adicionalmente, se generan para cada modelo:
- **Matriz de confusión** (heatmap con seaborn)
- **Curva ROC** graficada con sklearn + matplotlib

---

## 📈 Resultados Esperados

Los resultados concretos dependerán de la semilla aleatoria y del entorno de ejecución. En general, se espera la siguiente jerarquía de rendimiento:

```
GBT ≈ Random Forest > Árbol de Decisión > Regresión Logística ≈ SVM
```

Los modelos de ensamble basados en árboles superan consistentemente a los modelos lineales en este dataset, debido a la presencia de relaciones no lineales entre las variables predictoras y la variable objetivo.

---

## 👤 Autor

**Santiago Beltrán López**  
Estudiante de Ciencia de Datos — Quinto Semestre  
Pontificia Universidad Javeriana  
Materia: Procesamiento de Alto Volumen de Datos  
Mayo 2026

---

*Dataset original: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.*
