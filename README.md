# Proyecto de Machine Learning - Análisis del Titanic

Durante el bootcamp de NEOLAND, realizamos dos ejercicios utilizando la base de datos pública del Titanic. El objetivo de estos ejercicios fue construir modelos de machine learning para predecir la probabilidad de supervivencia de los pasajeros en base a los datos disponibles.

## Ejercicio 1: Modelo de Regresión Logística

En el primer ejercicio, construimos un modelo de machine learning utilizando regresiones logísticas. El modelo se diseñó para predecir la probabilidad de supervivencia de los pasajeros basándose en diversas características como la edad, el sexo, la clase del boleto, entre otras.

Para este ejercicio, utilizamos las siguientes bibliotecas:

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
```

## Ejercicio 2: Modelo KKN
En el segundo ejercicio, construimos un modelo de machine learning utilizando el algoritmo K-Nearest Neighbors (KNN). Este modelo también se utilizó para predecir la probabilidad de supervivencia y compararlo con el modelo de regresión logística para determinar cuál de los dos era más óptimo.

Además de las bibliotecas anteriores, para este ejercicio utilizamos la siguiente:

```python
from sklearn import neighbors
```

## Conclusión

Estos ejercicios nos permitieron entender las fortalezas y debilidades de los modelos de regresión logística y KNN en la tarea de predicción, y cómo seleccionar el modelo más adecuado para un conjunto de datos específico.
