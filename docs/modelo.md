# Modelo de Machine Learning

## Variables

Orden usado en entrenamiento e inferencia:

1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `SkinThickness`
5. `Insulin`
6. `BMI`
7. `DiabetesPedigreeFunction`
8. `Age`

Variable objetivo: `Outcome`.

## Preprocesamiento

Los ceros biologicamente improbables se tratan como datos faltantes en:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

La imputacion por mediana se ejecuta dentro del pipeline para evitar fuga de datos. Los modelos sensibles a escala usan `StandardScaler` dentro del mismo pipeline.

## Modelos comparados

- Regresion logistica
- SVM con `probability=True`
- Random Forest
- KNN
- Arbol de decision

## Criterio de seleccion

La seleccion prioriza `recall` de la clase positiva porque el proyecto es preventivo. El desempate usa F1, precision y ROC-AUC para no ignorar falsos positivos.

## Metricas reales generadas

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | CV recall media | CV recall desv. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arbol de Decision | 0.7597 | 0.6393 | 0.7222 | 0.6783 | 0.7622 | 0.5555 | 0.0907 |
| Random Forest | 0.7468 | 0.6596 | 0.5741 | 0.6139 | 0.8168 | 0.6029 | 0.0121 |
| SVM | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 | 0.5842 | 0.0572 |
| KNN | 0.7273 | 0.6250 | 0.5556 | 0.5882 | 0.7900 | 0.5700 | 0.0388 |
| Regresion Logistica | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | 0.5748 | 0.0227 |

Modelo seleccionado: `Arbol de Decision` version `1.0.0`.

## Matriz de confusion

Etiquetas: `no_diabetes`, `diabetes`.

```text
[[78, 22],
 [15, 39]]
```

La curva ROC se genera en `FastApi/model/roc_curve.csv`.

## Limitaciones

- Dataset pequeno y academico.
- No incorpora historial clinico, laboratorio completo ni evaluacion profesional.
- La probabilidad es una estimacion del modelo, no una probabilidad clinica real.
- El sistema no debe usarse para diagnosticar diabetes.
