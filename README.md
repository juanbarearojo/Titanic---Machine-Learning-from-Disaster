# Titanic Predictive Modeling :ship:

Este proyecto implementa un modelo predictivo para predecir la supervivencia de pasajeros en el famoso desastre del Titanic. Utiliza un conjunto de técnicas de ingeniería de características y varios clasificadores para lograr un rendimiento predictivo sólido.

## Contenido del Archivo

- **titanic_predictor.py**: El archivo principal que contiene el código fuente de Python para el análisis y modelado predictivo.
- **ensemble.py**: Un módulo personalizado que implementa una clase `SklearnHelper` para ayudar en el proceso de ensamblaje de clasificadores.

## Estructura del Código

El código se organiza en varias secciones:

1. **Preprocesamiento de Datos y Limpieza**
   - Se leen los datos del conjunto de entrenamiento y prueba.
   - Se realizan operaciones de ingeniería de características, como la creación de variables y la imputación de valores nulos.

2. **Codificación de Variables Categóricas y Eliminación de Características No Relevantes**
   - Se codifican manualmente algunas variables categóricas.
   - Se eliminan características que no contribuyen significativamente al modelo.

3. **Definición de Clasificadores y Parámetros**
   - Se definen varios clasificadores, como Random Forest, AdaBoost, Gradient Boosting y Support Vector Classifier.
   - Se establecen los hiperparámetros para cada clasificador.

4. **Generación de Características con Clasificadores**
   - Se utilizan los clasificadores para generar características nuevas (OOF predictions).

5. **Modelo XGBoost**
   - Se entrena un modelo XGBoost utilizando las características generadas.

6. **Resultados y Exportación**
   - Se crea un archivo CSV con las predicciones del modelo para su posterior evaluación.

## Ejecución del Código

Para ejecutar el código, asegúrate de tener las bibliotecas requeridas instaladas y luego ejecuta el script principal:

```bash
python titanic_predictor.py
```

Los resultados finales se encuentran en el archivo submission.csv. ¡Siéntete libre de contribuir, mejorar el código o proporcionar sugerencias!

:rocket: ¡Esperamos que este modelo sirva como punto de partida interesante para análisis de datos y predicción en conjuntos de datos similares!

