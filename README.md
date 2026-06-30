# Sistema de Diagnóstico Visual de Señalización Urbana (G2-UG-2026)

Proyecto de la asignatura **Inteligencia Artificial** — Universidad de Guayaquil, Facultad de Ingeniería Industrial.

## Integrantes del grupo (Grupo 2)

- Bajaña Cevallos Rosa
- Delgado Quiñonez Adonis
- Macias Carranza Krystel
- Suarez Barco Jose
- Tirado Mendoza Kelvin

**Docente:** Ing. Juan Carlos García

## Descripción del problema

La señalización vial urbana puede estar deteriorada, mal ubicada, poco iluminada u obstruida, lo cual representa un riesgo directo para la seguridad de conductores y peatones. La inspección manual de estas condiciones es lenta, costosa e inconsistente entre distintos evaluadores. Este proyecto explora si la visión por computador puede automatizar una auditoría visual preliminar del estado de las señales de tránsito a partir de fotografías.

## Objetivo del sistema

Construir un sistema de diagnóstico visual que, a partir de una imagen de una señal de tránsito, sea capaz de:

1. Clasificar el estado general de la señal (visible/legible, deteriorada o poco visible).
2. Localizar evidencias visuales concretas que expliquen esa clasificación (daño físico, obstrucción, condiciones de baja visibilidad).
3. Generar un diagnóstico textual interpretable que combine ambos resultados y sugiera una acción correctiva.

## Modelos utilizados

| Etapa | Modelo | Rol |
|---|---|---|
| Avance 1 | **CNN Baseline** (5 bloques convolucionales + skip connection residual, entrenada desde cero) | Punto de partida y control experimental |
| Avance 2 | **Transfer Learning — MobileNetV2** (fine-tuning en 2 fases sobre ImageNet) | Mejora de la clasificación general aprovechando representaciones preentrenadas |
| Avance 3 | **YOLO (detección de objetos)** | Localización de evidencias visuales puntuales (daños, obstrucciones, condiciones de baja visibilidad) dentro de la imagen |
| Propuesta para el proyecto final | **Modelo multimodal (CLIP / BLIP / VQA / Vision Transformer)** | Generación de descripciones automáticas y razonamiento visual-textual adicional |

## Estructura del repositorio

```
G2-SenalizacionUrbana/
├── README.md
├── notebooks/
│   ├── avance1_cnn_baseline.ipynb
│   ├── avance2_transfer_learning.ipynb
│   └── avance3_yolo_deteccion.ipynb
├── informes/
│   ├── avance1_informe.pdf
│   ├── avance2_informe.pdf
│   └── avance3_informe.pdf
├── dataset_clasificacion/
│   ├── visible_legible/
│   ├── deteriorada/
│   └── poco_visible/
├── dataset_yolo/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── modelos/
│   ├── modelo_3clases_v5.pth
│   ├── mobilenetv2_finetuned.pth
│   └── yolo_senales_best.pt
├── resultados/
│   ├── matrices_confusion/
│   ├── curvas_entrenamiento/
│   └── ejemplos_detecciones/
└── requirements.txt
```

## Instrucciones básicas de instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario-o-grupo>/G2-SenalizacionUrbana.git
cd G2-SenalizacionUrbana
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `scikit-learn`, `seaborn`, `matplotlib`, `pandas`.

### 3. Preparar el dataset

- El dataset de clasificación (`dataset_clasificacion/`) debe montarse desde Google Drive o copiarse localmente respetando la estructura por clase.
- El dataset de detección (`dataset_yolo/`) ya sigue el formato estándar de Ultralytics (`images/`, `labels/`, `data.yaml`).

### 4. Ejecutar los notebooks

Los notebooks están pensados para Google Colab (montaje de Drive incluido), pero pueden ejecutarse localmente ajustando `BASE_DIR`:

```bash
jupyter notebook notebooks/avance3_yolo_deteccion.ipynb
```

### 5. Entrenar el detector YOLO

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="dataset_yolo/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)
```

### 6. Ejecutar inferencia / diagnóstico

```python
from ultralytics import YOLO
model = YOLO("modelos/yolo_senales_best.pt")
resultados = model.predict("ruta/a/imagen.jpg", conf=0.4)
```

## Resumen de resultados obtenidos

| Modelo | Accuracy / mAP50 (test) | Observación principal |
|---|---|---|
| CNN Baseline (Avance 1) | Accuracy test: 44.4% (train: 81.9%) | Sobreajuste pronunciado por dataset pequeño (240 imágenes) |
| MobileNetV2 Transfer Learning (Avance 2) | Pendiente de validación con métricas reales | Notebook estructurado; resultados deben recalcularse para evitar valores poco realistas |
| YOLO (Avance 3) | _Completar tras entrenamiento_ | Primer detector entrenado sobre clases visuales concretas (daño físico, obstrucción, baja visibilidad) |

## Limitaciones actuales

- Dataset de clasificación reducido (80 imágenes por clase), lo que favorece el sobreajuste observado en la CNN baseline.
- Alta similitud visual entre las clases `deteriorada` y `poco_visible`, especialmente en condiciones de baja iluminación nocturna.
- El dataset YOLO requiere anotación manual adicional y aún no alcanza el volumen necesario para una detección robusta en producción.
- El sistema actual no realiza análisis temporal (video) ni explica visualmente sus decisiones (sin Grad-CAM ni mapas de atención todavía).
- La propuesta multimodal (CLIP/BLIP/VQA) es exploratoria y no está integrada de forma productiva en el pipeline.

## Próximos pasos para el proyecto final (12 de julio)

1. Ampliar y balancear el dataset de clasificación y el dataset YOLO (más imágenes por clase, mayor diversidad de condiciones).
2. Completar el entrenamiento y validación de YOLO con métricas de precision, recall, mAP50 y mAP50-95 reportadas de forma confiable.
3. Integrar formalmente clasificación (CNN/MobileNetV2) + detección (YOLO) en un único pipeline de inferencia.
4. Implementar el módulo de diagnóstico textual automático combinando ambas salidas.
5. Incorporar un modelo multimodal (CLIP o BLIP) como capa adicional de interpretación o generación de descripciones.
6. Evaluar el sistema completo sobre un conjunto de prueba final y documentar resultados en el informe final.
7. Dejar el repositorio organizado, documentado y reproducible para la entrega final.
