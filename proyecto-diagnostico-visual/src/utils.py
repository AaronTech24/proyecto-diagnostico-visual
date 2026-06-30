"""
utils.py
Funciones y clases compartidas del proyecto "Sistema de Diagnostico Visual
de Senalizacion Urbana" (G2 - Universidad de Guayaquil).

Se reutiliza en:
  - train_classifier.py (Avances 1 y 2: CNN baseline / MobileNetV2)
  - train_yolo.py (Avance 3: deteccion YOLO)

Autor: Grupo 2 - Inteligencia Artificial
"""

import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image


# ───────────────────────────── Constantes globales ──────────────────────────

CLASES_CLASIFICACION = ['visible_legible', 'deteriorada', 'poco_visible']
CLASES_YOLO = ['senal_transito', 'grafiti_o_pintura', 'oxido_o_corrosion', 'vegetacion_obstruyendo']

IMG_SIZE_CLASIFICACION = 512
IMG_SIZE_YOLO = 640
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def fijar_semilla(seed: int = SEED) -> None:
    """Fija la semilla aleatoria para reproducibilidad en todo el proyecto."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obtener_device() -> torch.device:
    """Devuelve 'cuda' si hay GPU disponible, si no 'cpu'."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ───────────────────────────── Dataset de clasificacion ─────────────────────

class SenalesUrbanasDataset(Dataset):
    """
    Dataset para el modulo de clasificacion general del estado de la senal.
    Espera una lista de tuplas (ruta_imagen, etiqueta_int).
    """

    def __init__(self, datos, transform=None):
        self.datos = datos
        self.transform = transform

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        ruta, etiqueta = self.datos[idx]
        imagen = Image.open(ruta).convert('RGB')
        if self.transform:
            imagen = self.transform(imagen)
        return imagen, torch.tensor(etiqueta, dtype=torch.long)


def cargar_imagenes_por_clase(base_dir, clases, extensiones=('.jpg', '.jpeg', '.png', '.webp')):
    """
    Recorre BASE_DIR/<clase>/ para cada clase y devuelve una lista de
    tuplas (ruta_imagen, indice_clase).
    """
    datos = []
    for idx, clase in enumerate(clases):
        carpeta = os.path.join(base_dir, clase)
        if not os.path.exists(carpeta):
            print(f'AVISO: carpeta no encontrada -> {carpeta}')
            continue
        for archivo in sorted(os.listdir(carpeta)):
            if archivo.lower().endswith(extensiones):
                datos.append((os.path.join(carpeta, archivo), idx))
    return datos


def pesos_por_clase(datos, num_clases):
    """Calcula los pesos de CrossEntropyLoss inversamente proporcionales a la frecuencia de cada clase."""
    conteo = Counter(lbl for _, lbl in datos)
    total = len(datos)
    pesos = [total / (num_clases * max(conteo.get(i, 0), 1)) for i in range(num_clases)]
    return torch.tensor(pesos, dtype=torch.float)


# ───────────────────────────── Transformaciones ──────────────────────────────

def transform_entrenamiento_cnn(img_size: int = IMG_SIZE_CLASIFICACION):
    """Pipeline de augmentation usado por la CNN baseline (Avance 1). Sin flips."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def transform_evaluacion_cnn(img_size: int = IMG_SIZE_CLASIFICACION):
    """Pipeline de evaluacion (sin augmentation) para la CNN baseline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def transform_entrenamiento_mobilenet(img_size: int = 224):
    """Pipeline de augmentation para fine-tuning de MobileNetV2 (Avance 2)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def transform_evaluacion_mobilenet(img_size: int = 224):
    """Pipeline de evaluacion para MobileNetV2."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ───────────────────────────── Arquitectura CNN baseline ────────────────────

class CNN_SenalesUrbanas(nn.Module):
    """
    CNN de 5 bloques convolucionales con skip connection residual (Avance 1).
    Entrada 512x512x3, salida: num_clases logits.
    """

    def __init__(self, num_clases: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.skip_proj = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout_conv = nn.Dropout(p=0.5)
        self.dropout_fc = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_clases)

    def forward(self, x):
        import torch.nn.functional as F

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        skip = x

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        skip_proyectado = self.skip_proj(skip)
        x = F.relu(x + skip_proyectado)

        x = self.adaptive_pool(x)
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def construir_mobilenetv2(num_clases: int = 3):
    """Construye MobileNetV2 preentrenada en ImageNet con un head propio para fine-tuning."""
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_clases),
    )
    return modelo


# ───────────────────────────── Diagnostico combinado (Avance 3) ─────────────

PLANTILLAS_DIAGNOSTICO = {
    'grafiti_o_pintura': {
        'descripcion': 'grafiti o pintura no autorizada sobre la superficie de la senal',
        'recomendacion': 'Programar limpieza o reemplazo de la senal y reportar el grafiti a la unidad de mantenimiento vial.',
    },
    'oxido_o_corrosion': {
        'descripcion': 'oxido o corrosion en el material de la senal',
        'recomendacion': 'Evaluar el reemplazo de la placa o aplicar tratamiento anticorrosivo segun el grado de deterioro.',
    },
    'vegetacion_obstruyendo': {
        'descripcion': 'vegetacion (ramas u hojas) que obstruye parcialmente la senal',
        'recomendacion': 'Coordinar con el area de mantenimiento de parques y jardines la poda de la vegetacion cercana.',
    },
    None: {
        'descripcion': 'sin evidencia especifica adicional detectada por el modelo',
        'recomendacion': 'Mantener monitoreo periodico; no se identifico una causa visual puntual del estado reportado.',
    },
}


def generar_diagnostico_textual(clase_general: str, evidencia: dict | None) -> str:
    """
    Genera el texto de diagnostico combinando la clasificacion general
    con la evidencia detectada por YOLO.

    evidencia: dict con llaves 'clase', 'confianza', 'bbox', o None si no hay evidencia.
    """
    clase_yolo = evidencia['clase'] if evidencia else None
    plantilla = PLANTILLAS_DIAGNOSTICO.get(clase_yolo, PLANTILLAS_DIAGNOSTICO[None])

    lineas = [f'La imagen fue clasificada como: {clase_general}.', 'Evidencia visual detectada:']

    if evidencia:
        lineas.append(f'  - Objeto o condicion: {clase_yolo}')
        lineas.append(f'  - Confianza: {evidencia["confianza"]:.2f}')
        lineas.append(f'  - Ubicacion aproximada: {[round(c) for c in evidencia["bbox"]]}')
    else:
        lineas.append('  - No se detecto un objeto o condicion especifica adicional a la senal.')

    lineas.append('')
    lineas.append('Diagnostico:')
    lineas.append(f'La escena presenta una senal {clase_general} debido a la presencia de '
                   f'{plantilla["descripcion"]}.')
    lineas.append('')
    lineas.append('Recomendacion:')
    lineas.append(plantilla['recomendacion'])

    return '\n'.join(lineas)
