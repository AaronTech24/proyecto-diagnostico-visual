"""
train_classifier.py
Entrena el modulo de clasificacion general del estado de la senal de transito
(CNN baseline o MobileNetV2 con transfer learning) y exporta las metricas
finales a metricas_clasificacion.csv.

Uso:
    python train_classifier.py --modelo mobilenet --epocas 30 --data_dir dataset_clasificacion --out_dir resultados

    python train_classifier.py --modelo cnn --epocas 120 --data_dir dataset_clasificacion --out_dir resultados

Requiere la siguiente estructura en --data_dir:
    dataset_clasificacion/
        visible_legible/
        deteriorada/
        poco_visible/
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import (
    CLASES_CLASIFICACION,
    SenalesUrbanasDataset,
    cargar_imagenes_por_clase,
    construir_mobilenetv2,
    CNN_SenalesUrbanas,
    fijar_semilla,
    obtener_device,
    pesos_por_clase,
    transform_entrenamiento_cnn,
    transform_entrenamiento_mobilenet,
    transform_evaluacion_cnn,
    transform_evaluacion_mobilenet,
    IMG_SIZE_CLASIFICACION,
    SEED,
)


def parsear_argumentos():
    parser = argparse.ArgumentParser(description='Entrenamiento del clasificador de senalizacion urbana.')
    parser.add_argument('--modelo', choices=['cnn', 'mobilenet'], default='mobilenet',
                         help='Modelo a entrenar: "cnn" (baseline Avance 1) o "mobilenet" (transfer learning Avance 2).')
    parser.add_argument('--data_dir', default='dataset_clasificacion',
                         help='Carpeta con subcarpetas por clase (visible_legible, deteriorada, poco_visible).')
    parser.add_argument('--out_dir', default='resultados',
                         help='Carpeta donde se guardan el modelo entrenado y las metricas.')
    parser.add_argument('--epocas', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--test_size', type=float, default=0.15)
    return parser.parse_args()


def entrenar_una_epoca(modelo, loader, criterion, optimizer, device):
    modelo.train()
    perdida_total, correctos, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        salidas = modelo(imgs)
        perdida = criterion(salidas, labels)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        optimizer.step()

        perdida_total += perdida.item()
        _, preds = torch.max(salidas, 1)
        correctos += (preds == labels).sum().item()
        total += labels.size(0)
    return perdida_total / len(loader), 100.0 * correctos / total


def evaluar(modelo, loader, criterion, device):
    modelo.eval()
    perdida_total, correctos, total = 0.0, 0, 0
    todas_preds, todas_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            salidas = modelo(imgs)
            perdida = criterion(salidas, labels)
            perdida_total += perdida.item()
            _, preds = torch.max(salidas, 1)
            correctos += (preds == labels).sum().item()
            total += labels.size(0)
            todas_preds.extend(preds.cpu().tolist())
            todas_labels.extend(labels.cpu().tolist())
    acc = 100.0 * correctos / total
    return perdida_total / len(loader), acc, todas_preds, todas_labels


def guardar_metricas_csv(ruta_csv, modelo_nombre, reporte_dict, acc_train, acc_test):
    """Guarda metricas_clasificacion.csv con accuracy global y precision/recall/f1 por clase."""
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    existe = os.path.exists(ruta_csv)

    with open(ruta_csv, 'a', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        if not existe:
            escritor.writerow(['modelo', 'clase', 'precision', 'recall', 'f1_score',
                                'support', 'accuracy_train', 'accuracy_test'])

        for clase in CLASES_CLASIFICACION:
            metricas_clase = reporte_dict.get(clase, {})
            escritor.writerow([
                modelo_nombre,
                clase,
                round(metricas_clase.get('precision', 0), 4),
                round(metricas_clase.get('recall', 0), 4),
                round(metricas_clase.get('f1-score', 0), 4),
                int(metricas_clase.get('support', 0)),
                round(acc_train, 2),
                round(acc_test, 2),
            ])

        # Fila resumen "accuracy" global
        escritor.writerow([
            modelo_nombre, 'GLOBAL_ACCURACY', '', '', '', '',
            round(acc_train, 2), round(acc_test, 2)
        ])

    print(f'Metricas guardadas en: {ruta_csv}')


def main():
    args = parsear_argumentos()
    fijar_semilla(SEED)
    device = obtener_device()
    print(f'Dispositivo: {device} | Modelo seleccionado: {args.modelo}')

    # ── 1. Cargar datos ───────────────────────────────────────────────────
    datos = cargar_imagenes_por_clase(args.data_dir, CLASES_CLASIFICACION)
    if len(datos) == 0:
        raise RuntimeError(f'No se encontraron imagenes en {args.data_dir}. '
                            f'Verifica la estructura de carpetas.')

    labels_todas = [d[1] for d in datos]
    idx_train, idx_test = train_test_split(
        range(len(datos)), test_size=args.test_size, stratify=labels_todas, random_state=SEED
    )
    datos_train = [datos[i] for i in idx_train]
    datos_test = [datos[i] for i in idx_test]
    print(f'Total imagenes: {len(datos)} | Train: {len(datos_train)} | Test: {len(datos_test)}')

    # ── 2. Transforms y modelo segun el tipo elegido ──────────────────────
    if args.modelo == 'cnn':
        t_train = transform_entrenamiento_cnn(IMG_SIZE_CLASIFICACION)
        t_eval = transform_evaluacion_cnn(IMG_SIZE_CLASIFICACION)
        modelo = CNN_SenalesUrbanas(num_clases=len(CLASES_CLASIFICACION)).to(device)
    else:
        t_train = transform_entrenamiento_mobilenet()
        t_eval = transform_evaluacion_mobilenet()
        modelo = construir_mobilenetv2(num_clases=len(CLASES_CLASIFICACION)).to(device)

    ds_train = SenalesUrbanasDataset(datos_train, transform=t_train)
    ds_test = SenalesUrbanasDataset(datos_test, transform=t_eval)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    # ── 3. Loss, optimizer ────────────────────────────────────────────────
    pesos = pesos_por_clase(datos, len(CLASES_CLASIFICACION)).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos)
    optimizer = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # ── 4. Entrenamiento ──────────────────────────────────────────────────
    mejor_acc_train = 0.0
    for epoca in range(args.epocas):
        t0 = time.time()
        loss_train, acc_train = entrenar_una_epoca(modelo, loader_train, criterion, optimizer, device)
        loss_test, acc_test, _, _ = evaluar(modelo, loader_test, criterion, device)
        scheduler.step(loss_test)
        mejor_acc_train = max(mejor_acc_train, acc_train)

        if (epoca + 1) % 5 == 0 or epoca == 0:
            print(f'Epoca {epoca + 1:3d}/{args.epocas} | '
                  f'Train Loss {loss_train:.4f} Acc {acc_train:.1f}% | '
                  f'Test Loss {loss_test:.4f} Acc {acc_test:.1f}% | '
                  f'{time.time() - t0:.1f}s')

    # ── 5. Evaluacion final + guardar modelo y metricas ───────────────────
    _, acc_test_final, preds_test, labels_test = evaluar(modelo, loader_test, criterion, device)
    _, acc_train_final, _, _ = evaluar(modelo, loader_train, criterion, device)

    reporte = classification_report(
        labels_test, preds_test, target_names=CLASES_CLASIFICACION,
        labels=range(len(CLASES_CLASIFICACION)), zero_division=0, output_dict=True
    )
    print('\nReporte de clasificacion (test):')
    print(classification_report(labels_test, preds_test, target_names=CLASES_CLASIFICACION, zero_division=0))
    print('Matriz de confusion (test):')
    print(confusion_matrix(labels_test, preds_test))

    os.makedirs(args.out_dir, exist_ok=True)
    ruta_modelo = os.path.join(args.out_dir, f'{args.modelo}_clasificador.pth')
    torch.save(modelo.state_dict(), ruta_modelo)
    print(f'\nModelo guardado en: {ruta_modelo}')

    ruta_csv = os.path.join(args.out_dir, 'metricas_clasificacion.csv')
    guardar_metricas_csv(ruta_csv, args.modelo, reporte, acc_train_final, acc_test_final)


if __name__ == '__main__':
    main()
