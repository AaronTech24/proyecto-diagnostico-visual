# ── Entrenamiento del modelo YOLOv8n ──────────────────────────────────────

modelo_yolo = YOLO('yolov8n.pt')  # pesos preentrenados en COCO

resultados_entrenamiento = modelo_yolo.train(
    data=ruta_data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    seed=SEED,
    project=RUTA_RESULTADOS_YOLO,
    name='yolo_senales_v1',
    patience=15,           # early stopping si no mejora en 15 epocas
    verbose=True
)

print('Entrenamiento finalizado.')
print(f'Resultados guardados en: {os.path.join(RUTA_RESULTADOS_YOLO, "yolo_senales_v1")}')
