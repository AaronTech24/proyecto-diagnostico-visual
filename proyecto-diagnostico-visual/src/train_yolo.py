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

# ── Validacion del modelo entrenado sobre el conjunto de test ─────────────

modelo_yolo_final = YOLO(RUTA_MEJOR_YOLO) if os.path.exists(RUTA_MEJOR_YOLO) else modelo_yolo

metricas_val = modelo_yolo_final.val(
    data=ruta_data_yaml,
    split='test',
    imgsz=640,
    batch=8,
    project=RUTA_RESULTADOS_YOLO,
    name='evaluacion_test'
)

precision_yolo = float(metricas_val.box.mp)        # precision promedio (mean precision)
recall_yolo    = float(metricas_val.box.mr)        # recall promedio (mean recall)
map50_yolo     = float(metricas_val.box.map50)     # mAP @ IoU 0.50
map5095_yolo   = float(metricas_val.box.map)       # mAP @ IoU 0.50:0.95

print('=' * 60)
print('  METRICAS DE DETECCION — Conjunto de PRUEBA')
print('=' * 60)
print(f'  Precision   : {precision_yolo:.3f}')
print(f'  Recall      : {recall_yolo:.3f}')
print(f'  mAP50       : {map50_yolo:.3f}')
print(f'  mAP50-95    : {map5095_yolo:.3f}')
print('-' * 60)

# Metricas por clase
nombres_clases = metricas_val.names
print('\nMetricas por clase (AP50):')
for i, ap in enumerate(metricas_val.box.ap50):
    print(f'  {nombres_clases[i]:<22}: AP50 = {ap:.3f}')
