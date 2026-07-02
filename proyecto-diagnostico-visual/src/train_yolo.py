# ── Entrenamiento YOLOv8n ─────────────────────────────────────────────────
modelo_yolo = YOLO('yolov8n.pt')  # pesos preentrenados en COCO

resultados_entrenamiento = modelo_yolo.train(
    data=ruta_data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    seed=SEED,
    project=RUTA_RESULTADOS,
    name='yolo_senales_v1',
    patience=15,
    verbose=True
)

print('Entrenamiento finalizado.')
print(f'Resultados en: {os.path.join(RUTA_RESULTADOS, "yolo_senales_v1")}')

# ── Copiar el mejor modelo a la carpeta raiz del proyecto ─────────────────
RUTA_MEJOR_YOLO = os.path.join(BASE_DIR, 'yolo_senales_best.pt')
ruta_best = os.path.join(RUTA_RESULTADOS, 'yolo_senales_v1', 'weights', 'best.pt')

if os.path.exists(ruta_best):
    shutil.copy(ruta_best, RUTA_MEJOR_YOLO)
    print(f'Mejor modelo copiado a: {RUTA_MEJOR_YOLO}')
else:
    print('AVISO: best.pt aun no existe. Ejecuta primero la celda de entrenamiento.')

# ── Validacion sobre el conjunto de test ──────────────────────────────────

# Function to convert a segmentation label line to a bounding box label line
def seg_to_bbox_line(line):
    parts = line.strip().split()
    if not parts:
        return ""
    cid = int(parts[0])
    coords = list(map(float, parts[1:]))

    # A YOLO bbox line has 5 parts (class_id x_c y_c w h). A polygon has more (class_id x1 y1 x2 y2 ...).
    # If the line already has 5 parts, assume it's already bbox format. Else, it's polygon.
    if len(parts) == 5: # Already bbox format
        return line.strip()

    # Convert polygon to bounding box (min/max x, y) in YOLO format (x_center, y_center, width, height)
    xs = [coords[i] for i in range(0, len(coords), 2)]
    ys = [coords[i+1] for i in range(0, len(coords), 2)]

    # Ensure we have enough coordinates for a valid polygon (at least 3 points, so 6 coordinates)
    if len(xs) < 3 or len(ys) < 3: 
        return "" # Skip invalid polygons

    min_x_norm = min(xs)
    max_x_norm = max(xs)
    min_y_norm = min(ys)
    max_y_norm = max(ys)

    x_center = (min_x_norm + max_x_norm) / 2
    y_center = (min_y_norm + max_y_norm) / 2
    width = max_x_norm - min_x_norm
    height = max_y_norm - min_y_norm

    # Return in YOLO bounding box format: class_id x_center y_center width height
    return f"{cid} {x_center} {y_center} {width} {height}"

# Pre-process test labels to ensure they are bounding box format
# This is necessary because the dataset may contain segmentation labels,
# which cause issues with the val() function's metric calculation (IndexError).
# The ideal fix is to re-export the dataset from Roboflow as YOLOv8 Detection format.
print("Verificando y convirtiendo etiquetas de segmentacion a bounding box en el conjunto de test...")
test_label_dir = os.path.join(YOLO_DIR, 'labels', 'test')
if os.path.exists(test_label_dir):
    for filename in os.listdir(test_label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(test_label_dir, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                converted_line = seg_to_bbox_line(line)
                if converted_line:
                    new_lines.append(converted_line)
            
            # Overwrite the original file with converted bounding box lines
            with open(filepath, 'w') as f:
                f.write('\n'.join(new_lines))
    print("Conversion de etiquetas de test completada.")
else:
    print(f"AVISO: No se encontro el directorio de etiquetas de test: {test_label_dir}")

modelo_yolo_final = YOLO(RUTA_MEJOR_YOLO) if os.path.exists(RUTA_MEJOR_YOLO) else modelo_yolo

metricas = modelo_yolo_final.val(
    data=ruta_data_yaml,
    split='test',
    imgsz=640,
    batch=8,
    project=RUTA_RESULTADOS,
    name='evaluacion_test',
    task='detect',
    plots=False, # Disable plots to avoid IndexError with confusion matrix
    verbose=False # Disable verbose output to avoid KeyError in print_results
)

precision = float(metricas.box.mp)
recall    = float(metricas.box.mr)
map50     = float(metricas.box.map50)
map5095   = float(metricas.box.map)

print('=' * 60)
print('  METRICAS DE DETECCION — Conjunto de PRUEBA')
print('=' * 60)
print(f'  Precision  (mean) : {precision:.3f}')
print(f'  Recall     (mean) : {recall:.3f}')
print(f'  mAP50             : {map50:.3f}')
print(f'  mAP50-95          : {map5095:.3f}')
print('-' * 60)
print('\nAP50 por clase:')
for i, ap in enumerate(metricas.box.ap50):
    print(f'  {CLASES_YOLO[i]:<28}: {ap:.3f}')

# ── Inferencia sobre imágenes del conjunto de test ───────────────────────
carpeta_test = os.path.join(YOLO_DIR, 'images', 'test')

if os.path.exists(carpeta_test) and len(os.listdir(carpeta_test)) > 0:
    imagenes_test = sorted(os.listdir(carpeta_test))[:6]
    predicciones = modelo_yolo_final.predict(
        [os.path.join(carpeta_test, img) for img in imagenes_test],
        imgsz=640, conf=0.05, save=False, verbose=False # Lowered confidence threshold to 0.05
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for ax, pred, nombre in zip(axes, predicciones, imagenes_test):
        # Get the original image and convert to RGB
        img_ann = cv2.cvtColor(pred.orig_img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_ann.shape

        detected_class_names = []
        # Iterate through detected boxes and draw them manually with class names
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            if class_id < len(CLASES_YOLO):
                label = f'{CLASES_YOLO[class_id]} {confidence:.2f}'
                detected_class_names.append(CLASES_YOLO[class_id])
            else:
                label = f'Class {class_id} {confidence:.2f}'
                detected_class_names.append(f'Class {class_id}')

            color = COLORES_YOLO[class_id % len(COLORES_YOLO)]

            cv2.rectangle(img_ann, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_ann, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        ax.imshow(img_ann)

        # Update: Set title to include detected classes
        unique_detected_class_names = sorted(list(set(detected_class_names)))
        if unique_detected_class_names:
            title_detections = f"Detecciones: {', '.join(unique_detected_class_names)}"
        else:
            title_detections = "No hay detecciones"
        ax.set_title(f"{nombre}\n({title_detections})", fontsize=8) # Display filename and detections
        ax.axis('off')
    fig.suptitle('Detecciones YOLO sobre el conjunto de prueba (7 clases)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_RESULTADOS, 'ejemplos_detecciones.png'), dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('AVISO: no hay imagenes en test/ todavia. Ejecuta el split primero.')
