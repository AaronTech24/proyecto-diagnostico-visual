A diferencia de la clasificación general (que responde preguntas abstractas como *"¿bueno, deteriorado o poco visible?"*), YOLO debe detectar **objetos y evidencias visuales concretas y observables** dentro de la imagen. Se definieron 4 clases, coherentes con el tema de señalización vial:

| Clase YOLO | Descripción visual | Justificación | Ejemplo esperado |
|---|---|---|---|
| `senal_transito` | Región rectangular o poligonal que contiene la señal de tránsito completa (placa + soporte visible) | Sirve como ancla espacial: localiza dónde está la señal dentro de la escena antes de buscar evidencias específicas sobre ella | Una señal de PARE, un disco de velocidad máxima, un señal de cruce peatonal |
| `grafiti_o_pintura` | Marca de pintura, aerosol o vandalismo sobre la superficie de la señal | Evidencia directa y visible de vandalismo que reduce la legibilidad de la señal | Señal de PARE con letras "ABVO" pintadas encima |
| `oxido_o_corrosion` | Manchas de óxido, corrosión metálica o deformación del material de la señal | Evidencia física de deterioro por antigüedad o exposición ambiental | Señal triangular con bordes oxidados y placa abollada |
| `vegetacion_obstruyendo` | Ramas, hojas o vegetación que cubre parcial o totalmente la señal | Evidencia concreta de obstrucción que reduce la visibilidad sin que la señal esté dañada físicamente | Señal de PARE semi-cubierta por ramas de un árbol |

**Nota:** se evitó definir clases abstractas como `riesgo_alto` o `senalizacion_deficiente`; cada clase corresponde a un objeto o condición físicamente visible y delimitable con una caja (*bounding box*).
