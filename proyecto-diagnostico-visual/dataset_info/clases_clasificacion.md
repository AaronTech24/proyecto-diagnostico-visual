### 1.2 Clases de clasificación general (Avances 1 y 2)
El modelo de clasificación (CNN baseline y MobileNetV2) trabaja sobre **3 clases generales**, que responden a la pregunta *"¿en qué estado general se encuentra la señal?"*:

| Clase de clasificación | Descripción |
|---|---|
| `visible_legible` | Señal en buen estado, bien iluminada y legible |
| `deteriorada` | Señal con daño físico visible (óxido, grafiti, deformación, pintura borrada) |
| `poco_visible` | Señal legible pero con visibilidad reducida (mala iluminación, ángulo, distancia, obstrucción) |
