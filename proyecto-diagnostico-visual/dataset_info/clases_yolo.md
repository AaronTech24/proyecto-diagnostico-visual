A diferencia de la clasificación general (que responde preguntas abstractas como *"¿bueno, deteriorado o poco visible?"*), YOLO debe detectar **objetos y evidencias visuales concretas y observables** dentro de la imagen. Se definieron 4 clases, coherentes con el tema de señalización vial:

| Clase YOLO | Instancias (aprox.) | Observación |
|---|---|---|
| `senal_vertical` | 241 | Una o más instancias por imagen (señal siempre presente) |
| `senal_deteriorada` | 80 | Presente en imágenes de la clase `deteriorada` |
| `senal_obstruida` | 30 | Presente en algunas imágenes de `poco_visible` con obstrucción no vegetal |
| `semaforo` | 25 | Presente en imágenes que incluyan semáforos en el encuadre |
| `paso_cebra_visible` | 40 | Imágenes de cruce peatonal en buen estado |
| `paso_cebra_desgastado` | 30 | Imágenes de cruce peatonal deteriorado |
| `vegetacion_obstruyendo` | 35 | Imágenes de `poco_visible` con obstrucción vegetal |
