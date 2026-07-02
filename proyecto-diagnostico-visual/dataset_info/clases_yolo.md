A diferencia de la clasificación general, YOLO detecta **objetos y evidencias visuales concretas y observables** dentro de la imagen. Se definieron **7 clases**, todas visualmente delimitables con una bounding box:

| # | Clase YOLO | Descripción visual | Justificación | Ejemplo esperado |
|---|---|---|---|---|
| 0 | `senal_vertical` | Señal de tránsito vertical sobre poste (placa + soporte) | Objeto principal del proyecto; sirve de ancla espacial para las demás detecciones | Señal de PARE, señal de velocidad máxima, señal de cruce peatonal |
| 1 | `senal_deteriorada` | Señal con daño físico visible: grafiti, óxido, abolladuras, pintura borrada | Evidencia directa de deterioro que reduce la legibilidad y la efectividad de la señal | Señal de PARE con grafiti; señal triangular oxidada y doblada |
| 2 | `senal_obstruida` | Señal tapada parcial o totalmente por un objeto, publicidad, cable o elemento externo distinto de vegetación | Evidencia de obstrucción no vegetal que impide leer la señal | Señal tapada por una lona publicitaria o por un poste superpuesto |
| 3 | `semaforo` | Semáforo vehicular o peatonal (caja con luces sobre soporte) | Elemento de señalización urbana distinto de las señales verticales; puede estar deteriorado u obstruido también | Semáforo en intersección, semáforo peatonal de cruce |
| 4 | `paso_cebra_visible` | Marcas horizontales blancas del paso peatonal, nítidas y legibles | Evidencia de señalización horizontal en buen estado | Paso de cebra con líneas claras sobre asfalto |
| 5 | `paso_cebra_desgastado` | Marcas del paso peatonal borrosas, casi imperceptibles o muy desgastadas | Evidencia de señalización horizontal deteriorada que no es claramente visible para peatones y conductores | Cruce peatonal con líneas casi invisibles |
| 6 | `vegetacion_obstruyendo` | Ramas, hojas o vegetación que cubre parcialmente una señal vertical o un semáforo | Evidencia concreta de obstrucción vegetal sin daño físico en la señal misma | Señal de PARE semi-cubierta por ramas de un árbol |
