# Escalado de Cuadrado con Detección de Manos

En este proyecto utilizamos **MediaPipe** y **OpenCV** para detectar ambas manos
en tiempo real y medir la distancia entre los dedos **índice izquierdo** e **índice derecho**.

Con esa distancia, se dibuja un cuadro verde que cambia de tamaño:
- Entre más cerca estén los dedos el cuadrado se hace pequeño.
- Entre más lejos estén el cuadrado crece.

---

## Como funciona el código
Primero que nada eliminé en el codigo proporcionado, la parte para detectar las letras con las manos
y todo lo relacionado ya que no haremos uso de ello en este proyecto.
1. Se inicializa **MediaPipe Hands** con detección de mis 2 manos (`max_num_hands=2`).
2. Se obtienen las landmarks de los puntos 8 (que son los índices) de cada mano.
3. Se calcula la distancia euclidiana entre ellos.
4. Esa distancia se usa para escalar el tamaño de un cuadrado en la pantalla.
5. El cuadrado se dibuja en medio de ambas manos.

---

## Código principal

```python
import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=2)


cap = cv2.VideoCapture(0)


tamano_base = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    indice_derecho = None
    indice_izquierdo = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            indice = hand_landmarks.landmark[8]
            x, y = int(indice.x * w), int(indice.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            label = handedness.classification[0].label
            cv2.putText(frame, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if label == "Right":
                indice_derecho = (x, y)
            else:
                indice_izquierdo = (x, y)

        if indice_derecho and indice_izquierdo:
            distancia = np.linalg.norm(np.array(indice_derecho) - np.array(indice_izquierdo))


            cv2.line(frame, indice_derecho, indice_izquierdo, (255, 0, 0), 3)



            tamano = int(np.interp(distancia, [50, 400], [50, 400]))


            centro_x = int((indice_derecho[0] + indice_izquierdo[0]) / 2)
            centro_y = int((indice_derecho[1] + indice_izquierdo[1]) / 2)


            x1 = centro_x - tamano // 2
            y1 = centro_y - tamano // 2
            x2 = centro_x + tamano // 2
            y2 = centro_y + tamano // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)



    cv2.imshow("escalar cuadro con las manos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

