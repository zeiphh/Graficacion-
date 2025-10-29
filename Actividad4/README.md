# Actividad 3 — Operaciones de imagen y video (OpenCV)

Esta actividad incluye tres programas que aplican distintos conceptos de **procesamiento digital de imágenes** y **captura/edición de video** utilizando la biblioteca **OpenCV**.

---

## 3.1 Operación puntual — Umbral binario
**Archivo:** `act3_op_puntual.py`  
**Descripción:** Carga `colores.jpg` en escala de grises y aplica una operación puntual de umbral binario (TH = 150) para convertir la imagen a blanco y negro.

**Salidas:**
- `_output/act3_op_puntual_entrada.png`
- `_output/act3_op_puntual_binaria.png`

---

## 3.2 Split / Merge de canales
**Archivo:** `act3_split_merge.py`  
**Descripción:** Carga la imagen `colores.jpg` a color, separa sus canales **B**, **G**, **R**, crea imágenes con cada canal aislado y una versión con los canales reordenados (**B R G**).

**Salidas:**
- `_output/act3_split_merge_original.png`
- `_output/act3_split_merge_r_only.png`
- `_output/act3_split_merge_g_only.png`
- `_output/act3_split_merge_b_only.png`
- `_output/act3_split_merge_brg.png`

---

## 3.3 Captura y edición de video
**Archivo:** `act3_video_capture.py`  
**Descripción:** Captura video desde la cámara (índice 0), guarda el video original y una versión procesada con inversión de color (negativo) y reordenamiento de canales.  
Muestra ambas vistas en tiempo real y se detiene al presionar **ESC**.

**Salidas:**
- `_output/video_original.mp4`
- `_output/video_procesado.mp4`

---

## Requisitos
- Python 3  
- OpenCV  `pip install opencv-python`  
- NumPy  `pip install numpy`

> Coloca `colores.jpg` en el mismo directorio donde se ejecutan los scripts o ajusta la ruta en el código según sea necesario.

## Código 1
```python
import cv2 as cv
import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv.imread(r"C:\Users\nothi\Desktop\colores.jpg", cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("No se encontró la imagen. Verifica la ruta del archivo.")

h, w = img.shape
print("Dimensiones (alto, ancho):", (h, w))

TH = 150
# Versión vectorizada (más rápida y clara):
_, binaria = cv.threshold(img, TH, 255, cv.THRESH_BINARY)

cv.imwrite(os.path.join(OUTPUT_DIR, "act3_op_puntual_entrada.png"), img)
cv.imwrite(os.path.join(OUTPUT_DIR, "act3_op_puntual_binaria.png"), binaria)

print("Listo. Archivos generados en:", OUTPUT_DIR)
```
## Código 2
```python
import cv2 as cv
import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv.imread(r"C:\Users\nothi\Desktop\colores.jpg", cv.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("No se encontró la imagen. Verifica la ruta del archivo.")

h, w = img.shape[:2]
print("Dimensiones (alto, ancho):", (h, w))

b, g, r = cv.split(img)
zeros = np.zeros((h, w), dtype=np.uint8)

r_only = cv.merge([zeros, zeros, r])
g_only = cv.merge([zeros, g, zeros])
b_only = cv.merge([b, zeros, zeros])

img_brg = cv.merge([b, r, g])

cv.imwrite(os.path.join(OUTPUT_DIR, "act3_split_merge_original.png"), img)
cv.imwrite(os.path.join(OUTPUT_DIR, "act3_split_merge_r_only.png"), r_only)
cv.imwrite(os.path.join(OUTPUT_DIR, "act3_split_merge_g_only.png"), g_only)
cv.imwrite(os.path.join(OUTPUT_DIR, "act3_split_merge_b_only.png"), b_only)
cv.imwrite(os.path.join(OUTPUT_DIR, "act3_split_merge_brg.png"), img_brg)

print("Listo. Archivos generados en:", OUTPUT_DIR)
```
## Código 3
```python
import cv2 as cv
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cam = cv.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("No se pudo abrir la cámara. Revisa permisos y conexión.")

FPS = 20.0
ancho = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
alto = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (ancho, alto)

fourcc = cv.VideoWriter_fourcc(*"mp4v")
salida_original = cv.VideoWriter(os.path.join(OUTPUT_DIR, "video_original.mp4"), fourcc, FPS, size)
salida_procesado = cv.VideoWriter(os.path.join(OUTPUT_DIR, "video_procesado.mp4"), fourcc, FPS, size)

print("Grabando... Presiona ESC para terminar.")

while True:
    ok, frame = cam.read()
    if not ok:
        print("Se detuvo la captura de la cámara.")
        break

    salida_original.write(frame)

    negativo = 255 - frame
    b, g, r = cv.split(negativo)
    procesado = cv.merge([b, r, g])

    salida_procesado.write(procesado)

    cv.imshow("Original", frame)
    cv.imshow("Procesado", procesado)

    if cv.waitKey(1) & 0xFF == 27:
        print("Captura finalizada.")
        break

cam.release()
salida_original.release()
salida_procesado.release()
cv.destroyAllWindows()

print("Videos guardados en:", OUTPUT_DIR)
