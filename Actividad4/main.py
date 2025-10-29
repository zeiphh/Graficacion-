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
