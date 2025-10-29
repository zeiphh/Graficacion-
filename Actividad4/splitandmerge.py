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
