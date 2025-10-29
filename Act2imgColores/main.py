import cv2
import os


ruta_imagen = r"C:\Users\nothi\Desktop\char.png"


carpeta_salida = os.path.join(os.path.dirname(__file__), "salidas")
os.makedirs(carpeta_salida, exist_ok=True)

imagen_color = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

if imagen_color is None:
    print("No se pudo cargar la imagen.")
    exit()


imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)


alto, ancho = imagen_gris.shape
print(f"Tamaño (alto x ancho): {alto} x {ancho}")


negativo = 255 - imagen_gris


imagen_rgb = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB)
imagen_bgra = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2BGRA)

cv2.imwrite(os.path.join(carpeta_salida, "gastly_original.png"), imagen_color)
cv2.imwrite(os.path.join(carpeta_salida, "gastly_gris_negativo.png"), negativo)
cv2.imwrite(os.path.join(carpeta_salida, "gastly_rgb.png"), imagen_rgb)
cv2.imwrite(os.path.join(carpeta_salida, "gastly_bgra.png"), imagen_bgra)

print("Imágenes generadas correctamente en:", carpeta_salida)
