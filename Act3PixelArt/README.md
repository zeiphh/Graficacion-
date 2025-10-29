# Actividad 3 — Pixel Art del Creeper (OpenCV)

Este programa genera una imagen tipo **pixel art** del rostro del Creeper usando escala de grises.  
Se utiliza **OpenCV**, **Matplotlib** y **NumPy** para crear, escalar, mostrar y guardar la figura.

## Descripción
1. Se crea una matriz de 40x40 pixeles con fondo gris claro.
2. Se dibujan las zonas negras correspondientes a los ojos y la boca del Creeper.
3. La imagen se escala para apreciarse mejor con `cv2.resize()` usando interpolación `INTER_NEAREST` (mantiene los píxeles cuadrados).
4. Se muestra en ventana y se guarda automáticamente en una carpeta de salida.

## Código
```python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def crear_creeper_pixel_art():
    alto, ancho = 40, 40
    img = np.full((alto, ancho), 180, dtype=np.uint8)
    NEGRO = 0
    for y in range(8, 16):
        for x in range(8, 16):
            img[y, x] = NEGRO
        for x in range(24, 32):
            img[y, x] = NEGRO
    for y in range(16, 28):
        for x in range(16, 24):
            img[y, x] = NEGRO
    for y in range(24, 32):
        for x in range(8, 16):
            img[y, x] = NEGRO
        for x in range(24, 32):
            img[y, x] = NEGRO
    return img

def mostrar_creeper(imagen, factor=12):
    alto, ancho = imagen.shape
    escalada = cv.resize(imagen, (ancho * factor, alto * factor), interpolation=cv.INTER_NEAREST)
    cv.imshow('Creeper Pixel Art', escalada)
    plt.figure(figsize=(8, 8))
    plt.imshow(imagen, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()
    return escalada

def analizar_imagen(imagen):
    print(f"Dimensiones: {imagen.shape[1]}x{imagen.shape[0]} px")
    print(f"Tamaño: {imagen.nbytes} bytes")
    print(f"Valores únicos: {np.unique(imagen).tolist()}")

def guardar_imagen(imagen, nombre='creeper_pixel_art.png', escala=14):
    alto, ancho = imagen.shape
    hd = cv.resize(imagen, (ancho * escala, alto * escala), interpolation=cv.INTER_NEAREST)
    cv.imwrite(nombre, hd)

def main():
    carpeta_out = os.path.join(os.path.dirname(__file__), "salidas_act3")
    os.makedirs(carpeta_out, exist_ok=True)
    creeper = crear_creeper_pixel_art()
    analizar_imagen(creeper)
    ruta_png = os.path.join(carpeta_out, "creeper_pixel_art_14x.png")
    guardar_imagen(creeper, nombre=ruta_png, escala=14)
    creeper_borde = cv.copyMakeBorder(creeper, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    mostrar_creeper(creeper_borde, factor=12)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

