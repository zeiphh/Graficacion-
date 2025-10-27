# Actividad 1 — Carga y visualización de imagen en escala de grises

Este programa utiliza **OpenCV** para cargar y mostrar una imagen en escala de grises.  
La imagen utilizada en este caso es `gastly.png`.

## Descripción
1. Se importa la librería `cv2`.
2. Se carga la imagen en modo **grises** con `cv2.IMREAD_GRAYSCALE`.
3. Se verifica que la imagen haya sido cargada correctamente.
4. Si la carga es exitosa, se muestra en una ventana emergente usando `cv2.imshow()`.
5. El programa espera una tecla (`cv2.waitKey(0)`) y cierra la ventana con `cv2.destroyAllWindows()`.

## Código
```python
import cv2

img_gray = cv2.imread(r"C:\Users\nothi\Downloads\gastly.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    print("no se pudo cargar la imagen.")
else:
    cv2.imshow("Imagen original (gris)", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
