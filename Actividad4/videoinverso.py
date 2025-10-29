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
