import cv2

img_gray = cv2.imread(r"C:\Users\nothi\Downloads\gastly.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    print("no se pudo cargar la imagen.")
else:
    cv2.imshow("Imagen original (gris)", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
