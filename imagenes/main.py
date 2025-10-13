import cv2 as cv
import numpy as np
import math

def load_rgba(path):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b,g,r = cv.split(img)
        a = np.full_like(b, 255)
        img = cv.merge([b,g,r,a])
    return img

def rotate_bound_bilinear(img, angle_deg):
    h, w = img.shape[:2]
    angle = math.radians(angle_deg)
    cos = abs(math.cos(angle))
    sin = abs(math.sin(angle))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv.warpAffine(img, M, (new_w, new_h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def translate_to_center_canvas(img, canvas_scale=2.0):
    h, w = img.shape[:2]
    new_w = int(w * canvas_scale)
    new_h = int(h * canvas_scale)
    dx = (new_w - w) // 2
    dy = (new_h - h) // 2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv.warpAffine(img, M, (new_w, new_h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

src_path = r"C:\Users\nothi\Downloads\gastly.png"
img0 = load_rgba(src_path)

img1_scale2 = cv.resize(img0, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
img1 = rotate_bound_bilinear(img1_scale2, 45)
cv.imshow("1 Escalar2 Rotar45 Bilineal", img1)
cv.waitKey(0)

img2a = cv.resize(img0, None, fx=2.0, fy=2.0, interpolation=cv.INTER_NEAREST)
img2b = rotate_bound_bilinear(img2a, 45)
H, W = img2b.shape[:2]
I = np.float32([[1,0,0],[0,1,0]])
img2 = cv.warpAffine(img2b, I, (W, H), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
cv.imshow("2 Escalar2 Rotar45 Bilineal", img2)
cv.waitKey(0)

img3a = translate_to_center_canvas(img0, canvas_scale=2.0)
img3b = rotate_bound_bilinear(img3a, 90)
img3 = cv.resize(img3b, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
cv.imshow("3 TrasladarCentro Rotar90 Escalar2 Bilineal", img3)
cv.waitKey(0)

cv.destroyAllWindows()
