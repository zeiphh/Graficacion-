import cv2 as cv
import numpy as np

def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

def ema(prev, cur, alpha=0.2):
    return cur if prev is None else (1 - alpha) * prev + alpha * cur

def eye_roi_bounds(xc, yc, w_face, h_face, img_w, img_h):
    roi_w = int(w_face * 0.28)
    roi_h = int(h_face * 0.18)
    x1 = clamp(xc - roi_w // 2, 0, img_w - 1)
    y1 = clamp(yc - roi_h // 2, 0, img_h - 1)
    x2 = clamp(x1 + roi_w, 0, img_w - 1)
    y2 = clamp(y1 + roi_h, 0, img_h - 1)
    return x1, y1, x2, y2

def pupil_from_darkest(gray_roi):
    g = cv.GaussianBlur(gray_roi, (7, 7), 0)
    minVal, _, minLoc, _ = cv.minMaxLoc(g)
    mean, std = cv.meanStdDev(g)
    thr = float(mean - 0.7 * std)
    _, bw = cv.threshold(g, thr, 255, cv.THRESH_BINARY_INV)
    h, w = bw.shape
    x0 = clamp(minLoc[0] - 12, 0, w - 1)
    y0 = clamp(minLoc[1] - 12, 0, h - 1)
    x1 = clamp(minLoc[0] + 12, 0, w - 1)
    y1 = clamp(minLoc[1] + 12, 0, h - 1)
    sub = bw[y0:y1, x0:x1]
    cnts, _ = cv.findContours(sub, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv.contourArea)
        M = cv.moments(c)
        if M['m00'] > 1e-3:
            cx = int(M['m10'] / M['m00']) + x0
            cy = int(M['m01'] / M['m00']) + y0
            area = cv.contourArea(c)
            r = int(max(2, min(12, np.sqrt(area / np.pi))))
            return (cx, cy), r
    return minLoc, 6

def openness_score(gray_roi, use_laplacian=True):
    if use_laplacian:
        lap = cv.Laplacian(gray_roi, cv.CV_64F, ksize=3)
        v = float(lap.var())
    else:
        v = float(gray_roi.var())
    return v

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)

open_ref_L = None
open_ref_R = None
open_min = 25.0
alpha_ref = 0.08
pupil_smooth_L = None
pupil_smooth_R = None
CLOSED_THRESHOLD = 0.12

while True:
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, fw, fh) in rostros:
        cv.rectangle(img, (x, y), (x + fw, y + fh), (255, 0, 255), 3)
        left_center = (x + int(fw * 0.3), y + int(fh * 0.4))
        right_center = (x + int(fw * 0.7), y + int(fh * 0.4))

        lx1, ly1, lx2, ly2 = eye_roi_bounds(left_center[0], left_center[1], fw, fh, w, h)
        rx1, ry1, rx2, ry2 = eye_roi_bounds(right_center[0], right_center[1], fw, fh, w, h)

        Lroi = gray[ly1:ly2, lx1:lx2]
        Rroi = gray[ry1:ry2, rx1:rx2]

        vL = openness_score(Lroi, use_laplacian=True)
        vR = openness_score(Rroi, use_laplacian=True)

        open_ref_L = vL if open_ref_L is None else max((1 - alpha_ref) * open_ref_L + alpha_ref * vL, open_ref_L)
        open_ref_R = vR if open_ref_R is None else max((1 - alpha_ref) * open_ref_R + alpha_ref * vR, open_ref_R)

        def norm_open(v, v_open_ref):
            v_max = max(v_open_ref, open_min + 10.0)
            f = (v - open_min) / (v_max - open_min + 1e-6)
            return float(np.clip(f, 0.0, 1.0))

        openL = norm_open(vL, open_ref_L)
        openR = norm_open(vR, open_ref_R)

        (plx, ply), rL = pupil_from_darkest(Lroi)
        (prx, pry), rR = pupil_from_darkest(Rroi)

        pupL = (lx1 + plx, ly1 + ply)
        pupR = (rx1 + prx, ry1 + pry)

        def smooth(prev, cur):
            if prev is None:
                return cur
            return (int(0.7 * prev[0] + 0.3 * cur[0]), int(0.7 * prev[1] + 0.3 * cur[1]))

        pupil_smooth_L = smooth(pupil_smooth_L, pupL)
        pupil_smooth_R = smooth(pupil_smooth_R, pupR)

        oreja_y = y + int(fh * 0.60)
        radio_oreja = int(fw * 0.12)
        oreja_izq_x = x - int(fw * 0.05)
        oreja_der_x = x + int(fw * 1.05)
        cv.circle(img, (oreja_izq_x, oreja_y), radio_oreja, (255, 255, 0), -1)
        cv.circle(img, (oreja_der_x, oreja_y), radio_oreja, (255, 255, 0), -1)

        nariz_x = x + int(fw * 0.5)
        nariz_y = y + int(fh * 0.65)
        radio_nariz = int(fw * 0.08)
        cv.circle(img, (nariz_x, nariz_y), radio_nariz, (128, 0, 128), -1)

        boca_x = x + int(fw * 0.5)
        boca_y = y + int(fh * 0.85)
        ancho = int(fw * 0.4)
        alto = int(fh * 0.1)
        x1 = boca_x - ancho // 2; y1 = boca_y - alto // 2
        x2 = boca_x + ancho // 2; y2 = boca_y + alto // 2
        cv.rectangle(img, (x1, y1), (x2, y2), (255, 182, 193), -1)

        radio_ojo_L = int(fw * 0.07)
        radio_ojo_R = int(fw * 0.07)

        def draw_eye(center, r_base, open_factor, pupil_pt):
            rx = r_base
            ry = max(2, int(r_base * (0.55 * open_factor + 0.15)))
            r_pup = max(3, int(r_base * 0.22))
            if open_factor < CLOSED_THRESHOLD:
                thickness = max(2, int(r_base * 0.22))
                cv.line(img, (center[0] - rx, center[1]), (center[0] + rx, center[1]), (60, 60, 80), thickness)
                cv.line(img, (center[0] - rx, center[1] - thickness//2), (center[0] + rx, center[1] - thickness//2), (100, 100, 120), 1)
                cv.line(img, (center[0] - rx, center[1] + thickness//2), (center[0] + rx, center[1] + thickness//2), (100, 100, 120), 1)
                return
            cv.ellipse(img, center, (rx, ry), 0, 0, 360, (230, 230, 250), -1)
            cv.ellipse(img, center, (rx, ry), 0, 0, 360, (100, 100, 120), 1)
            max_off = max(1, min(rx, ry) - r_pup - 2)
            v = np.array(pupil_pt) - np.array(center)
            norm = np.linalg.norm(v)
            if norm > max_off and norm > 0:
                v = v * (max_off / norm)
            p = (int(center[0] + v[0]), int(center[1] + v[1]))
            cv.circle(img, p, r_pup, (128, 0, 128), -1)

        draw_eye(left_center, radio_ojo_L, openL, pupil_smooth_L)
        draw_eye(right_center, radio_ojo_R, openR, pupil_smooth_R)

    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
