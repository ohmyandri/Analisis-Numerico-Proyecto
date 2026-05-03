import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img_path = r"assets/bubbleGuppie.jpeg"
output_xlsx = r"assets/bubble_guppies_coordinate_mapping.xlsx"

y_up =   True  # True y grows upward; False -> image coordinates (y down)

def pick_contour(contours, shape):
    height, width = shape

    def touches_border(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1

    inner = [c for c in contours if not touches_border(c)]
    candidates = inner if inner else contours
    if not candidates:
        return None
    return max(candidates, key=cv2.contourArea)


def build_mask(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    border = np.vstack(
        [
            blur[:10, :, :].reshape(-1, 3),
            blur[-10:, :, :].reshape(-1, 3),
            blur[:, :10, :].reshape(-1, 3),
            blur[:, -10:, :].reshape(-1, 3),
        ]
    )
    bg_color = np.median(border, axis=0)

    diff = np.linalg.norm(blur.astype(np.float32) - bg_color, axis=2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned


def smooth_contour(points, window):
    if window <= 1:
        return points
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.vstack([points[-pad:], points, points[:pad]])
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.column_stack([x, y])


image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"No se encontro la imagen: {img_path}")

mask = build_mask(image)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = pick_contour(contours, mask.shape)
if contour is None:
    raise ValueError("No se encontro un contorno valido.")

contour_xy = contour[:, 0, :].astype(float)
contour_xy = smooth_contour(contour_xy, window=2)

contour_xy_for_plot = contour_xy.copy()

if y_up:
    height = image.shape[0]
    contour_xy = contour_xy.copy()
    contour_xy[:, 1] = (height - 1) - contour_xy[:, 1]

points = pd.DataFrame({"x": contour_xy[:, 0], "y": contour_xy[:, 1]})
points.to_excel(output_xlsx, index=False)

points.head()


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

area_gauss = polygon_area(contour_xy)
area_cv2 = cv2.contourArea(contour)

print(f"Area con fórmula de Gauss: {area_gauss:.2f} pixeles^2")
print(f"Area con cv2.contourArea(): {area_cv2:.2f} pixeles^2")

print('image shape:', image.shape)
print('contour_xy_for_plot bounds: x', contour_xy_for_plot[:,0].min(), contour_xy_for_plot[:,0].max(), 'y', contour_xy_for_plot[:,1].min(), contour_xy_for_plot[:,1].max())

# 1) Contorno exclusivo
plt.figure(figsize=(6, 6))
contour_closed = np.vstack([contour_xy, contour_xy[0]])
plt.plot(contour_closed[:, 0], contour_closed[:, 1], color="blue", linewidth=2)
plt.title("Contorno de la imagen")
plt.gca().set_aspect('equal', 'box')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("assets/bubble_guppie_contour_exclusive.png", dpi=150, bbox_inches="tight")
plt.show()

plt.close()