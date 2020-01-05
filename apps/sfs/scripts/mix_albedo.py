import png
import sys
import numpy as np
import pyexr
import cv2

def read_png(filename):
    with open(filename) as f:
        reader = png.Reader(file=f)
        idirect = reader.asDirect()
        idata = list(idirect[2])
        (w, h) = idirect[3]['size']
        c = len(idata[0]) / w

        img = np.zeros((h, w, c), dtype=np.uint16)
        for i in range(len(idata)):
            for j in range(c):
                img[i, :, j] = idata[i][j::c]

        return np.array(img, dtype=np.float)


if len(sys.argv) < 3:
    print("Missing arguments: mix_albedo.py <path> <idx>")
    exit(0)



path = sys.argv[1]
num = sys.argv[2]

def get_abs(path, file, num):
    postfix = "_" + str(num) if int(num) > 0 else ""
    return path + "/" + file + postfix + ".png"

albedo = read_png(get_abs(path, "albedo", num)) / 255
if albedo.shape[2] > 3:
    albedo = albedo[:, :, :3]
shading = np.array(pyexr.read(path + "/shading.exr")) / 255

if albedo.shape[:2] != shading.shape[:2]:
    print("Resizing albedo")
    albedo = cv2.resize(albedo, (shading.shape[1], shading.shape[0]))

color = albedo * shading

cv2.imshow("Shading", shading)
cv2.imshow("Color", color)
cv2.waitKey(0)

color = np.reshape(color, (-1, color.shape[1] * 3))
color = np.array(color * 255, dtype=np.uint8)

with open(get_abs(path, "color", num), "wb") as f:
    w = png.Writer(albedo.shape[1], albedo.shape[0])
    w.write(f, np.array(color, dtype=np.uint8))

