import png
import sys
import numpy as np
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

        return np.array(img, dtype=np.float32)


if len(sys.argv) < 3:
    print("Missing arguments: create_data_from_depth.py <path> <idx>")
    exit(0)


path = sys.argv[1]
num = sys.argv[2]

albedo = read_png(path + "/albedo_" + str(num) + ".png") / 255
if albedo.shape[2] > 3:
    albedo = albedo[:, :, :3]
depth = read_png(path + "/depth.png") / 4096

if albedo.shape[:2] != depth.shape[:2]:
    albedo = cv2.resize(albedo, (depth.shape[1], depth.shape[0]))

#light = np.array([0.0268, 0.0138, -0.6445, 0.2949], dtype=np.float)
light = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float)
fx = 1399.19
fy = 1399.19
cx = 640
cy = 480

# calculate theta
dx = np.pad(np.diff(depth, 1, axis=1), ((0, 0), (0, 1), (0, 0)), 'constant')
dy = np.pad(np.diff(depth, 1, axis=0), ((0, 1), (0, 0), (0, 0)), 'constant')

theta = np.array([depth, dx, dy])

# calculate normals from theta
idx = np.indices(depth.shape[0:2]) # x: idx[1], y: idx[0]

n_x = fx * theta[1]
n_y = fy * theta[2]
n_z = -theta[0] - theta[1]  *  np.expand_dims(idx[1] - cx, axis=2) - theta[2] * np.expand_dims(idx[0] - cy, axis=2)

normals = np.concatenate([n_x, n_y, n_z], axis=2)

normals_norm = np.maximum(np.finfo(np.float).eps, np.linalg.norm(normals, 2, axis=2))
normals = normals / np.expand_dims(normals_norm, axis=2)

# calculate shading from harmonics and lighting
sharms = np.concatenate([normals, np.ones_like(depth)], axis=2)
shading = np.dot(sharms, light)

shading_orig = read_png(path + "/shading.png") / 255
shading_orig = shading_orig[:, :, 0]


color = albedo * np.expand_dims(shading, axis=2)
cv2.imshow("Color", cv2.cvtColor(color.astype(np.float32), cv2.COLOR_RGB2BGR))
color = np.array(color * 255, dtype=np.uint8)
color = np.reshape(color, (-1, color.shape[1] * 3))


cv2.imshow("Shading", shading)
cv2.imshow("Normals", (normals  + 1) * 0.5)
cv2.waitKey(0)


with open(path + "/color_" + str(num) + ".png", "wb") as f:
    w = png.Writer(albedo.shape[1], albedo.shape[0])
    w.write(f, np.array(color, dtype=np.uint8))

