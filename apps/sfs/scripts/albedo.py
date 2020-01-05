import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load color, shading
color = cv2.imread('/data/real/color.png')
shading = cv2.imread('/data/real/shading.png')

# Filter bilaterally
color = cv2.bilateralFilter(color, 10, 5, 10)
shading = cv2.bilateralFilter(shading, 10, 5, 10)

color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
color = color.astype(np.float32) / 255
shading = shading.astype(np.float32) / 255

# Correct shading to be less "steep"
fac = 0.7
shading = (shading * fac) + (1 - fac)
#plt.imshow(shading)
#plt.show()

# Divide to get albedo estimate
albedo_est = color / shading
#albedo_est = color
albedo_est[albedo_est > 1] = 1
albedo_est[albedo_est < 0] = 0
#plt.imshow(albedo_est)
#plt.show()

def rgb_2_cielab(image):
    """
    Convert an image from RGB to CIE L*a*b space
    Adapted from https://www.mathworks.com/matlabcentral/fileexchange/24009-rgb2lab
    """
    
    # Make float [0..1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255

    # Threshold
    threshold = 0.008856

    h, w, c = image.shape

    # Split channels
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    RGB = np.stack([R.ravel(), G.ravel(), B.ravel()])
    
    # Conversion matrix
    conversion = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])

    XYZ = np.dot(image, conversion)
        
    # Normalize to D65 whitepoint
    X = XYZ[:, :, 0] / 0.950456
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2] / 1.088754

    # Create masks
    XT = X > threshold
    YT = Y > threshold
    ZT = Z > threshold

    X3 = np.power(X, 1/3)
    Y3 = np.power(Y, 1/3)
    Z3 = np.power(Z, 1/3)

    fX = XT * X3 + np.logical_not(XT) * (7.787 * X + 16/116)
    fY = ZT * Y3 + np.logical_not(YT) * (7.787 * Y + 16/116)
    fZ = YT * Z3 + np.logical_not(ZT) * (7.787 * Z + 16/116)

    L = (YT * (116 * Y3 - 16.0) + np.logical_not(YT) * (903.3 * Y)).reshape(w, h)
    a = (500 * (fX - fY)).reshape(w, h)
    b = (200 * (fY - fZ)).reshape(w, h)

    return np.stack([L, a, b], axis=2)

albedo_cielab = rgb_2_cielab(albedo_est)
plt.imshow(albedo_cielab)
plt.show()
