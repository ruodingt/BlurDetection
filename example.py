import os
import cv2
import numpy
import BlurDetection

folder_dir = "/media/ruodingt/D4F27D38F27D2042/dentalpoc/data/tooth_decay"

img_fn = "0-niLWplAELuAW_vPWaNCQ==.png"
# img_fn = "turnpike-blur.jpg"

img_path = os.path.join(folder_dir, img_fn)
assert os.path.exists(img_path), "img_path does not exists"
img = cv2.imread(img_path)
img_fft, val, blurry = BlurDetection.blur_detector(img)

print("blur score:", val)
print("this image {0} blurry".format(["isn't", "is"][blurry]))
msk, result, blurry = BlurDetection.blur_mask(img)
BlurDetection.scripts.display('img', img)
# cv2.waitKey(0)
BlurDetection.scripts.display('msk', msk)
cv2.waitKey(0)