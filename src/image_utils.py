import imutils
import os
import cv2

image_path = "./../assets/resource/gan/paper1"

WIDTH = 500
filenames = os.listdir(image_path)

for file in filenames:
    full_path = os.path.join(image_path, file)
    image = cv2.imread(full_path)
    tmp_image = imutils.resize(image, width=WIDTH)
    cv2.imshow("test", tmp_image)
    k = cv2.waitKey(0)
    if k == 121 or k == 89:
        print("Resize Image")
        cv2.imwrite(full_path, tmp_image)
    else:
        print("Not Resize Image")
        continue