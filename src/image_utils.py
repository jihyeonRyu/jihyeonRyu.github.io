import imutils
import os
import cv2

image_path = "./assets/resource/ai_paper/paper57"

WIDTH = 500
filenames = os.listdir(image_path)

for file in filenames:
    try:
        full_path = os.path.join(image_path, file)
        image = cv2.imread(full_path)
        w = image.shape[1]
        target = min(WIDTH, w)
        tmp_image = imutils.resize(image, width=target)
        cv2.imshow("test", tmp_image)
        k = cv2.waitKey(0)
        if k == 121 or k == 89: # key (Y, y)
            print("Resize Image")
            cv2.imwrite(full_path, tmp_image)
        else:
            print("Not Resize Image")
            continue
    except Exception:
        continue

