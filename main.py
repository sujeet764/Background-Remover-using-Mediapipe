import os

import numpy as np
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,90)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()


listImg = os.listdir("bgimage")

imgList = []
for imgPath in listImg:
    img = cv2.imread(f'bgimage/{imgPath}')
    imgList.append(img)

indexImage = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgList[indexImage],threshold=0.8  )
    print(indexImage)
    imageStacked = cvzone.stackImages([img,imgOut],2,1)
    _, imageStacked = fpsReader.update(imageStacked, color=(0,0,255))
    cv2.imshow("image", imageStacked)


    k = cv2.waitKey(1)
    if k == ord('a'):
        if indexImage > 0:
            indexImage-=1
    elif k == ord('d'):
        if indexImage < len(imgList)-1:
            indexImage+=1
    elif k == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
