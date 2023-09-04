import cv2
import numpy
import csv_util

# img = cv2.imread(r".\sample-dayClip6\sample-dayClip6\frames\dayClip6--00000.jpg")
# img = cv2.imread("/sample-dayClip6/sample-dayClip6/frames/dayClip6--00000.jpg")

# C:/Users/tomne/Downloads/opencv/build/x64/vc15/bin/opencv_createsamples -info image_descriptor/positive.dat -w 24 -h 24 -num 1000 -vec image_descriptor/positive.vec
# C:/Users/tomne/Downloads/opencv/build/x64/vc15/bin/opencv_traincascade -data cascade/ -vec image_descriptor/positive.vec -bg image_descriptor/negative.txt -w 24 -h 24 -numPos 300 -numNeg 300 -numStages 10 -minHitRate 0.998 -maxFalseAlarmRate 0.3

with open(r"image_descriptor\positive.dat", "r") as f:
    pos_imgs = f.readlines()
    color = (0,255,0)
    thickness = 1
    for pos_img in pos_imgs:
        img_info = pos_img.split(" ")
        print(img_info[0])
        img = cv2.imread(img_info[0])
        # cv2.imshow("image", img)

        # cv2.waitKey(0)
        
        # # closing all open windows
        # cv2.destroyAllWindows()
        # assert False
        img_info = [int(img_info[i]) for i in range(1, len(img_info))]
        img_info.insert(0, 0)
        for i in range(img_info[1]):
            start = (img_info[2+4*i], img_info[3+4*i])
            end = (start[0] + img_info[4+4*i], start[1] + img_info[5+4*i])
            img = cv2.rectangle(img, start, end, color, thickness)

        cv2.imshow("image", img)

        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()