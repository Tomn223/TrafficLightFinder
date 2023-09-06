import cv2
import numpy as np
import os

def detect_red_and_yellow(img, Threshold=0.001):

    desired_dim = (20, 30)  # width, height

    if not img.any(): # TODO: weird error
        return False
    
    img = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red1 = np.array([170, 70, 50])
    upper_red1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

    # defining the Range of yellow color
    lower_yellow = np.array([21, 39, 64])
    upper_yellow = np.array([40, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # red pixels' mask
    mask = mask0 + mask1 + mask2

    # Compare the percentage of red values
    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

    if rate > Threshold:
        return True
    else:
        return False
    

tl_cascade_classifier = cv2.CascadeClassifier("cascade/cascade.xml")

for file in os.listdir("./test_images"):
    model_color = (0,255,0)
    thickness = 1
    img = cv2.imread("./test_images/" + file)

    save_img = img
    boxes = tl_cascade_classifier.detectMultiScale(img)

    for box in boxes:
        # print(box)
        for i in range(len(box)):
            start = (box[0], box[1])
            text_start = (box[0], box[1] - 10)
            end = (start[0] + box[2], start[1] + box[3])

            tl_status = "red" if detect_red_and_yellow(img[box[0]:end[0], box[1]:end[1]]) else "green"

            save_img = cv2.rectangle(save_img, start, end, model_color, thickness)
            cv2.putText(img, tl_status, text_start, cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 1)

    cv2.imshow("image", save_img)
    cv2.imwrite("./test_results/" + file, save_img)
    cv2.waitKey(0)
    
    # closing all open windows
    cv2.destroyAllWindows()

with open(r"image_descriptor\positive_test.txt", "r") as f:
    pos_imgs = f.readlines()
    model_color = (0,255,0)
    actual_color = (0,0,255)
    thickness = 1
    for pos_img in pos_imgs:
        img_info = pos_img.split(" ")
        img = cv2.imread(img_info[0][1:])

        boxes = tl_cascade_classifier.detectMultiScale(img)

        for box in boxes:
            # print(box)
            for i in range(len(box)):
                start = (box[0], box[1])
                text_start = (box[0], box[1] - 10)
                end = (start[0] + box[2], start[1] + box[3])

                tl_status = "stop" if detect_red_and_yellow(img[box[0]:end[0], box[1]:end[1]]) else "go"

                img = cv2.rectangle(img, start, end, model_color, thickness)
                cv2.putText(img, tl_status, text_start, cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 1)

        actual_boxes = [[int(img_info[i]), int(img_info[i+1]), int(img_info[i+2]), int(img_info[i+3])] for i in range(1, len(img_info), 4)]
        for x1, y1, w, h in actual_boxes:
            start = (x1, y1)
            end = (x1 + w, y1 + h)
            img = cv2.rectangle(img, start, end, actual_color, thickness)

        cv2.imshow("image", img)

        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()