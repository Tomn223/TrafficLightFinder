import cv2
# from csv_util import getTestArray


tl_cascade_classifier = cv2.CascadeClassifier("cascade/cascade.xml")

with open(r"image_descriptor\positive_test.txt", "r") as f:
    pos_imgs = f.readlines()
    model_color = (0,255,0)
    actual_color = (0,0,255)
    thickness = 1
    for pos_img in pos_imgs[-10:]:
        img_info = pos_img.split(" ")
        print(img_info[0])
        img = cv2.imread(img_info[0][1:])

        boxes = tl_cascade_classifier.detectMultiScale(img)

        for box in boxes:
            for i in range(len(box)):
                start = (box[0], box[1])
                end = (start[0] + box[2], start[1] + box[3])
                img = cv2.rectangle(img, start, end, model_color, thickness)

        img_info = [int(img_info[i]) for i in range(1, len(img_info))]
        img_info.insert(0, 0) # TODO: improve this
        for i in range(img_info[1]):
            start = (img_info[2+4*i], img_info[3+4*i])
            end = (start[0] + img_info[4+4*i], start[1] + img_info[5+4*i])
            img = cv2.rectangle(img, start, end, actual_color, thickness)

        cv2.imshow("image", img)

        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()

        print(boxes)