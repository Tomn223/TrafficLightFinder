import cv2
import numpy
import csv

hasTL = [False for i in range(466)] # used to find negative images

with open(r'.\sample-dayClip6\sample-dayClip6\frameAnnotationsBOX.csv', newline='') as csvfile:
    annotations = csv.reader(csvfile, delimiter=";")
    next(annotations) # skip header line
    for row in annotations:
        # print(row[-1])
        # print(', '.join(row))
        hasTL[int(row[-1])] = True

count = 0
for i in range(len(hasTL)):
    if not hasTL[i]:
        count += 1
        print(f"No TL's in image: {i+1}")

print(count)

img = cv2.imread(r".\sample-dayClip6\sample-dayClip6\frames\dayClip6--00000.jpg")

cv2.imshow("image", img)

cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()