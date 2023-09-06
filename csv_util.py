import csv
import numpy as np
import random
import os

# hasTL = [False for i in range(468)] # used to find negative images
positive_dict = {} # i: [# TL's, [(), ...]]
# positive_array = np.array([])
negative_array = []

negCount = 0

def intToStrFillZeros(num, strSize):
    intWithZeros = str(num)

    while len(intWithZeros) != strSize:
        intWithZeros = "0" + intWithZeros
    
    return intWithZeros

for dir in os.listdir("./images"):
    dir_path = "images\\" + dir
    image_path = dir_path + "\\frames\\"
    with open(dir_path + "\\frameAnnotationsBOX.csv", newline='') as csvfile:
        annotations = csv.reader(csvfile, delimiter=";")
        next(annotations) # skip header line
        lastImgNum = 0
        for row in annotations:
            imgNum = int(row[-1])
            path = row[0].split("/")
            file_name = path[1]
            file_path = image_path + file_name

            if imgNum - lastImgNum > 1:
                # no traffic lights in these images (supposedly)
                for i in range(lastImgNum+1, imgNum):
                    negative_array.append((image_path + dir + "--", i))

            lastImgNum = imgNum
            # add bounding box (upper left x, upper left y, w, h)
            # row[4] is lower right x, row[5] is lower right y
            boundingBox= (row[2], row[3], str(int(row[4])-int(row[2])), str(int(row[5])-int(row[3])))

            # img = cv2.imread(dir_path + "\\frames\\" + fName)
            # img = cv2.rectangle(img, (int(row[2]), int(row[3])), (int(row[4]), int(row[5])), (0,0,255), 1)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            if file_path in positive_dict:
                positive_dict[file_path][0] += 1
                positive_dict[file_path][1].append(boundingBox)
            else:
                # positive_array = np.append(positive_array, imgNum)
                positive_dict[file_path] = [1, [boundingBox]]

# print(negative_array)
# print("neg count:", negative_array.size())
# randomize order of images, used for randomized test set
# np.random.shuffle(positive_array)

train_ratio = .98
train_cutoff = int(train_ratio*len(positive_dict))

# train_array = positive_array[:train_cutoff]
# test_array = positive_array[train_cutoff:]

keys = list(positive_dict.keys())
random.shuffle(keys)

train_keys = keys[:train_cutoff]
test_keys = keys[train_cutoff:]

def allBoundingBoxStr(boxList):

    coordStr = ""

    for coords in boxList:
        for coord in coords:
            coordStr += coord + " "

    coordStr = coordStr[:-1] # shave off hanging space
    return coordStr

count = 0

# add negative image file paths to negative.txt
with open(r"image_descriptor\negative.txt", "w") as f:
    for negative in negative_array:
        f.write(".\\" + negative[0] + f"{intToStrFillZeros(negative[1], 5)}.jpg\n")

print(count)

# # TODO: There has to be a better way to do this...
with open(r"image_descriptor\positive_test.txt", "w") as f:
    for file_path in test_keys:
        count += 1
        f.write("..\\" + file_path + f" {allBoundingBoxStr(positive_dict[file_path][1])}\n")
#     for i in test_array:
#         count += 1
#         f.write(f"../sample-dayClip6/sample-dayClip6/frames/dayClip6--{intToStrFillZeros(int(i), 5)}.jpg {positive_dict[int(i)][0]} {allBoundingBoxStr(positive_dict[int(i)][1])}\n")

with open(r"image_descriptor\positive.dat", "w") as f:
    for file_path in train_keys:
        count += 1
#         # TODO: BIG PROBLEM, FILE INCONGRUENCY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        f.write("..\\" + file_path + f" {positive_dict[file_path][0]} {allBoundingBoxStr(positive_dict[file_path][1])}\n")

# for 
print(count)