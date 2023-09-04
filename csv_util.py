import csv
import numpy as np

hasTL = [False for i in range(468)] # used to find negative images
positive_dict = {} # i: [# TL's, [(), ...]]
positive_array = np.array([])

with open(r'.\sample-dayClip6\sample-dayClip6\frameAnnotationsBOX.csv', newline='') as csvfile:
    annotations = csv.reader(csvfile, delimiter=";")
    next(annotations) # skip header line
    for row in annotations:
        imgNum = int(row[-1])
        hasTL[imgNum] = True
        # add bounding box (upper left x, upper left y, w, h)
        # row[4] is lower right x, row[5] is lower right y
        boundingBox= (row[2], row[3], str(int(row[4])-int(row[2])), str(int(row[5])-int(row[3])))
        if imgNum in positive_dict:
            positive_dict[imgNum][0] += 1
            positive_dict[imgNum][1].append(boundingBox)
        else:
            positive_array = np.append(positive_array, imgNum)
            positive_dict[imgNum] = [1, [boundingBox]]


# randomize order of images, used for randomized test set
np.random.shuffle(positive_array)

train_ratio = .98
train_cutoff = int(train_ratio*positive_array.size)
# print(train_cutoff)
train_array = positive_array[:train_cutoff]
test_array = positive_array[train_cutoff:]
# print(test_array.size)



def intToStrFillZeros(strSize, num):
    intWithZeros = str(num)

    while len(intWithZeros) != strSize:
        intWithZeros = "0" + intWithZeros
    
    return intWithZeros

def allBoundingBoxStr(boxList):

    coordStr = ""

    for coords in boxList:
        for coord in coords:
            coordStr += coord + " "

    coordStr = coordStr[:-1] # shave off hanging space
    return coordStr

count = 0

with open(r"image_descriptor\negative.txt", "w") as f:
    for i in range(len(hasTL)):
        if not hasTL[i]:
            if count < 54:  #TODO remove count/specify
                count += 1
                f.write(f"./sample-dayClip6/sample-dayClip6/frames/dayClip6--{intToStrFillZeros(5, i)}.jpg\n")
            else:
                # TODO add to test set
                print(f"Negative image used to test: {i}")

print(count)

# TODO: There has to be a better way to do this...
with open(r"image_descriptor\positive_test.txt", "w") as f:
    for i in test_array:
        count += 1
        f.write(f"../sample-dayClip6/sample-dayClip6/frames/dayClip6--{intToStrFillZeros(5, int(i))}.jpg {positive_dict[int(i)][0]} {allBoundingBoxStr(positive_dict[int(i)][1])}\n")

with open(r"image_descriptor\positive.dat", "w") as f:
    for i in train_array:
        count += 1
        # TODO: BIG PROBLEM, FILE INCONGRUENCY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        f.write(f"../sample-dayClip6/sample-dayClip6/frames/dayClip6--{intToStrFillZeros(5, int(i))}.jpg {positive_dict[int(i)][0]} {allBoundingBoxStr(positive_dict[int(i)][1])}\n")

print(count)