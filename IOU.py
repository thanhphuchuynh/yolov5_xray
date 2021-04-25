#lib
import pandas as pd
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

PATH_FOLDER = '/Final/Demo/VINAI_Chest_Xray/VINAI_Chest_Xray/train/'
PATH_csv = '/Final/Demo/VINAI_Chest_Xray/VINAI_Chest_Xray/train.csv'

df_train = pd.read_csv(PATH_csv)
# print(df_train)

# test_img_id = '02617da0a33fe0446a508186417c2646'
test_img_id = '89e6ab133f587191383608ee04cea79a'

test_df = df_train[df_train['image_id'] == test_img_id]
print(test_df) 

lables =[
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]

def draw_bbox(img_id, df):
    # print(df)
    image_path = PATH_FOLDER + img_id + '.jpg'
    
    img = cv2.imread(image_path)
    dh, dw, _ = img.shape # ex (960, 768, 3)
    for i,y in df.iterrows():
        class_name = y['class_name']
        l = int(y['x_min']*dw/y['width'])
        r = int(y['x_max']*dw/y['width'])
        t = int(y['y_min']*dh/y['height'])
        b = int(y['y_max']*dh/y['height'])
        color = (0,0,255)
        cv2.putText(img, class_name, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.rectangle(img, (l, t), (r, b), (255,0,0), 2)
    return img
imgOriginal = draw_bbox(test_img_id, test_df)

#https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
   # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     Consider a 1D example :
# - 2 points : x1 = 1 and x2 = 3, the distance is indeed x2-x1 = 2
# - 2 pixels of index : i1 = 1 and i2 = 3, the segment from pixel i1 to i2 contains 3 pixels ie l = i2 - i1 + 1
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def averageCoordinates(df,threshold):
    tmp_df = df.reset_index() # 0 1 2 3 
    duplicate = {}
    for index1, row1 in tmp_df.iterrows():
        if index1 < len(tmp_df) -1 :
            nextIndex = index1 + 1
            for index2, row2 in tmp_df.loc[nextIndex:,:].iterrows():
                if row1['class_id'] == row2['class_id']:
                    boxA = [row1['x_min'], row1['y_min'], row1['x_max'], row1['y_max']]
                    boxB = [row2['x_min'], row2['y_min'], row2['x_max'], row2['y_max']]
                    iou = bb_intersection_over_union(boxA, boxB)
                    # print("class_id", row1["class_id"])
                    # print("iou", iou)
                    if iou > threshold:
                        if row1["index"] not in duplicate:
                            duplicate[row1["index"]] = []
                        duplicate[row1["index"]].append(row2["index"])
    remove_keys = []
    # print(tmp_df,"\n",duplicate)
    for k in duplicate:
        for i in duplicate[k]:
            if i in duplicate:
                for id in duplicate[i]:
                    if id not in duplicate[k]:
                        duplicate[k].append(id)
                if i not in remove_keys:
                    remove_keys.append(i)
    print("\n")        
    print(remove_keys,"\n",duplicate)
    for i in remove_keys:
        del duplicate[i]
    # print(tmp_df,duplicate)
    print(remove_keys,"\n",duplicate)
    rows = []
    removed_index = []
    for k in duplicate:
        row = tmp_df[tmp_df['index'] == k].iloc[0]
        # print(row,"\n")
        X_min = [row['x_min']]
        X_max = [row['x_max']]
        Y_min = [row['y_min']]
        Y_max = [row['y_max']]
        removed_index.append(k)
        for i in duplicate[k]:
            removed_index.append(i)
            row = tmp_df[tmp_df['index'] == i].iloc[0]
            X_min.append(row['x_min'])
            X_max.append(row['x_max'])
            Y_min.append(row['y_min'])
            Y_max.append(row['y_max'])
        X_min_avg = sum(X_min) / len(X_min)
        X_max_avg = sum(X_max) / len(X_max)
        Y_min_avg = sum(Y_min) / len(Y_min)
        Y_max_avg = sum(Y_max) / len(Y_max)
        new_row = [row['image_id'], row['class_name'], row['class_id'], X_min_avg, Y_min_avg, X_max_avg, Y_max_avg, row['width'], row['height']]
        # print(new_row)
        rows.append(new_row)

    for index, row in tmp_df.iterrows():
        if row['index'] not in removed_index:
            new_row = [row['image_id'], row['class_name'], row['class_id'], row['x_min'], row['y_min'], row['x_max'], row['y_max'], row['width'], row['height']]
            rows.append(new_row)

    new_df = pd.DataFrame(rows, columns =['image_id', 'class_name', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height'])
    return new_df
    
def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

new_df = averageCoordinates(test_df,0.5)
print(new_df)

rows = 1
columns = 2 
imgIOU = draw_bbox(test_img_id, new_df)
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(rows, columns,1) 

plt.imshow(imgOriginal)
plt.axis('off')
plt.title("Original")
fig.add_subplot(rows, columns,2)
plt.axis('off')
plt.title("With IOU")
plt.imshow(imgIOU)
plt.show()


im = resize(imgIOU, size=512)  
print(np.ascontiguousarray(im).shape)

# im.save("256.jpg")
im = Image.fromarray(imgIOU)
print(np.ascontiguousarray(im).shape)

# im.save("ori.jpg")
print(im)
