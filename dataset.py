import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#reading all the images of a specific number(folder)
def read_all_imgs(num): 
    folder_path = r"C:\Users\lenovo\10mlprojects\pelak\dataset" + f"\\{num}"
    file_names = os.listdir(folder_path)
    return file_names

def create_dataset_segments(num):
    files = read_all_imgs(num)
    imgs = [] 
    for i in range(0, len(files)):
        img = cv2.imread(r"C:\Users\lenovo\10mlprojects\pelak\dataset"+ f"\\{num}\\" + files[i])
        img_resized = cv2.resize(img, (8, 32))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_flattened = img_gray.flatten()
        imgs.append(img_flattened)
    imgs = np.array(imgs)
    return imgs, len(files)

def create_dataset():
    dataset = []
    lengths = []
    for i in range(1,10):
        dataset.extend(create_dataset_segments(i)[0])
        lengths.append(create_dataset_segments(i)[1])
    labels = [1]*350 + [2] *403 + [3] * 346 + [4] *328 + [5] * 349 + [6] * 300 + [7] * 303 + [8] * 345 + [9] *377
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

def clean_scale_dataset(dataset, labels):
    mask = np.isnan(dataset).any(axis = 1)
    cleaned_dataset = dataset[~mask]
    cleaned_labels = labels[~mask]

    cleaned_normalized_dataset = cleaned_dataset/255.0
    x_train, x_test, y_train, y_test = train_test_split(cleaned_normalized_dataset, cleaned_labels, train_size = 0.8, random_state = 42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test
