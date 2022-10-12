from operator import mod
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from skimage.transform import resize
from skimage.io import imread
import pickle
import os

Categories = [
    "apple_scab",
    "black_rot",
    "cedar_apple_rust",
    "healthy",
    "powdery_mildew",
    "cercospora_leaf_spot_gray_leaf_spot",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "target_spot",
    "common_rust",
    "esca_black_measles",
    "leaf_blight",
    "septoria_leaf_spot",
    "tomato_mosaic_virus",
    "leaf_mold",
    "leaf_scorch",
    "tomato_yellow_leaf_curl_virus",
    "haunglongbing_citrus_greening",
    "spider_mites",
]

test_folder = 'test_images'

model_name = 'classifier_model.pkl'

with open(model_name, 'rb') as file:
    model = pickle.load(file)

test_img = 'leaf.jpg'

img = imread(os.path.join(test_folder, test_img))

img_resized = resize(img, (150, 150, 3))

l = [img_resized.flatten()]

probability = model.predict_proba(l)

for ind, val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')

print("Predicted image is: " + Categories[model.predict(l)[0]])