import pandas as pd
import os
import matplotlib.pyplot as plt
import skimage
# import skimage.transform
from skimage.transform import resize
from skimage.io import imread
import scipy
import imageio
import numpy as np
import pickle

Categories = [
    "apple",
    "corn",
    "grape",
    "none",
    "potato",
    "tomato"
]

AUGS = ["aug", "unaug"]

flat_data_arr = [] #input array
target_arr = [] #output array

datadir = 'type_data' 

#path which contains all the categories of images
for i in Categories:   
    print(f'Current Category : {i}')
    path = os.path.join(datadir, i)

    for a in AUGS:
        print(f"Current augmentaiton: {a}")
        a_path = os.path.join(path, a)
        for s in os.listdir(a_path):
            s_path = os.path.join(a_path, s)
            for img in os.listdir(s_path):
                img_array = imread(os.path.join(s_path,img))
                img_resized = resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                target_arr.append(Categories.index(i))

    # for img in os.listdir(path):
    #     # img_array = imageio.imread(os.path.join(path,img))
    #     img_array = imread(os.path.join(path,img))
    #     img_resized = resize(img_array,(150,150,3))
    #     # img_resized = scipy.misc.imresize(img_array,(150,150,3))
    #     flat_data_arr.append(img_resized.flatten())
    #     target_arr.append(Categories.index(i))

    print(f'Loaded category: {i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data) #dataframe
df['Target'] = target

x = df.iloc[:, :-1] #input data 
y = df.iloc[:, -1] #output data

from sklearn import svm
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

param_grid = {
    'C':[0.1, 1, 10, 100],
    'gamma':[0.0001, 0.001, 0.1, 1],
    'kernel':['rbf', 'poly'],
}
svc = svm.SVC(probability=True, verbose=True, kernel='rbf')
model = OneVsRestClassifier(svc, verbose=1, n_jobs=-1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

print('Splitted Successfully')

model.fit(x_train, y_train)

print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV

# save model
filename = 'type_classifier_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# to open saved model-
# with open(pkl_filename, 'rb') as file:
    # pickle_model = pickle.load(file)  

# url=input('Enter URL of Image :')
# url = "sick_leaf.jpg"
# img=imread(url)

# plt.imshow(img)
# # plt.show()

# img_resize=resize(img, (150, 150, 3))

# l = [img_resize.flatten()]

# probability = model.predict_proba(l)

# for ind,val in enumerate(Categories):
#     print(f'{val} = {probability[0][ind]*100}%')

# print("The predicted image is : "+Categories[model.predict(l)[0]])

# '''
# https://www.collinsdictionary.com/images/thumb/leaf_198990134_250.jpg?version=4.0.276
# '''