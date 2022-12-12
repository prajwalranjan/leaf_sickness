from argparse import Action
from operator import mod
from sre_parse import CATEGORIES
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from skimage.transform import resize
from skimage.io import imread
import pickle
import os
import tkinter
from PIL import ImageTk, Image

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

CATEGORIES = {
    "apple_scab": "Apple Scab",
    "black_rot": "Black Rot",
    "cedar_apple_rust": "Cedar Apple Rust",
    "healthy": "Healthy",
    "powdery_mildew": "Powdery Mildew",
    "cercospora_leaf_spot_gray_leaf_spot": "Cercospora (Gray) Leaf Spot",
    "bacterial_spot": "Bacterial Spot",
    "early_blight": "Early Blight",
    "late_blight": "Late Blight",
    "target_spot": "Target Spot",
    "common_rust": "Common Rust",
    "esca_black_measles": "Esca Black Measles",
    "leaf_blight": "Leaf Blight",
    "septoria_leaf_spot": "Septoria Leaf Spot",
    "tomato_mosaic_virus": "Tomato Mosaic Virus",
    "leaf_mold": "Leaf Mold",
    "leaf_scorch": "Leaf Scorch",
    "tomato_yellow_leaf_curl_virus": "Tomato Yellow Leaf Curl Virus",
    "haunglongbing_citrus_greening": "Haunglongbing Citrus Greening",
    "spider_mites": "Spider Mites",
}

# test_folder = 'test_images'
test_folder = "custom_test/tomato"

model_name = 'classifier_model.pkl'

with open(model_name, 'rb') as file:
    model = pickle.load(file)

test_img = 'leaf.jpg'

root = tkinter.Tk()
root.title("Leaf Sickness Detector")
root.geometry("1000x800")

filename_variable = tkinter.StringVar()

def predict_image():
    # frame = tkinter.Frame(root, width=600, height=400)
    # frame.grid(row=1, column=0)
    # frame.place(anchor='center', relx=0.5, rely=0.5)

    loading = tkinter.Label(root, text="Loading...", font=('calibre', 10, 'normal'))
    loading.grid(row=1, column=0)

    

    # temp_img = ImageTk.PhotoImage(Image.open(os.path.join(test_folder, filename)))

    filename = filename_variable.get()

    temp_image = Image.open(os.path.join(test_folder, filename))
    resized_temp_image = temp_image.resize((150, 150))
    photo = ImageTk.PhotoImage(resized_temp_image)
 
    entry_label = tkinter.Label(root, text="Image: ", font=('calibre', 10, 'normal'))
    entry_label.grid(row=2, column=2)
    
    img_label = tkinter.Label(root, image=photo)
    img_label.photo = photo
    img_label.grid(row=3, column=2)

    img = imread(os.path.join(test_folder, filename))
    img_resized = resize(img, (150, 150, 3))
    l = [img_resized.flatten()]
    probability = model.predict_proba(l)

    intro_label = tkinter.Label(root, text="The following predictions were made: ", font=('calibre', 10, 'normal'))
    intro_label.grid(row=5, column=0)

    row = 7

    final_vals = []
    
    for ind, val in enumerate(Categories):
        type_label = tkinter.Label(root, text=CATEGORIES[val], font=('calibre', 10, 'normal'))
        prob = probability[0][ind]*100
        probs_label = tkinter.Label(root, text=str(prob)+"%", font=('calibre', 10, 'normal'))
        # print(f'{val} = {probability[0][ind]*100}%')
        type_label.grid(row=row, column=0)
        probs_label.grid(row=row, column=1)
        row += 1

        final_vals.append([val, prob])

    sorted_final_vals = sorted(final_vals, key=lambda x: x[1], reverse=True)

    # def get_prediction(probabilities):
    #     max_p = 0
    #     for i in range(1, len(probabilities)):
    #         if probabilities[i] > probabilities[max_p]:
    #             max_p = i
    #     return max_p
    
    # prediction = CATEGORIES[Categories[model.predict(l)[0]]]

    # acc_pred = get_prediction(probabilities=probability)

    acc_pred = sorted_final_vals[0][0]

    # prediction = CATEGORIES[Categories[acc_pred]]

    prediction = CATEGORIES[acc_pred]

    pred_label = tkinter.Label(root, text="Predicted leaf condition: "+prediction, font=('calibre', 15, 'bold'))
    pred_label.grid(row=row+4, column=2)

    close_button = tkinter.Button(root, text="Exit", command=root.destroy)
    close_button.grid(row=row+6, column=1)


file_label = tkinter.Label(root, text="Enter filename: ", font=('calibre', 10, 'normal'))
file_entry = tkinter.Entry(root, textvariable=filename_variable, font=('calibre', 10, 'normal'))
submit_button = tkinter.Button(root, text='Submit', command=predict_image)

file_label.grid(row=0, column=0)
file_entry.grid(row=0, column=1)

submit_button.grid(row=0, column=2)

root.mainloop()