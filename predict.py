import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description= "Image classifier project - Part 2")
parser.add_argument('--input', default='./test_images/wild_pansy.jpg', action='store', type=str , help='image path')
parser.add_argument('--model', default='test_model.h5', action='store', type=str , help='Model file')
parser.add_argument('--top_k', default=5, dest='top_k', action='store', type=int , help='top k most likely classes')
parser.add_argument('--category_names',dest='category_names', action='store', default='label_map.json',help='flowers real names')

arg_parser = parser.parse_args()

image_path = arg_parser.input
model = arg_parser.model
top_k = arg_parser.top_k
category_names = arg_parser.category_names



with open('label_map.json', 'r') as f:
    class_names = json.load(f)

model_2 = tf.keras.models.load_model('./test_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
    
    
    
def process_image(img):
    #tensor_img = tf.image.convert_image_dtype(img, dtype=tf.int16, saturate=False)
    image = tf.convert_to_tensor(img)
    resized_img = tf.image.resize(image,(224,224)).numpy()
    final_img = resized_img/255
    return final_img



def predict(image_path, model, top_k=5):
    img = Image.open(image_path)
    test_img = np.asarray(img)
    processed_img = process_image(test_img)
    redim_img = np.expand_dims(processed_img, axis = 0)
    prob_pred = model_2.predict(redim_img)
    prob_pred = prob_pred.tolist()
    
    probs, classes = tf.math.top_k(prob_pred, k = top_k)
    
    probs = probs.numpy().tolist()[0]
    classes = classes.numpy().tolist()[0]

    return probs, classes
    
    
    
class_new_names = dict()
for n in class_names:
    class_new_names[str(int(n)-1)]= class_names[n]
    
def plot_img(img_path):
    
    img = Image.open(img_path)
    test_img = np.asarray(img)
    processed_img = process_image(test_img)
    
    
    probs, classes = predict(img_path, model_2, 5)
    print('Probabilities : ',probs)
    print('Lables : ',classes)
    
    flower_name = [class_new_names[str(idd)] for idd in classes]
    print('Flower\'s Names : ',flower_name)

    
    
if __name__ == "__main__" :
    print("start Prediction")
    probs, classes = predict(image_path, model, top_k)
    plot_img(image_path)
    print("End Predidction")