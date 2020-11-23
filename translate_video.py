# Load the necessary modules 
import os
import re
import cv2
import numpy as np
import imageio
import base64
from PIL import Image
import skimage
from skimage import transform
import keras
from keras.models import Model
from net import Net

# Global variables 
numbers = re.compile(r'(\d+)')
imageSize = 64
folder = 'data/video_images'  
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}
use_frames = ['frame35.jpg', 'frame120.jpg', 'frame140.jpg', 'frame255.jpg', 'frame370.jpg', 'frame435.jpg', 'frame450.jpg',
                'frame545.jpg', 'frame615.jpg', 'frame670.jpg', 'frame770.jpg', 'frame840.jpg', 'frame915.jpg', 'frame1015.jpg',
                'frame1060.jpg', 'frame1150.jpg', 'frame1170.jpg', 'frame1315.jpg', 'frame1395.jpg', 'frame1445.jpg', 'frame1520.jpg',
                'frame1600.jpg', 'frame1680.jpg', 'frame1705.jpg', 'frame1800.jpg', 'frame1860.jpg']

def get_frames(video):
    """
    Converts a video to images for every 5 frames and
    saves images to data/video_images
    video: path to video 
    """
    vidcap = cv2.VideoCapture(video)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        if count % 5 == 0:
            cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(int(count))), image)
        count += 1
    print("{} images are extacted in {}.".format(int(count/5),folder))

def predict_frames():
    """
    Loads pretrained CNN to predict on all frame images
    predictions: nparray with class prediction on all images
    filenames: nparray of all frame filenames
    """
    frames, filenames = get_data()
    model = Net.build(width = imageSize, height = imageSize, depth = 3)
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.load_weights('trained_weights_2.h5')
    predicted_classes = model.predict(frames)
    predictions = np.argmax(predicted_classes,axis=1)

    #for i in range(len(predictions)):
        #print('Predicted class: {} for image: {} with confidence: {}'.format(map_characters[predictions[i]], filenames[i], np.amax(predicted_classes[i])*100))
    return predictions, filenames

def get_data():
    """
    Converts images from data/video_images to cropped
    nparrays 
    frames: nparray of cropped images with Sobel filter
    filenames: nparray of all filenames
    """
    frames = []
    filenames = []
    for imname in sorted(os.listdir(folder), key=numericalSort):
        if not imname.startswith('.'):
            im = imageio.imread(folder+'/'+imname)
            #im = im[:,180:1100,:]
            im = im[:,275:1000,:]
            im = skimage.transform.resize(im, (imageSize, imageSize, 3))
            img_arr = np.asarray(im)
            img_arr = preprocess_image(img_arr)
            frames.append(img_arr)
            filenames.append(imname)
    frames = np.asarray(frames)
    print('Finished converting frames to nparray')
    return frames, filenames

def preprocess_image(image):
    """
    Applied Sobel filter to image
    image: 3 channel image as nparray
    sobely: 3 channel image with Sobel filter as nparray
    """
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

def numericalSort(value):
    """
    Sorts without numbers i.e. frame1, frame2, frame3, frame4...
    value: filename of frame
    parts: parts to be sorted by
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def display_data(predictions, filenames):
    """
    Displays frames in use_frames on web interface 
    predictions: nparray of model predictions on each frame
    filenames: nparray of all filenames 
    """
    images = []
    translations = []
    for i in range(0,len(use_frames)):
        filename = use_frames[i]
        index = filenames.index(filename)
        im = Image.open(folder+'/'+filename)
        images.append(im)
        translations.append(map_characters[predictions[index]])
    return images, translations
