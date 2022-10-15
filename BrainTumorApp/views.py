from logging import exception
from unittest import result
from django.shortcuts import render


import torch
import torch.nn as nn
from torchvision import transforms
from model.inceptionresnetv2 import InceptionResNetV2, inceptionresnetv2
import torchvision.transforms.functional as TF

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

# import matplotlib.pyplot as plt



import os
import cv2
from PIL import Image
import numpy as np
from random import random

from django.conf import settings

from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage



model_predict = InceptionResNetV2()
for param in model_predict.parameters():
    param.requires_grad = False

num_inputs = model_predict.last_linear.in_features
model_predict.last_linear = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(num_inputs, 1000),
    nn.ReLU(),
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512,448),
    nn.ReLU(),
    nn.Linear(448, 320),
    nn.ReLU(),
    nn.Linear(320, 2)
)

model_predict.load_state_dict(torch.load("brain_tumor_inceptionresnetv2.pth", map_location=torch.device('cpu')))
model_predict.eval()


model_bbox = load_model('bbox_regression.h5')

import imutils

MEAN = [0.23740229, 0.23729787, 0.23700129]
STD = [0.23173477, 0.23151317, 0.23122775]


class ImageEnhanced(object):
    """_summary_
    transform to enhanced image quality for prediction 
    """
    def __init__(self):
        pass
    def __call__(self, img ,add_pixels_value = 0):
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        return Image.fromarray(new_img)

transforms = transforms.Compose([
    ImageEnhanced(),
    transforms.ToTensor(),
    transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(MEAN, STD)
])


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

def predictor(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()
    if request.method == 'POST':
        try:
            # Lay file gui len
            image = request.FILES["image"]
            print("Name", image.file)
            _image = fss.save(image.name, image)
            path = str(settings.MEDIA_ROOT) + "/" + image.name
            # image details
            image_url = fss.url(_image)
            # Read the image
            imag=cv2.imread(path)

            img = transforms(imag)

            img = torch.unsqueeze(img, 0)

            # img_from_ar = Image.fromarray(imag, 'RGB')
            # resized_image = img_from_ar.resize((3, 299, 299))
            # imag_resized = np.array(resized_image)
            # test_image =np.expand_dims(resized_image, axis=0) 

            outputs = model_predict(img)

            _,result = torch.max(outputs, 1)

            if (result == 0):
                prediction = "No"
            elif (result == 1):
                prediction = "Yes"
                img_test = load_img(path, target_size=(224, 224))
                img_test = img_to_array(img_test) / 255.0
                img_test = np.expand_dims(img_test, axis=0)
                preds = model_bbox.predict(img_test)[0]
                (startX, startY, endX, endY) = preds
                # image = cv2.imread(path)
                # img3 = imutils.resize(imag, width=600)
                (h, w) = imag.shape[:2]
                # scale the predicted bounding box coordinates based on the image
                # dimensions
                startX = int(startX * w)
                startY = int(startY * h)
                endX = int(endX * w)
                endY = int(endY * h)

                img1 = read_image(path)

                boxes = torch.tensor([[startX, startY, endX, endY]], dtype=torch.float)

                result1 = draw_bounding_boxes(img1, boxes, width=4)
                result2 = torch.transpose(result1, 0, 2)
                result2 = torch.transpose(result2, 0, 1)
                result2 = result2.numpy()
                # image = result1
                cv2.imwrite(path, result2)
                # _image = fss.save(image.name, image)
                # image_url = fss.url(_image)

            else:
                prediction = "Unknown"

            return TemplateResponse(
                request,
                "index.html",
                {
                    "message": message,
                    "image": image,
                    "image_url": image_url,
                    "prediction": prediction,
                },
            )

        except MultiValueDictKeyError:
            return TemplateResponse(
                request,
                "index.html",
                {"message": "No Image Selected"},
            )

    else:
        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )