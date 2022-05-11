import gradio as gr
import torch
import torchvision
import numpy as np
from PIL import Image

# Face masks
model = torch.hub.load('ultralytics/yolov5', 'custom', "model_weights/face_masks_partial.pt")

# Animals
# model = torch.hub.load('ultralytics/yolov5', 'custom', "model_weights/datasets_1000_41class.pt",force_reload=True)



def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0])


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "Detecting masked and unmasked faces with YOLOv5"
description = "YOLOv5 Gradio demo for finding faces with and without masks, using object detection. Upload an image or click an example image to use."
article = "<p style='text-align: center'>YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"

examples = [['data/picard.jpg'], ['data/stockmasks.jpg']]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples, theme="huggingface").launch(cache_examples=True,enable_queue=True)
