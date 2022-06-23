# WIP
# MegaDetector v5 Demo
import gradio as gr
import torch
import torchvision
import numpy as np
from PIL import Image

# Markdown Content
title = """<h1 id="title">MegaDetector v5</h1>"""
description = "Detect and identify animals, people and vehicles in camera trap images."
article = "<p style='text-align: center'>MegaDetector makes predictions using a YOLOv5 model that was trained to detect animals, humans, and vehicles in camera trap images; find out more about the project on <a href='https://github.com/microsoft/CameraTraps'>Microsoft's CameraTraps GitHub</a>. This app was built by <a href='https://github.com/hlydecker'>Henry Lydecker</a> but really depends on code and models developed by <a href='http://ecologize.org/'>Ecologize</a> and <a href='http://aka.ms/aiforearth'>Microsoft AI for Earth</a>. Find out more about the YOLO model from the original creator, <a href='https://pjreddie.com/darknet/yolo/'>Joseph Redmon</a>. YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset and developed by Ultralytics, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"

# Load MegaDetector v5a model
# TODO: Allow user selectable model?
models = ["model_weights/md_v5a.0.0.pt","model_weights/md_v5b.0.0.pt"]

# model = torch.hub.load('ultralytics/yolov5', 'custom', "model_weights/md_v5a.0.0.pt")

def yolo(img_input, size=640, model_name):

    model = model = torch.hub.load('ultralytics/yolov5', 'custom', model_name)
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0])

demo = gr.Blocks()

with demo:
    gr.Markdown(title)
    gr.Markdown(description)
    options = gr.Dropdown(choices=models, label="Select MegaDetector Model", show_label=True)
    
    with gr.Row():
        img_input = gr.Image(type='pil', label="Original Image")
        img_output = gr.Image(type="pil", label="Output Image")
    
    with gr.Row():
        example_images = gr.Dataset(components=[img_input],
                                    samples=[['data/Macropod.jpg'], ['data/koala2.jpg'],['data/cat.jpg'],['data/BrushtailPossum.jpg']])
    
    detect_button = gr.Button('Detect')
    
    detect_button.click(yolo, inputs=[options,img_input], outputs=img_output, queue=True)
    example_images.click(fn=set_example_image, inputs = [example_images], outputs=[img_input])
    
    gr.Markdown(article)
 
demo.launch(enable_queue=True)
