from fastai.vision.all import load_learner
from fastai.vision.all import *
import gradio as gr
import pathlib
import torch
import torchvision.transforms as T
from PIL import Image

# Fix pathlib for cross-platform compatibility
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

trained_model = load_learner("model/safety_densenet121_model_Version_1.pkl")

categories = [
"Bangladesh Cracked or Damaged Asphalt Road",
"Bangladesh Crowded Pedestrian Footpath",
"Bangladesh Flooded Road",
"Bangladesh Pedestrians Crossing Busy Roads",
"Bangladesh Roads Potholes",
"Bus Stopping in Road",
"CNG-Autorickshaw in Traffic",
"Illegal Parking on Roadside in Bangladesh",
"Overloaded Rickshaw or Van"
]


def road_condition(image):
    img = Image.open(image).convert("RGB") if isinstance(image, str) else image

    tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    x = tfms(img).unsqueeze(0)
    trained_model.model.eval()

    with torch.no_grad():
        probs = torch.softmax(trained_model.model(x), dim=-1)[0]

    return dict(zip(categories, map(float, probs)))

category_image = gr.Image()
category_label = gr.Label()
examples = [
    'test_images/image_1.jpg',
    'test_images/image_2.jpg'
]


with gr.Blocks() as iface:
    #gr.Markdown("### Image Classifier")
    category_image = gr.Image(type="pil")
    category_label = gr.Label()
    btn = gr.Button("Predict")

    btn.click(fn=road_condition, inputs=category_image, outputs=category_label)

    gr.Examples(
        examples=[
            'test_images/image_1.jpg',
            'test_images/image_2.jpg'
        ],
        inputs=category_image
    )

iface.launch(inline=False, share=True)