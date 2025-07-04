import gradio as gr
from transformers import pipeline
from PIL import Image, ImageDraw

# Load object detection pipeline
detector = pipeline("object-detection")

# Function to run detection and draw boxes
def detect_objects(image):
    # Run detection
    predictions = detector(image)

    # Draw results
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        box = pred['box']
        label = pred['label']
        score = pred['score']
        draw.rectangle(
            [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
            outline='red',
            width=3
        )
        draw.text((box['xmin'], box['ymin'] - 10), f"{label} ({score:.2f})", fill='red')
    
    return image

# Gradio Interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object Detection with Hugging Face",
    description="Upload an image and detect objects using a pretrained model from Hugging Face Transformers."
)

# Launch the app
interface.launch()
