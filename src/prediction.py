from pathlib import Path
import gradio as gr

import torch
from lightning.pytorch import loggers
import torchvision.transforms.functional as v_F
import torch.nn.functional as F

from model import MNIST
from PIL import Image

root_path = Path('../')

artifact_dir = root_path / 'artifacts'
artifact_dir.mkdir(exist_ok=True)

model_path = Path(loggers.WandbLogger.download_artifact(  # type: ignore
    artifact="sampath017/model-registry/MNIST:v2",
    artifact_type='model',
    save_dir=artifact_dir
))

model_path = model_path / 'model.ckpt'


model = MNIST.load_from_checkpoint(
    model_path, map_location=torch.device('cpu'))


def predict(image_obj):
    image = image_obj.get("composite")
    resized_img = image.resize(size=(28, 28))

    data = v_F.to_tensor(resized_img).unsqueeze(0)

    logits = model.predict_step(data, 0)
    probs = F.softmax(logits, dim=-1)[0].tolist()

    confidences = {i: p for i, p in enumerate(probs)}

    return confidences


demo = gr.Interface(
    fn=predict,
    inputs=gr.ImageEditor(
        value=Image.new(mode="L", size=(500, 500), color="black"),
        image_mode="L",
        type="pil",
        brush=gr.Brush(colors=["white"]),
    ),
    outputs=gr.Label(num_top_classes=3)
)

demo.launch()
