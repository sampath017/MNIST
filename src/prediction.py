from pathlib import Path
import gradio as gr
from lightning.pytorch import loggers
from pathlib import Path

import torch
from lightning.pytorch import loggers
import torchvision.transforms.functional as v_F
import torch.nn.functional as F

from model import Digits
from PIL import Image

root_path = Path('../')

artifact_dir = root_path / 'artifacts'
artifact_dir.mkdir(exist_ok=True)

model_path = Path(loggers.WandbLogger.download_artifact(
    artifact="sampath017/MNIST/model-74c3j2pz:v12",
    artifact_type='model',
    save_dir=artifact_dir
))  # type: ignore

model_path = model_path / 'model.ckpt'


model = Digits.load_from_checkpoint(
    model_path, map_location=torch.device('cpu'))


def predict(image_obj):
    image = image_obj.get("composite")
    resized_img = image.resize(size=(28, 28))

    data = v_F.to_tensor(resized_img)

    logits = model.predict_step(data, 0)
    probs = F.softmax(logits, dim=-1).item()

    return probs


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
