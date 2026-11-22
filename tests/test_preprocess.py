from PIL import Image, ImageDraw
from src.preprocess import image_to_tensor

def test_preprocessing():
    image = Image.new("RGB", (100, 100), "black")
    ImageDraw.Draw(image).line((20, 20, 80, 80), fill="white", width=8)
    tensor = image_to_tensor(image)
    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert tensor.isfinite().all()
