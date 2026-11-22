import torch
from src.model import DigitCNN

def test_output_shape():
    model = DigitCNN()
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)
