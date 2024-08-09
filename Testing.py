import torch as torch
from CellClassifier import CellClassifier
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
model_path = os.getenv("MODEL_PATH")
path = os.getenv("IMAGE_PATH")
def process(imagePath):
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])
    image = Image.open(imagePath)
    image = transform(image).unsqueeze(0)
    return image

if __name__ == "__main__":
    model = CellClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #path = "/Users/anushkashome/StreamLit/test/testing_data/C-NMC_test_final_phase_data/2424.bmp" #find image
    image = process(path)

    device = torch.device("cpu")
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    print(predicted.item())



