import torch
import torch.nn as nn
import timm

class CellClassifier(nn.Module):
    def __init__(self, num_classes=2): #Define parts of model
        super(CellClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x): #connect parts
        x = self.features(x)
        output = self.classifier(x)
        return output