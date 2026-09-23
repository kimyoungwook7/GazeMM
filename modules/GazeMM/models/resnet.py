import torch.nn as nn
from torch.nn import functional as F
from .utils_resnet import resnet34


class Resnet34Triplet(nn.Module):

    def __init__(self, embedding_dimension=512, pretrained=False):
        super(Resnet34Triplet, self).__init__()
        self.model = resnet34(pretrained=pretrained)

        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension, bias=False)

    def forward(self, images):
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
