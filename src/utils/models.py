import torch
import torch.nn as nn

# FeatureExtractor: ResNet18 with feature extraction at multiple stages
# plus an intermediate layer as the previous-to-last one (after last avg pooling)
class FeatureExtractor(nn.Module):
    def __init__(self, base_model, intermediate_size=256):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool

        self.intermediate = nn.Sequential(
            nn.Linear(base_model.fc.in_features, intermediate_size),
            nn.ReLU()
        )

        self.classifier = nn.Linear(intermediate_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features1 = self.maxpool(x)

        features2 = self.layer1(features1)
        features3 = self.layer2(features2)
        features4 = self.layer3(features3)
        features5 = self.layer4(features4)

        pooled = self.avgpool(features5)
        pooled = torch.flatten(pooled, 1)

        interm = self.intermediate(pooled)

        self.features_all = [
            torch.flatten(features1, 1),
            torch.flatten(features2, 1),
            torch.flatten(features3, 1),
            torch.flatten(features4, 1),
            pooled,
            interm
        ]
        out = self.classifier(interm)
        return out


class PatientFeatureExtractor(FeatureExtractor):
    """FeatureExtractor with dropout regularization to reduce overfitting."""
    def __init__(self, base_model, intermediate_size=256, dropout=0.3):
        super().__init__(base_model, intermediate_size)
        self.intermediate = nn.Sequential(
            nn.Linear(base_model.fc.in_features, intermediate_size),
            nn.ReLU(),
            nn.Dropout(p=dropout) # Complica la interpretabilitat (?)
        )
