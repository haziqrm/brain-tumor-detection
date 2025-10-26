import torch
import torch.nn as nn
import torchvision.models as models

class AnomalyDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_name='resnet18'):
        super(AnomalyDetector, self).__init__()
        self.model_name = model_name

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x):
        return self.backbone(x)

def create_model(num_classes=2, model_name='resnet18', device='cuda'):
    model = AnomalyDetector(num_classes=num_classes,
                            pretrained=True,
                            model_name=model_name)
    model = model.to(device)

    print(f"Created {model_name} model with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model