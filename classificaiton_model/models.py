import torch
import torch.nn as nn
import torchvision.models as models

class LogisticRegression(nn.Module):
    """Simple logistic regression model"""
    def __init__(self, input_size=224*224*3):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        x = self.linear(x)
        return x

def get_model(model_name, pretrained=True):
    """Get the specified model architecture"""
    if model_name == 'logistic':
        model = LogisticRegression()
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    
    elif model_name == 'shufflenetv2':
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    # elif model_name == 'inceptionv3':
    #     model = models.inception_v3(pretrained=pretrained)
    #     model.fc = nn.Linear(model.fc.in_features, 2)
    #     # InceptionV3 has an auxiliary classifier that needs to be modified as well
    #     if model.AuxLogits is not None:
    #         model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Add softmax layer to generate class probabilities
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    
    return model