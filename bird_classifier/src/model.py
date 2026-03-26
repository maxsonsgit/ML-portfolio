import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


def create_model(params, device):
    """Создает и настраивает модель ResNet152"""
    
    if params['model']['pretrained']:
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
    else:
        model = resnet152(weights=None)
    
    if params['model']['freeze_backbone']:
        model.requires_grad_(False)
    
    num_features = model.fc.in_features
    hidden_size = params['model']['fc_hidden_size']
    num_classes = params['model']['num_classes']
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_size, num_classes),
    )
    
    model.fc.requires_grad_(True)
    model = model.to(device)
    
    print(f"Модель создана: {params['model']['name']}")
    print(f"Параметров для обучения: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def get_embedding_model(model):
    """Возвращает модель без последнего FC слоя для получения эмбедингов"""
    return nn.Sequential(*list(model.children())[:-1])


def save_model(model, optimizer, epoch, metrics, path):
    """Сохраняет модель и информацию об обучении"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    print(f"Модель сохранена: {path}")


def load_model(params, path, device):
    """Загружает обученную модель"""
    model = create_model(params, device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint
