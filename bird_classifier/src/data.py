import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_augmented_transforms(params):
    """Создает расширенные трансформации для аугментации"""
    aug_list = [
        transforms.RandomResizedCrop(params['augmentation']['random_resized_crop'])
    ]
    
    if params['augmentation']['random_horizontal_flip']:
        aug_list.append(transforms.RandomHorizontalFlip())
    
    if params['augmentation']['random_vertical_flip']:
        aug_list.append(transforms.RandomVerticalFlip())
    
    if params['augmentation']['random_rotation'] > 0:
        aug_list.append(
            transforms.RandomRotation(params['augmentation']['random_rotation'])
        )
    
    if params['augmentation']['color_jitter']:
        cj = params['augmentation']['color_jitter']
        aug_list.append(
            transforms.ColorJitter(
                brightness=cj['brightness'],
                contrast=cj['contrast'],
                saturation=cj['saturation'],
                hue=cj['hue']
            )
        )
    
    aug_list.extend([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(aug_list)


def get_val_transforms(params):
    """Создает трансформации для валидации"""
    crop_size = params['augmentation']['random_resized_crop']
    return transforms.Compose([
        transforms.Resize(crop_size + 10),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_inference_transforms(image_size):
    """Трансформации для инференса"""
    return transforms.Compose([
        transforms.Resize(image_size + 10),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_dataloaders(params, for_evaluation=False):
    """Создает DataLoader для обучения и валидации"""
    data_dir = params['data']['raw_dir']
    
    full_dataset = datasets.ImageFolder(data_dir)
    
    if not for_evaluation:
        print("Анализ датасета...")
        analyze_dataset_images(full_dataset)
    
    train_size = int(params['data']['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(params['data']['random_seed'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    if for_evaluation:
        batch_size = params['evaluation']['batch_size']
        val_transform = get_val_transforms(params)
        val_dataset.dataset.transform = val_transform
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=params['data']['num_workers'],
            pin_memory=True
        )
        
        return val_loader, full_dataset.classes
    
    train_transform = get_augmented_transforms(params) if params['augmentation']['enable'] else get_val_transforms(params)
    val_transform = get_val_transforms(params)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['training']['batch_size'],
        shuffle=True,
        num_workers=params['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['training']['batch_size'],
        shuffle=False,
        num_workers=params['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    
    return train_loader, val_loader, full_dataset.classes


def analyze_dataset_images(dataset):
    """Анализирует размеры изображений в датасете"""
    avg_w, avg_h = 0, 0
    min_w, min_h = 10000, 10000
    max_w, max_h = 0, 0
    
    for img, _ in dataset:
        w, h = img.size
        avg_w += w
        avg_h += h
        min_w = min(min_w, w)
        min_h = min(min_h, h)
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    
    n = len(dataset)
    print(f"Средний размер: {avg_w/n:.0f}x{avg_h/n:.0f}")
    print(f"Минимальный размер: {min_w}x{min_h}")
    print(f"Максимальный размер: {max_w}x{max_h}")
