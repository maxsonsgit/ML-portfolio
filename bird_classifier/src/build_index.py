import os
import yaml
import json
import torch
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets

from model import load_model, get_embedding_model
from data import get_inference_transforms


def extract_embeddings(model, dataset, device, batch_size=32):
    """Извлекает эмбединги для всех изображений"""
    model.eval()
    embeddings = []
    image_paths = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            emb = model(images)
            emb = emb.squeeze()
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    # Получаем пути к изображениям
    for path, _ in dataset.samples:
        image_paths.append(path)
    
    return embeddings, image_paths


def build_faiss_index(embeddings, index_type='L2'):
    """Строит FAISS индекс"""
    dimension = embeddings.shape[1]
    
    if index_type == 'L2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IP':
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    index.add(embeddings.astype('float32'))
    
    return index


def main():
    """Построение FAISS индекса"""
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    Path(params['output']['faiss_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['output']['embeddings_dir']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    print("\nЗагрузка модели...")
    model_path = os.path.join(params['output']['model_dir'], 'best_model.pth')
    model, _ = load_model(params, model_path, device)
    
    # Получаем модель для эмбедингов (без FC слоя)
    embedding_model = get_embedding_model(model)
    embedding_model.eval()
    
    print("\nЗагрузка датасета...")
    transform = get_inference_transforms(params['inference']['image_size'])
    dataset = datasets.ImageFolder(params['data']['raw_dir'], transform=transform)
    
    print(f"Всего изображений: {len(dataset)}")
    
    print("\nИзвлечение эмбедингов...")
    embeddings, image_paths = extract_embeddings(
        embedding_model, dataset, device, 
        batch_size=params['evaluation']['batch_size']
    )
    
    print(f"Размер эмбедингов: {embeddings.shape}")
    
    # Сохраняем эмбединги
    embeddings_path = os.path.join(params['output']['embeddings_dir'], 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"Эмбединги сохранены: {embeddings_path}")
    
    print("\nПостроение FAISS индекса...")
    index = build_faiss_index(embeddings, params['faiss']['index_type'])
    
    # Сохраняем индекс
    index_path = os.path.join(params['output']['faiss_dir'], 'embeddings.faiss')
    faiss.write_index(index, index_path)
    print(f"FAISS индекс сохранен: {index_path}")
    
    # Сохраняем пути к изображениям
    paths_data = {
        'image_paths': image_paths,
        'class_names': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }
    
    paths_path = os.path.join(params['output']['faiss_dir'], 'image_paths.json')
    with open(paths_path, 'w') as f:
        json.dump(paths_data, f, indent=2)
    print(f"Пути к изображениям сохранены: {paths_path}")
    
    print("\n" + "=" * 80)
    print("FAISS ИНДЕКС ПОСТРОЕН!")
    print(f"Индексировано изображений: {len(image_paths)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
