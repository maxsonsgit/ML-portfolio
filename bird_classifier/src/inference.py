import os
import yaml
import json
import torch
import faiss
import numpy as np
import gradio as gr
from PIL import Image

from model import load_model, get_embedding_model
from data import get_inference_transforms


class BirdClassifier:
    """Классификатор птиц с поиском похожих изображений"""
    
    def __init__(self, params):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем модель
        model_path = params['inference']['model_path']
        self.model, _ = load_model(params, model_path, self.device)
        self.model.eval()
        
        # Модель для эмбедингов
        self.embedding_model = get_embedding_model(self.model)
        self.embedding_model.eval()
        
        # Трансформации
        self.transform = get_inference_transforms(params['inference']['image_size'])
        
        # FAISS индекс
        faiss_path = os.path.join(params['output']['faiss_dir'], 'embeddings.faiss')
        self.index = faiss.read_index(faiss_path)
        
        # Пути к изображениям
        paths_path = os.path.join(params['output']['faiss_dir'], 'image_paths.json')
        with open(paths_path, 'r') as f:
            data = json.load(f)
            self.image_paths = data['image_paths']
            self.class_names = data['class_names']
        
        print(f"Модель загружена на {self.device}")
        print(f"Классы: {self.class_names}")
        print(f"FAISS индекс: {self.index.ntotal} изображений")
    
    def classify_image(self, image):
        """Классифицирует изображение"""
        if image is None:
            return None, []
        
        # Преобразуем изображение
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Классификация
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Формируем результат для Gradio
        result = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        return result
    
    def find_similar(self, image, top_k=10):
        """Находит похожие изображения"""
        if image is None:
            return []
        
        # Преобразуем изображение
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Получаем эмбединг
        with torch.no_grad():
            embedding = self.embedding_model(img_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Нормализуем для косинусного сходства если нужно
        if self.params['faiss']['index_type'] == 'IP':
            embedding = embedding / np.linalg.norm(embedding)
        
        # Поиск похожих
        embedding = embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(embedding, top_k)
        
        # Формируем результат
        similar_images = []
        for idx, dist in zip(indices[0], distances[0]):
            img_path = self.image_paths[idx]
            similar_images.append((img_path, f"Distance: {1 - dist:.4f}"))
        
        return similar_images
    
    def process_image(self, image):
        """Обрабатывает изображение: классификация + поиск похожих"""
        if image is None:
            return None, []
        
        # Классификация
        classification = self.classify_image(image)
        
        # Поиск похожих
        similar = self.find_similar(image, self.params['inference']['top_k'])
        
        return classification, similar


def create_app(classifier):
    """Создает Gradio приложение"""
    
    with gr.Blocks(title="Bird Classifier with Similar Images Search") as app:
        gr.Markdown("# Классификатор птиц с поиском похожих изображений")
        gr.Markdown("Загрузите изображение птицы для классификации и поиска похожих изображений в датасете")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Загрузите изображение")
                classify_btn = gr.Button("Классифицировать и найти похожие", variant="primary")
            
            with gr.Column(scale=1):
                classification_output = gr.Label(num_top_classes=5, label="Классификация")
        
        gr.Markdown("## Топ-10 похожих изображений")
        similar_output = gr.Gallery(
            label="Похожие изображения",
            show_label=True,
            elem_id="gallery",
            columns=5,
            rows=2,
            height="auto"
        )
        
        classify_btn.click(
            fn=classifier.process_image,
            inputs=input_image,
            outputs=[classification_output, similar_output]
        )
            
    return app


def main():
    """Запуск веб-приложения"""
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    print("Инициализация классификатора...")
    classifier = BirdClassifier(params)
    
    print("\nСоздание веб-приложения...")
    app = create_app(classifier)
    
    print(f"\nЗапуск на {params['inference']['host']}:{params['inference']['port']}")
    app.launch(
        server_name=params['inference']['host'],
        server_port=params['inference']['port'],
        share=False
    )


if __name__ == '__main__':
    main()
