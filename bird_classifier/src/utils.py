import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch


class MetricsTracker:
    """Класс для отслеживания метрик во время обучения"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сбрасывает все метрики"""
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = -1
        self.epochs_without_improvement = 0
    
    def update(self, train_loss, train_acc, val_loss, val_acc, epoch):
        """Обновляет метрики (epoch начинается с 0)"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch  # Сохраняем 0-based индекс
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def save_metrics(self, path):
        """Сохраняет метрики в JSON"""
        metrics = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch + 1,  # +1 для человекочитаемого формата
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_train_acc': self.train_accs[-1] if self.train_accs else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'final_val_acc': self.val_accs[-1] if self.val_accs else 0,
        }
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Метрики сохранены: {path}")
    
    def save_training_curves_csv(self, path):
        """Сохраняет данные для DVC plots в CSV"""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
            for i in range(len(self.train_losses)):
                writer.writerow([
                    i + 1,  # Эпоха начинается с 1 для отображения
                    self.train_losses[i],
                    self.val_losses[i],
                    self.train_accs[i],
                    self.val_accs[i]
                ])
        print(f"Данные для графиков сохранены: {path}")


def plot_training_curves_png(metrics_tracker, save_path):
    """Строит PNG графики обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(metrics_tracker.train_losses) + 1)
    
    # График Loss
    axes[0].plot(epochs, metrics_tracker.train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, metrics_tracker.val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # График Accuracy
    axes[1].plot(epochs, metrics_tracker.train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, metrics_tracker.val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].axhline(y=metrics_tracker.best_val_acc, color='g', linestyle='--', 
                    label=f'Best Val Acc: {metrics_tracker.best_val_acc:.2f}% (Epoch {metrics_tracker.best_epoch + 1})', 
                    linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PNG графики обучения сохранены: {save_path}")


def save_confusion_matrix_csv(all_targets, all_preds, class_names, save_path):
    """Сохраняет confusion matrix в CSV формате для DVC"""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual', 'predicted'])
        for true, pred in zip(all_targets, all_preds):
            writer.writerow([class_names[true], class_names[pred]])
    print(f"Confusion matrix CSV сохранена: {save_path}")


def plot_confusion_matrix_png(cm, class_names, save_path):
    """Строит PNG confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PNG Confusion matrix сохранена: {save_path}")


def save_classification_report_json(all_targets, all_preds, class_names, save_path):
    """Сохраняет classification report в JSON"""
    report = classification_report(all_targets, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Classification report сохранен: {save_path}")
    return report


def save_per_class_metrics_csv(report, class_names, save_path):
    """Сохраняет метрики по классам в CSV для DVC"""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1_score', 'support'])
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                writer.writerow([
                    class_name,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1-score'],
                    metrics['support']
                ])
    print(f"Per-class metrics CSV сохранены: {save_path}")


def evaluate_model(model, data_loader, device):
    """Оценивает модель и возвращает предсказания"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    return all_targets, all_preds
