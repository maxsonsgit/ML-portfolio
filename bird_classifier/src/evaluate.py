import os
import yaml
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix

from data import get_dataloaders
from model import load_model
from utils import (
    evaluate_model,
    save_confusion_matrix_csv,
    plot_confusion_matrix_png,
    save_classification_report_json,
    save_per_class_metrics_csv
)


def main():
    """Оценка модели на валидационном датасете"""
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    Path(params['output']['metrics_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['output']['plots_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['output']['outputs_dir']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    print("\nЗагрузка данных...")
    val_loader, class_names = get_dataloaders(params, for_evaluation=True)
    
    print("\nЗагрузка модели...")
    model_path = os.path.join(params['output']['model_dir'], 'best_model.pth')
    model, checkpoint = load_model(params, model_path, device)
    
    print(f"Модель загружена (Epoch {checkpoint['epoch']+1})")
    
    print("\nОценка модели...")
    all_targets, all_preds = evaluate_model(model, val_loader, device)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Сохраняем CSV для DVC
    cm_csv_path = os.path.join(params['output']['plots_dir'], 'confusion_matrix.csv')
    save_confusion_matrix_csv(all_targets, all_preds, class_names, cm_csv_path)
    
    # Сохраняем PNG
    cm_png_path = os.path.join(params['output']['outputs_dir'], 'confusion_matrix.png')
    plot_confusion_matrix_png(cm, class_names, cm_png_path)
    
    # Classification Report
    report_path = os.path.join(params['output']['metrics_dir'], 'classification_report.json')
    report = save_classification_report_json(all_targets, all_preds, class_names, report_path)
    
    # Per-class metrics CSV для DVC
    per_class_csv = os.path.join(params['output']['plots_dir'], 'per_class_metrics.csv')
    save_per_class_metrics_csv(report, class_names, per_class_csv)
    
    # Общие метрики
    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    
    eval_metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_avg['precision'],
        'macro_recall': macro_avg['recall'],
        'macro_f1': macro_avg['f1-score']
    }
    
    import json
    eval_metrics_path = os.path.join(params['output']['metrics_dir'], 'evaluation_metrics.json')
    with open(eval_metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    
    print("\n" + "=" * 80)
    print("ОЦЕНКА ЗАВЕРШЕНА!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_avg['precision']:.4f}")
    print(f"Macro Recall: {macro_avg['recall']:.4f}")
    print(f"Macro F1-Score: {macro_avg['f1-score']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
