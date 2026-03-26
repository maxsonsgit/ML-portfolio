import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from data import get_dataloaders
from model import create_model, save_model
from utils import MetricsTracker, plot_training_curves_png


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Обучение на одной эпохе"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 9:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}: '
                  f'Loss {running_loss/(batch_idx+1):.3f}, '
                  f'Acc {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Валидация модели"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = val_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """Основная функция обучения"""
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    Path(params['output']['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['output']['metrics_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['output']['plots_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['output']['outputs_dir']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    print("\nЗагрузка данных...")
    train_loader, val_loader, class_names = get_dataloaders(params)
    
    print("\nСоздание модели...")
    model = create_model(params, device)
    
    if params['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['training']['learning_rate'],
            weight_decay=params['training']['weight_decay']
        )
    elif params['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['training']['learning_rate'],
            momentum=0.9,
            weight_decay=params['training']['weight_decay']
        )
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    print(f"Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=2)")
    
    metrics_tracker = MetricsTracker()
    
    print(f"\nНачало обучения на {params['training']['num_epochs']} эпох...")
    print("=" * 80)
    
    for epoch in range(params['training']['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        is_best = metrics_tracker.update(train_loss, train_acc, val_loss, val_acc, epoch)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f'Learning rate изменен: {old_lr:.6f} -> {new_lr:.6f}')
        
        print(f'\nEpoch {epoch+1}/{params["training"]["num_epochs"]} Summary:')
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {metrics_tracker.best_val_acc:.2f}% (Epoch {metrics_tracker.best_epoch+1})')
        print("=" * 80)
        
        if is_best and params['training']['save_best']:
            model_path = os.path.join(params['output']['model_dir'], 'best_model.pth')
            save_model(model, optimizer, epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_path)
        
        if params['training']['early_stopping_patience'] > 0:
            if metrics_tracker.epochs_without_improvement >= params['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    final_model_path = os.path.join(params['output']['model_dir'], 'final_model.pth')
    save_model(model, optimizer, epoch, {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, final_model_path)
    
    metrics_path = os.path.join(params['output']['metrics_dir'], 'train_metrics.json')
    metrics_tracker.save_metrics(metrics_path)
    
    csv_path = os.path.join(params['output']['plots_dir'], 'training_curves.csv')
    metrics_tracker.save_training_curves_csv(csv_path)
    
    png_path = os.path.join(params['output']['outputs_dir'], 'training_curves.png')
    plot_training_curves_png(metrics_tracker, png_path)
    
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Лучшая точность на валидации: {metrics_tracker.best_val_acc:.2f}% (Epoch {metrics_tracker.best_epoch+1})")
    print("=" * 80)


if __name__ == '__main__':
    main()
