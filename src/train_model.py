import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from compute_rouge_metrics import compute_rouge_metrics
from plot_training_progress import plot_training_progress
from show_generation_examples import show_generation_examples

def train_model(model, train_loader, val_loader, tokenizer, device, dataset_processed,
                epochs=5, lr=0.001, eval_every=1):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                             steps_per_epoch=len(train_loader),
                                             epochs=epochs)
    
    train_losses, val_losses = [], []
    rouge_history = []
    best_val_loss = float('inf')
    
    print("Начинаем обучение...")
    print(f"Всего эпох: {epochs}")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        # тренировка
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
        for context, target in train_pbar:
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # валидация
        model.eval()
        total_val_loss = 0
        num_val_batches = min(50, len(val_loader))
        
        with torch.no_grad():
            for i, (context, target) in enumerate(val_loader):
                if i >= num_val_batches:
                    break
                context, target = context.to(device), target.to(device)
                logits = model(context)
                loss = criterion(logits, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # вычисляем ROUGE метрики
        if epoch % eval_every == 0:
            print("Вычисление ROUGE метрик...")
            rouge_metrics = compute_rouge_metrics(
                              model=model,
                              dataloader=val_loader,
                              tokenizer=tokenizer,  
                              device=device,
                              model_type='lstm',
                              context_ratio=0.75,
                              max_gen_len=5
                              )
            rouge_history.append(rouge_metrics)
            print(f"  ROUGE-1: {rouge_metrics['rouge1']:.4f}")
            print(f"  ROUGE-2: {rouge_metrics['rouge2']:.4f}")
            print(f"  ROUGE-L: {rouge_metrics['rougeL']:.4f}")
        
        # сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Сохранена лучшая модель (val_loss: {avg_val_loss:.4f})")
        
        # рисуем графики после каждой эпохи
        plot_training_progress(train_losses, val_losses, rouge_history, epoch)
        
        # показываем примеры на последней эпохе
        if epoch == epochs:
            show_generation_examples(model, tokenizer, device, dataset_processed, num_examples=5)
        
        print("-" * 60)
    
    # загружаем лучшую модель
    model.load_state_dict(torch.load('best_model.pth'))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rouge_history': rouge_history,
        'best_val_loss': best_val_loss
    }