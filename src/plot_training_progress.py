import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_training_progress(train_losses, val_losses, rouge_history, epochs):
    # рисуем графики loss и ROUGE метрик 
    clear_output(wait=True)
    
    # создаем подграфики: 1 строка, 2 колонки
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs_range = range(1, len(train_losses) + 1)
    
    # график loss 
    axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # график ROUGE метрик 
    if rouge_history:
        rouge1_vals = [r['rouge1'] for r in rouge_history]
        rouge2_vals = [r['rouge2'] for r in rouge_history]
        rougeL_vals = [r['rougeL'] for r in rouge_history]
        
        axes[1].plot(epochs_range[:len(rouge1_vals)], rouge1_vals, 'g-', label='ROUGE-1', linewidth=2, marker='o')
        axes[1].plot(epochs_range[:len(rouge2_vals)], rouge2_vals, 'y-', label='ROUGE-2', linewidth=2, marker='s')
        axes[1].plot(epochs_range[:len(rougeL_vals)], rougeL_vals, 'm-', label='ROUGE-L', linewidth=2, marker='^')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ROUGE Score')
        axes[1].set_title('ROUGE Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
    else:
        # если нет ROUGE метрик, показываем сообщение
        axes[1, 0].text(0.5, 0.5, 'No ROUGE data available', 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=axes[1, 0].transAxes,
                       fontsize=12)
        axes[1, 0].set_title('ROUGE Metrics')
        axes[1, 0].axis('on')
    
 
    # удаляем пустое пространство между графиками
    plt.tight_layout()
    
    
    plt.show()