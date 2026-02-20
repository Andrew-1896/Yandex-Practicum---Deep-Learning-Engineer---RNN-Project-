import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
from generate_with_gpt2 import generate_with_gpt2

def compute_rouge_metrics(model, dataloader, tokenizer, device, 
                          gpt2_tokenizer=None, generation_params=None, 
                          model_type='lstm', max_examples=200,
                          context_ratio=0.75, max_gen_len=10):
    
    # универсальная функция для вычисления ROUGE метрик
    # Args:
        # model: модель для оценки
        # dataloader: DataLoader с данными
        # tokenizer: BERT токенизатор для декодирования (для всех моделей)
        # device: устройство
        # gpt2_tokenizer: токенизатор для GPT-2 (обязателен при model_type='gpt2')
        # generation_params: словарь с параметрами генерации (для GPT-2)
        # model_type: 'lstm' или 'gpt2'
        # max_examples: максимальное количество примеров
        # context_ratio: доля текста для контекста
        # max_gen_len: максимальная длина генерации
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    examples_processed = 0
    total_batches = len(dataloader)
    
    print(f"\nВычисление ROUGE метрик для модели {model_type.upper()}")
    print(f"Контекст: {int(context_ratio*100)}%, генерация: {int((1-context_ratio)*100)}%")
    print(f"Будет обработано примеров: {max_examples if max_examples else 'все'}")
    
    with torch.no_grad():
        for batch_idx, (context, _) in enumerate(tqdm(dataloader, desc=f"Оценка {model_type.upper()}")):
            batch_tokens = context.cpu().tolist()
            
            for i in range(len(batch_tokens)):
                if max_examples is not None and examples_processed >= max_examples:
                    break
                
                full_seq = batch_tokens[i]
                # ВСЕГДА используем tokenizer для декодирования контекста
                full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
                words = full_text.split()
                
                if len(words) < 5:
                    continue
                
                split_idx = int(len(words) * context_ratio)
                if split_idx >= len(words) or split_idx < 2:
                    continue
                
                context_words = ' '.join(words[:split_idx])
                target_words = ' '.join(words[split_idx:split_idx + max_gen_len])
                
                if len(target_words.split()) == 0:
                    continue
                
                if model_type == 'lstm':
                    # Для LSTM используем tokenizer для кодирования контекста
                    context_tokens = tokenizer.encode(context_words, add_special_tokens=False)
                    generated_text = model.generate(context_tokens, tokenizer, max_gen_len=5)
                else:  # gpt2
                    if generation_params is None:
                        raise ValueError("Для GPT-2 необходимо передать generation_params")
                    if gpt2_tokenizer is None:
                        raise ValueError("Для GPT-2 необходимо передать gpt2_tokenizer")
                    
                    max_gen = min(max_gen_len, len(words) - split_idx)
                    # Для GPT-2 используем gpt2_tokenizer для генерации
                    generated_text = generate_with_gpt2(
                        model, gpt2_tokenizer, context_words, 
                        max_new_tokens=max_gen,
                        device=device,
                        **generation_params
                    )
                
                if generated_text and target_words:
                    scores = scorer.score(target_words, generated_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                    examples_processed += 1
            
            if max_examples is not None and examples_processed >= max_examples:
                break
    
    final_metrics = {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0,
        'num_samples': examples_processed
    }
    
    print(f"\n" + "="*50)
    print(f"РЕЗУЛЬТАТЫ ДЛЯ МОДЕЛИ {model_type.upper()}:")
    print(f"  Обработано примеров: {final_metrics['num_samples']}")
    print(f"  ROUGE-1: {final_metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {final_metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {final_metrics['rougeL']:.4f}")
    print("="*50)
    
    return final_metrics