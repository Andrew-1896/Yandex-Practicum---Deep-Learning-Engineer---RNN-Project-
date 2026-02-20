import numpy as np
from rouge_score import rouge_scorer
import torch
from generate_with_gpt2 import generate_with_gpt2

def compare_model_predictions(lstm_model, gpt2_model, lstm_tokenizer, gpt2_tokenizer, 
                              device, dataset_processed, num_examples=5, gpt2_params=None):
    # сравниваем предсказания LSTM и GPT-2 моделей на одних и тех же примерах

    if gpt2_params is None:
        gpt2_params = {"temperature": 0.7, "top_k": 50, "top_p": 0.95}
    
    lstm_model.eval()
    gpt2_model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    sample_texts = dataset_processed['tweet'].sample(min(num_examples, len(dataset_processed))).tolist()
    
    print("\n" + "="*100)
    print("СРАВНЕНИЕ ПРЕДСКАЗАНИЙ МОДЕЛЕЙ LSTM И DISTILGPT2")
    print("="*100)
    
    lstm_rouge1, lstm_rouge2, lstm_rougeL = [], [], []
    gpt2_rouge1, gpt2_rouge2, gpt2_rougeL = [], [], []
    
    for i, text in enumerate(sample_texts):
        words = text.split()
        if len(words) < 5:
            continue
        
        split_idx = int(len(words) * 0.75)
        context = ' '.join(words[:split_idx])
        target = ' '.join(words[split_idx:split_idx + 8])
        
        if not target:
            continue
        
        # генерация LSTM
        lstm_context_tokens = lstm_tokenizer.encode(context, add_special_tokens=False)
        lstm_generated = lstm_model.generate(lstm_context_tokens, lstm_tokenizer, max_gen_len=5)
        
        # генерация GPT-2
        max_gen = min(8, len(words) - split_idx)
        gpt2_generated = generate_with_gpt2(
            gpt2_model, gpt2_tokenizer, context,
            max_new_tokens=max_gen,
            **gpt2_params, device=device
        )
        
        # вычисляем метрики для обеих моделей
        lstm_scores = scorer.score(target, lstm_generated) if lstm_generated else None
        gpt2_scores = scorer.score(target, gpt2_generated) if gpt2_generated else None
        
        print(f"\nПример {i+1}:")
        print(f"  Контекст: '{context}'")
        print(f"  Оригинал: '{target}'")
        print(f"\n  LSTM генерация:     '{lstm_generated}'")
        if lstm_scores:
            print(f"    ROUGE-1: {lstm_scores['rouge1'].fmeasure:.4f}")
            print(f"    ROUGE-2: {lstm_scores['rouge2'].fmeasure:.4f}")
            print(f"    ROUGE-L: {lstm_scores['rougeL'].fmeasure:.4f}")
            lstm_rouge1.append(lstm_scores['rouge1'].fmeasure)
            lstm_rouge2.append(lstm_scores['rouge2'].fmeasure)
            lstm_rougeL.append(lstm_scores['rougeL'].fmeasure)
        
        print(f"\n  DistilGPT2 генерация: '{gpt2_generated}'")
        if gpt2_scores:
            print(f"    ROUGE-1: {gpt2_scores['rouge1'].fmeasure:.4f}")
            print(f"    ROUGE-2: {gpt2_scores['rouge2'].fmeasure:.4f}")
            print(f"    ROUGE-L: {gpt2_scores['rougeL'].fmeasure:.4f}")
            gpt2_rouge1.append(gpt2_scores['rouge1'].fmeasure)
            gpt2_rouge2.append(gpt2_scores['rouge2'].fmeasure)
            gpt2_rougeL.append(gpt2_scores['rougeL'].fmeasure)
        print("-" * 80)
