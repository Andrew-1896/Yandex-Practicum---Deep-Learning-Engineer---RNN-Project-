import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
from generate_with_gpt2 import generate_with_gpt2

def find_best_generation_params(model, dataloader, tokenizer, device, num_test_samples=500):
  
    # подбираем оптимальные параметры генерации на основе валидационных данных
   
    print("\n" + "="*60)
    print("ПОДБОР ПАРАМЕТРОВ ГЕНЕРАЦИИ")
    print("="*60)
    
    # собираем тестовые примеры из валидационного датасета
    test_examples = []
    with torch.no_grad():
        for batch_idx, (context, _) in enumerate(dataloader):
            if len(test_examples) >= num_test_samples:
                break
            
            for i in range(context.size(0)):
                if len(test_examples) >= num_test_samples:
                    break
                
                # декодируем контекст в текст
                full_tokens = context[i].tolist()
                full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
                words = full_text.split()
                
                if len(words) >= 8:
                    split_idx = int(len(words) * 0.75)
                    context_words = ' '.join(words[:split_idx])
                    target_words = ' '.join(words[split_idx:split_idx + 8])
                    
                    if context_words and target_words:
                        test_examples.append({
                            'context': context_words,
                            'target': target_words
                        })
    
    print(f"Собрано {len(test_examples)} тестовых примеров из валидационной выборки")
    
    # различные комбинации параметров для тестирования
    param_combinations = [
        {"temperature": 0.6, "top_k": 30, "top_p": 0.85},  # консервативный
        {"temperature": 0.7, "top_k": 40, "top_p": 0.90},  # умеренно-консервативный
        {"temperature": 0.7, "top_k": 50, "top_p": 0.95},  # умеренный
        {"temperature": 0.8, "top_k": 50, "top_p": 0.92},  # умеренно-креативный
        {"temperature": 0.8, "top_k": 60, "top_p": 0.95},  # креативный
        {"temperature": 0.9, "top_k": 70, "top_p": 0.98},  # очень креативный
    ]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    best_score = 0
    best_params = param_combinations[2]  # на случай, если best_params окажется неопределённым после всех процедур (умеренный по умолчанию)
    
    print("\nОценка различных комбинаций параметров...")
    
    for params in param_combinations:
        rouge1_scores = []
        
        for example in tqdm(test_examples, desc=f"temp={params['temperature']:.1f}"):
            generated = generate_with_gpt2(
                model, tokenizer, example['context'],
                max_new_tokens=8,
                temperature=params['temperature'],
                top_k=params['top_k'],
                top_p=params['top_p'], 
                device=device
            )
            
            if generated:
                scores = scorer.score(example['target'], generated)
                rouge1_scores.append(scores['rouge1'].fmeasure)
        
        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
        print(f"  ROUGE-1: {avg_rouge1:.4f}")
        
        if avg_rouge1 > best_score:
            best_score = avg_rouge1
            best_params = params
    
    print("\n" + "="*60)
    print(f"ЛУЧШИЕ ПАРАМЕТРЫ (ROUGE-1: {best_score:.4f}):")
    print(f"  temperature = {best_params['temperature']}")
    print(f"  top_k = {best_params['top_k']}")
    print(f"  top_p = {best_params['top_p']}")
    print("="*60)
    
    return best_params