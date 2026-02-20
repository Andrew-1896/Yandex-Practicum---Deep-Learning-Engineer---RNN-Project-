from rouge_score import rouge_scorer
from generate_with_gpt2 import generate_with_gpt2

def show_gpt2_predictions(model, tokenizer, device, dataset_processed, num_examples=5, params=None):
    # показываем примеры предсказаний модели GPT-2
    
    if params is None:
        params = {"temperature": 0.7, "top_k": 50, "top_p": 0.95}
    
    print("\n" + "="*80)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ МОДЕЛИ DISTILGPT2")
    print("="*80)
    
    sample_texts = dataset_processed['tweet'].sample(min(num_examples, len(dataset_processed))).tolist()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for i, text in enumerate(sample_texts):
        words = text.split()
        if len(words) < 5:
            continue
        
        split_idx = int(len(words) * 0.75)
        context = ' '.join(words[:split_idx])
        target = ' '.join(words[split_idx:split_idx + 8])
        
        if not target:
            continue
        
        generated = generate_with_gpt2(
            model, tokenizer, context,
            max_new_tokens=min(8, len(words) - split_idx),
            **params, device=device
        )
        
        print(f"\nПример {i+1}:")
        print(f"  Контекст: '{context}'")
        print(f"  Оригинал: '{target}'")
        print(f"  Генерация: '{generated}'")
        
        if generated and target:
            scores = scorer.score(target, generated)
            print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        print("-" * 60)