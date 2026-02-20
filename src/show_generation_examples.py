from rouge_score import rouge_scorer

def show_generation_examples(model, tokenizer, device, dataset_processed, num_examples=5):

    # показываем примеры генерации текста с полными ROUGE метриками
    # модель получает первые 3/4 текста и генерирует оставшуюся 1/4
  
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    sample_texts = dataset_processed['tweet'].sample(min(num_examples, len(dataset_processed))).tolist()
    
    print("\n" + "="*80)
    print("ПРИМЕРЫ ГЕНЕРАЦИИ ТЕКСТА")
    print("="*80)
    print("(контекст: первые 3/4 текста, цель: оставшаяся 1/4)")
    print("="*80)
    
    for i, text in enumerate(sample_texts):
        words = text.split()
        if len(words) < 8:  # нужно минимум для 3/4 и 1/4 - если взяли бы 4, то ROUGE был бы либо 1, либо 0 (нельзя нормально интерпретировать)
            continue
        
        # вычисляем точное разделение 3/4 и 1/4
        split_idx = int(len(words) * 0.75)
        
        context = ' '.join(words[:split_idx])
        target = ' '.join(words[split_idx:])
        
        # определяем длину генерации (количество слов в целевой части)
        target_len = len(words) - split_idx
        
        generated = model.generate(context, tokenizer, max_gen_len=target_len, temperature=0.8)
        
        print(f"\nПример {i+1}:")
        print(f"  Контекст ({len(words[:split_idx])} слов): '{context}'")
        print(f"  Оригинал ({target_len} слов): '{target}'")
        print(f"  Генерация ({len(generated.split())} слов): '{generated}'")
        
        if target and generated:
            scores = scorer.score(target, generated)
            print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        print("-" * 60)