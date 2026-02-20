import torch

def generate_with_gpt2(model, tokenizer, context_text, max_new_tokens, 
                       temperature, top_k, top_p, device):
    
    # генерируем продолжение текста с помощью GPT-2
    
    # Args:
        # model: модель GPT-2
        # tokenizer: токенизатор
        # context_text: входной текст
        # max_new_tokens: максимальное число новых токенов
        # temperature: температура (контролирует случайность)
        # top_k: top-k сэмплинг
        # top_p: top-p (nucleus) сэмплинг

    model.eval()
    
    inputs = tokenizer(context_text, return_tensors="pt", truncation=True, 
                       max_length=100).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text