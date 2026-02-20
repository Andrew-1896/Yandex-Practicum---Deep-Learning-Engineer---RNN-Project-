import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random

class NextTokenDataset(Dataset):
    # датасет для предсказания следующего токена
    def __init__(self, texts, tokenizer, context_len=8):
        self.samples = []
        
        for text in tqdm(texts, desc="Создание датасета"):
            tokens = tokenizer.encode(text, add_special_tokens=False, 
                                      max_length=50, truncation=True)
            
            if len(tokens) > context_len + 2:
                # берем только несколько случайных позиций для ускорения
                num_samples = min(3, len(tokens) - context_len)
                positions = random.sample(range(context_len, len(tokens)), num_samples)
                
                for i in positions:
                    context = tokens[i-context_len:i]
                    target = tokens[i]
                    self.samples.append((context, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context), torch.tensor(target)