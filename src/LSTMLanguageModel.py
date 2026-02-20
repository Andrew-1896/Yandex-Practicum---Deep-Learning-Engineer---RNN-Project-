import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    # LSTM модель для предсказания следующего токена
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.2, context_len=8):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                            bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits
    
    def generate(self, context, tokenizer, max_gen_len=5, temperature=0.8):
        # генерируем продолжение текста
        self.eval()
        with torch.no_grad():
            if isinstance(context, str):
                context_tokens = tokenizer.encode(context, add_special_tokens=False)
            else:
                context_tokens = context
            
            generated = context_tokens.copy()
            device = next(self.parameters()).device  # получаем устройство модели
            
            for _ in range(max_gen_len):
                input_seq = generated[-self.context_len:]
                input_tensor = torch.tensor([input_seq]).to(device)
                
                logits = self.forward(input_tensor)
                
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = logits.argmax(dim=-1).item()
                
                generated.append(next_token)
                
                if next_token in [tokenizer.sep_token_id, tokenizer.eos_token_id]:
                    break
            
            return tokenizer.decode(generated[len(context_tokens):], skip_special_tokens=True)