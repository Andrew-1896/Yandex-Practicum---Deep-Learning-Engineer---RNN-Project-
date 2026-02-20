import pandas as pd
import requests
import io
import re


def load_and_clean_data():

    # загружает датасет твитов с постоянного URL и очищает его.
    
    # URL для загрузки
    url = "https://code.s3.yandex.net/deep-learning/tweets.txt"
    print(f"Загрузка данных с URL: {url}")
    
    try:
        # cкачиваем файл
        response = requests.get(url)
        response.raise_for_status()  # gроверяем, что загрузка успешна
        
        # читаем CSV из скачанного содержимого
        # указываем engine='python' на случай, если в файле есть проблемы с парсингом
        raw_dataset = pd.read_csv(
            io.StringIO(response.text), 
            sep='\r\n', 
            header=None,
            engine='python',
            names=['tweet']  # сразу задаем имя столбца
        )
        
        print(f"Успешно загружено {len(raw_dataset)} строк")
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка сети при загрузке файла: {e}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении CSV: {e}")

    def clean_tweet(text):
        if pd.isna(text):
            return text
        
        text = str(text).lower()
        
        # удаляем упоминания, хэштеги и ссылки
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # удаляем повторяющиеся скобки (смайлики)
        text = re.sub(r'[()]{2,}', '', text)
         
        # удаляем конкретные смайлики
        smileys = [
        r'= \)',      # =)
        r'= \(',      # =(
        r': \)',      # :)
        r': \(',      # :(
        r'; \)',      # ;)
        r'; \(',      # ;(
        r": '\(",      # :'(
        r": '\)",      # :')
        r': D',       # :D
        r'X D',       # XD
        r': -\)',     # :-)
        r': -\(',     # :-(
        r': -D'       # :-D
    ]
    
        # создаем общий паттерн из всех смайликов и удаляем их
        smileys_pattern = '|'.join(smileys)
        text = re.sub(smileys_pattern, '', text) 

        # удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    dataset_processed = pd.DataFrame({
        'tweet': raw_dataset['tweet'].apply(clean_tweet)
    })
    
    # удаляем пустые строки и слишком короткие (5 и менее символов)
    dataset_processed = dataset_processed[dataset_processed['tweet'].str.len() > 5]
    dataset_processed = dataset_processed.reset_index(drop=True)
    
    print(f"Загружено твитов: {len(raw_dataset)}")
    print(f"После очистки: {len(dataset_processed)}")
    
    return dataset_processed