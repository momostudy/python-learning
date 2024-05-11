import pandas as pd
import jieba
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('C:\Download')
df = pd.read_stata('企业CSR报告全文.dta')

def load_sentiment_words(file_path):
    encodings = ['utf-8', 'gb2312', 'gbk', 'gb18030']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                words = [line.strip() for line in file]
            return set(words)
        except UnicodeDecodeError:
            continue
    raise Exception("Error!")

positive_words = load_sentiment_words('tsinghua_positive_gb.txt')
negative_words = load_sentiment_words('tsinghua_negative_gb.txt')
positive_counts = []
negative_counts = []
tones = []

for text in tqdm(df['CSR_text']):
    words = jieba.cut(text)
    positive_count = 0
    negative_count = 0
    for word in words:
        if word in positive_words:
            positive_count += 1
        elif word in negative_words:
            negative_count += 1
    positive_counts.append(positive_count)
    negative_counts.append(negative_count)
    if positive_count + negative_count > 0:
        tone = (positive_count - negative_count) / (positive_count + negative_count)
    else:
        tone = 0
    tones.append(tone)

df['positive_count'] = positive_counts
df['negative_count'] = negative_counts
df['tone'] = tones
df.to_stata('CSR_report_with_sentiment.dta', write_index=False, version=118)