import re
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
def is_emoji(text):
    for char in text:
        if emoji.is_emoji(char):
            return True
    return False

def clean_emoji(text):
    for char in text:
        if emoji.is_emoji(char):
            text = text.replace(char, 'EMOJI')
    return text

def clean_url(text):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        text = text.replace(url, 'URL')
    return text

def clean_html(text):
    html_tags = re.findall(r'<[^>]+>', text)
    for tag in html_tags:
        text = text.replace(tag, 'HTML')
    return text

def clean_number(text):
    numbers = re.findall(r'\d+', text)
    for number in numbers:
        text = text.replace(number, 'NUMBER')
    return text

def clean_email_address(text):
    email_addresses = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',text)
    for email in email_addresses:
        text = text.replace(email, 'EMAIL')
    return text

def clean_special_characters(text):
    # Giữ chữ cái, khoảng trắng và dấu nháy trong từ
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    return text

def normalize_word(text):
    tmp = text.split()
    return ' '.join(stemmer.stem(word) for word in tmp)

def remove_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tmp = text.split()
    return ' '.join(word for word in tmp if word not in stop_words)

def clean_text(text):
    text = clean_emoji(text)
    # text = clean_url(text) 
    text = clean_html(text)
    text = clean_number(text)
    text = clean_email_address(text)
    
    text = text.lower()
    text = clean_special_characters(text)
    text = normalize_word(text)
    text = remove_stopwords(text)
    return text
