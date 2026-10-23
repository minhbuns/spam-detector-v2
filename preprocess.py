import re
import nltk
from nltk.corpus import stopwords
import emoji

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = nltk.stem.PorterStemmer()


def remove_subject(text):
    return re.sub(r"^\s*subject\s*:\s*", "", text, count=1, flags=re.I)


def clean_html(text):
    # Chuyển nội dung comment và thẻ HTML thành token HTML.
    text = re.sub(r"<!--.*?-->", " HTML ", text, flags=re.S)
    return re.sub(r"<[^>]*>", " HTML ", text)


def clean_email_address(text):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.sub(pattern, " EMAIL ", text)


def clean_url(text):
    # Nhận diện URL có http(s), www hoặc domain phổ biến.
    pattern = (
        r"(?:https?://|www\.)[^\s<>\"']+"
        r"|(?<![@\w])(?:[a-zA-Z0-9-]+\.)+"
        r"(?:com|net|org|edu|gov|io|co|info|biz|vn)"
        r"(?:/[^\s<>\"']*)?"
    )

    def replace_url(match):
        raw_url = match.group(0)
        url = raw_url.rstrip(".,!?;:)")
        suffix = raw_url[len(url):]
        return f" URL{len(url)} " + suffix

    return re.sub(pattern, replace_url, text, flags=re.I)


def clean_emoji(text):
    # Hỗ trợ emoji gồm nhiều code point nếu phiên bản emoji cho phép.
    try:
        return emoji.replace_emoji(text, replace=" EMOJI ")
    except AttributeError:
        # Tương thích phiên bản emoji cũ.
        return "".join(
            " EMOJI " if emoji.is_emoji(char) else char
            for char in text
        )


def clean_number(text):
    return re.sub(
        r"\d+",
        lambda match: f" NUMBER{len(match.group(0))} ",
        text,
    )


def clean_special_characters(text):
    # Giữ chữ, chữ số và dấu nháy để không phá token URL25/NUMBER3.
    return re.sub(r"[^a-zA-Z0-9\s']", " ", text)


def normalize_word(text):
    return " ".join(STEMMER.stem(word) for word in text.split())


def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def clean_text(text):
    text = str(text).lower()
    text = remove_subject(text)

    # Thay các mẫu đặc biệt trước khi lọc ký tự.
    text = clean_html(text)
    text = clean_email_address(text)
    text = clean_url(text)
    text = clean_emoji(text)
    text = clean_number(text)

    text = clean_special_characters(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_stopwords(text)

    # Chỉ bật stemming nếu đã quyết định dùng nhất quán cho train/test.
    # text = normalize_word(text)

    return text
