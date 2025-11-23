import re
from typing import Iterable

# Очень простой набор русских стоп-слов (можно расширить при желании)
RUSSIAN_STOPWORDS: set[str] = {
    "и", "в", "во", "на", "за", "от", "по", "до", "из", "для", "с", "со",
    "над", "под", "при", "без", "что", "это", "как", "к", "до", "о",
    "об", "про", "не", "а", "или", "ли", "же", "же", "данный", "услуги",
    "товары", "работы"
}

# Частые опечатки/нормализации (как пример для ТРУ)
REPLACEMENTS: dict[str, str] = {
    "молокосодержащий": "молочн",
    "ультрапастеризованный": "ультрапастер",
    "принтеры": "принтер",
    "принтеров": "принтер",
    "бумаги": "бумага",
    "услуга": "услуги",
}


def normalize_text(text: str) -> str:
    """
    Базовая нормализация:
    - нижний регистр
    - замена опечаток
    - удаление лишних символов
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # заменим типичные опечатки и регулярные выражения
    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)

    # убираем всё, кроме букв, цифр и базовой пунктуации
    text = re.sub(r"[^a-zа-я0-9\s%./-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: Iterable[str] | None = None) -> str:
    if stopwords is None:
        stopwords = RUSSIAN_STOPWORDS

    tokens = text.split()
    filtered = [t for t in tokens if t not in stopwords]
    return " ".join(filtered)


def preprocess(text: str) -> str:
    """
    Полный пайплайн предобработки:
    1) нормализация
    2) удаление стоп-слов
    """
    text = normalize_text(text)
    text = remove_stopwords(text)
    return text