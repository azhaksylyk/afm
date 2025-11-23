import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import preprocess  # используем твой препроцессор


def read_descriptions(path: Path) -> List[str]:
    """
    Читаем файл с наименованиями ТРУ.
    Пытаемся сначала как CSV, если не удалось — читаем построчно.
    Возвращаем список строк-описаний.
    """
    # 1) Попробуем через pandas (авто-детект разделителя)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        # Берём первую колонку как текст
        first_col = df.columns[0]
        texts = df[first_col].astype(str).tolist()
        # отбрасываем возможный заголовок типа 'DESCRIPTION'
        if len(texts) > 0 and texts[0].upper() == "DESCRIPTION":
            texts = texts[1:]
        return texts
    except Exception:
        pass

    # 2) Фоллбэк: читаем построчно, пропуская первую строку как заголовок
    texts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(line)
    return texts


def build_embeddings(texts: List[str], model_name: str = "distiluse-base-multilingual-cased-v2") -> np.ndarray:
    """
    Строим sentence-эмбеддинги для списка текстов.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    return np.array(embeddings)


def choose_k(n_samples: int) -> int:
    """
    Очень простой хак: выбираем количество кластеров по числу объектов.
    Это можно сделать параметром, но пусть будет автоподбор.
    """
    if n_samples <= 50:
        return min(5, n_samples)
    if n_samples <= 200:
        return 10
    if n_samples <= 1000:
        return 20
    if n_samples <= 5000:
        return 30
    return 50


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Кластеризация KMeans поверх эмбеддингов.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def top_terms_for_cluster(
    texts: List[str],
    cluster_labels: np.ndarray,
    cluster_id: int,
    top_n: int = 4,
) -> Tuple[str, List[str]]:
    """
    Для заданного кластера вытаскиваем топ-N терминов через TF-IDF
    и делаем из них название категории.
    """
    cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
    if not cluster_texts:
        return f"Класс_{cluster_id}", []

    # применяем тот же препроцессор
    processed = [preprocess(t) for t in cluster_texts]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        min_df=1,
        max_df=0.9
    )
    X = vectorizer.fit_transform(processed)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # усредняем TF-IDF по каждому признаку
    mean_tfidf = X.mean(axis=0).A1
    top_idx = mean_tfidf.argsort()[::-1][:top_n]
    top_terms = feature_names[top_idx].tolist()

    # название категории: несколько самых информативных слов через запятую
    category_name = ", ".join(top_terms) if top_terms else f"Класс_{cluster_id}"
    return category_name, top_terms


def auto_cluster_and_label(
    input_path: Path,
    output_path: Path,
    model_name: str = "distiluse-base-multilingual-cased-v2",
) -> None:
    """
    Полный пайплайн:
    1) читаем описания
    2) строим эмбеддинги
    3) кластеризуем
    4) придумываем названия категорий
    5) сохраняем результат
    """
    print(f"Читаю данные из: {input_path}")
    texts = read_descriptions(input_path)
    print(f"Всего строк с наименованиями: {len(texts)}")

    if not texts:
        raise ValueError("Не найдено ни одного наименования ТРУ в файле")

    print("Строю эмбеддинги (это может занять время)...")
    embeddings = build_embeddings(texts, model_name=model_name)

    n_clusters = choose_k(len(texts))
    print(f"Выбрано количество кластеров: {n_clusters}")

    print("Запускаю кластеризацию...")
    cluster_labels = cluster_embeddings(embeddings, n_clusters)

    # генерируем названия категорий
    cluster_to_name: dict[int, str] = {}
    cluster_to_terms: dict[int, List[str]] = {}

    print("Генерирую названия категорий для каждого кластера...")
    for cid in sorted(set(cluster_labels)):
        cat_name, terms = top_terms_for_cluster(texts, cluster_labels, cid)
        cluster_to_name[cid] = cat_name
        cluster_to_terms[cid] = terms
        print(f"Кластер {cid}: '{cat_name}' (топ-термины: {terms})")

    # сохраняем результат
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Сохраняю результат в: {output_path}")

    rows = []
    for i, text in enumerate(texts):
        cid = int(cluster_labels[i])
        cat_name = cluster_to_name[cid]
        terms = cluster_to_terms[cid]
        rows.append(
            {
                "description": text,
                "cluster_id": cid,
                "auto_category": cat_name,
                "cluster_top_terms": "; ".join(terms),
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Готово.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Автоматическая кластеризация и категоризация наименований ТРУ без train.csv"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Путь к исходному файлу с наименованиями (например, esf_fulll_*.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/esf_auto_labeled.csv",
        help="Путь для сохранения результата (CSV с авто-категориями)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distiluse-base-multilingual-cased-v2",
        help="Имя sentence-transformers модели",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    auto_cluster_and_label(
        input_path=input_path,
        output_path=output_path,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()