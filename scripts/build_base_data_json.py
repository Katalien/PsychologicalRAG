import json
import os
import glob

def build_complete_dataset(base_json_path, articles_dir, output_path):
    """
    Собирает полный датасет из базового JSON и текстов статей
    
    Args:
        base_json_path (str): Путь к базовому JSON с метаданными
        articles_dir (str): Директория с текстовыми файлами статей
        output_path (str): Путь для сохранения полного датасета
    """
    
    # Загружаем базовый JSON
    with open(base_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Счетчики для статистики
    articles_found = 0
    articles_missing = 0
    
    # Проходим по всем статьям в базовом датасете
    for item in dataset['rows']:
        article_id = item['id']
        category = item['category']
        
        # Формируем путь к файлу статьи
        article_filename = f"{category}_{article_id}.txt"
        article_path = os.path.join(articles_dir, article_filename)
        
        # Проверяем существование файла и читаем текст
        if os.path.exists(article_path):
            with open(article_path, 'r', encoding='utf-8') as f:
                item['text'] = f.read().strip()
            articles_found += 1
            print(f" Загружена статья: {article_filename}")
        else:
            print(f" Файл не найден: {article_filename}")
            articles_missing += 1
    
    # Сохраняем полный датасет
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Выводим статистику
    print(f"\nСБОРКА ЗАВЕРШЕНА:")
    print(f"   Найдено статей: {articles_found}")
    print(f"   Отсутствует статей: {articles_missing}")
    print(f"   Всего записей: {len(dataset['rows'])}")
    print(f"   Полный датасет сохранен: {output_path}")

def check_articles_coverage(base_json_path, articles_dir):
    """
    Проверяет, для каких статей есть текстовые файлы
    """
    with open(base_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print("🔍 ПРОВЕРКА ПОКРЫТИЯ СТАТЕЙ:")
    
    for item in dataset['rows']:
        article_id = item['id']
        category = item['category']
        article_filename = f"{category}_{article_id}.txt"
        article_path = os.path.join(articles_dir, article_filename)
        
        status = "ЕСТЬ" if os.path.exists(article_path) else "ОТСУТСТВУЕТ"
        print(f"   {status} {article_filename}")

if __name__ == "__main__":
    # Пути к файлам
    BASE_JSON_PATH = "./data/base_data.json"
    ARTICLES_DIR = "./data/articles"
    OUTPUT_PATH = "./final_dataset.json"
    
    # Проверяем покрытие статей
    print("=== ПРОВЕРКА СТАТЕЙ ===")
    check_articles_coverage(BASE_JSON_PATH, ARTICLES_DIR)
    
    print("\n=== СБОРКА ДАТАСЕТА ===")
    build_complete_dataset(BASE_JSON_PATH, ARTICLES_DIR, OUTPUT_PATH)