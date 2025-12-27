import json
import logging
import os
import re
import time

import html2text
import requests
from bs4 import BeautifulSoup

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArticleDownloader:
    def __init__(self, base_json_path, articles_dir):
        self.base_json_path = base_json_path
        self.articles_dir = articles_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Инициализация html2text для чистого текста
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.body_width = 0

        # Создаем директорию для статей, если её нет
        os.makedirs(self.articles_dir, exist_ok=True)

    def clean_text(self, text):
        """Очистка текста от лишних пробелов и символов"""
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Удаляем лишние пустые строки
        text = re.sub(r'[ \t]+', ' ', text)  # Удаляем лишние пробелы
        return text.strip()

    def extract_main_content(self, soup, url):
        """
        Извлекает основной контент статьи, убирая боковые панели, меню и т.д.
        Использует различные стратегии для разных сайтов.
        """
        # Удаляем ненужные элементы
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Стратегии поиска основного контента (в порядке приоритета)
        content_selectors = [
            'article',
            '[class*="article"]',
            '[class*="content"]',
            '[class*="post"]',
            '[class*="blog"]',
            'main',
            '.entry-content',
            '.post-content',
            '.article-content',
            '#content',
            '.content',
            'div[role="main"]'
        ]

        main_content = None

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content and len(main_content.get_text(strip=True)) > 200:
                break

        # Если не нашли по селекторам, используем эвристики
        if not main_content:
            # Ищем div с наибольшим количеством текста
            all_divs = soup.find_all('div')
            if all_divs:
                main_content = max(all_divs, key=lambda x: len(x.get_text(strip=True)))

        return main_content

    def download_article(self, url, article_id, category):
        """Скачивает и очищает статью по URL"""
        try:
            logger.info(f"Скачиваем статью {article_id} из {url}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Извлекаем основной контент
            main_content = self.extract_main_content(soup, url)

            if not main_content:
                logger.warning(f"Не удалось извлечь контент для статьи {article_id}")
                return None

            # Конвертируем в чистый текст
            html_content = str(main_content)
            text_content = self.html_converter.handle(html_content)
            clean_content = self.clean_text(text_content)

            # Проверяем, что контент достаточно большой
            if len(clean_content) < 100:
                logger.warning(f"Слишком короткий контент для статьи {article_id}")
                return None

            return clean_content

        except Exception as e:
            logger.error(f"Ошибка при скачивании статьи {article_id}: {e}")
            return None

    def save_article(self, content, article_id, category):
        """Сохраняет статью в файл"""
        filename = f"{category}_{article_id}.txt"
        filepath = os.path.join(self.articles_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Статья сохранена: {filename}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении статьи {filename}: {e}")
            return False

    def download_all_articles(self, delay=2):
        """Скачивает все статьи из base_dataset.json"""
        # Загружаем базовый датасет
        with open(self.base_json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        success_count = 0
        fail_count = 0
        skipped_count = 0

        for item in dataset['rows']:
            article_id = item['id']
            category = item['category']
            url = item['link']

            # Проверяем, не скачана ли уже статья
            filename = f"{category}_{article_id}.txt"
            filepath = os.path.join(self.articles_dir, filename)

            if os.path.exists(filepath):
                logger.info(f"Статья уже существует, пропускаем: {filename}")
                skipped_count += 1
                continue

            # Скачиваем статью
            content = self.download_article(url, article_id, category)

            if content:
                # Сохраняем статью
                if self.save_article(content, article_id, category):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

            # Задержка между запросами
            time.sleep(delay)

        logger.info(f"\nИТОГИ СКАЧИВАНИЯ:")
        logger.info(f"   Успешно: {success_count}")
        logger.info(f"   Пропущено: {skipped_count}")
        logger.info(f"   Ошибки: {fail_count}")
        logger.info(f"   Всего: {len(dataset['rows'])}")

    def check_download_status(self):
        """Проверяет статус скачивания статей"""
        with open(self.base_json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        downloaded = 0
        missing = 0

        print("\nСТАТУС СКАЧИВАНИЯ СТАТЕЙ:")
        for item in dataset['rows']:
            article_id = item['id']
            category = item['category']
            filename = f"{category}_{article_id}.txt"
            filepath = os.path.join(self.articles_dir, filename)

            if os.path.exists(filepath):
                # Проверяем размер файла
                file_size = os.path.getsize(filepath)
                status = "ОК" if file_size > 100 else "Ошибка (маленький)"
                print(f"   {status} {filename} ({file_size} байт)")
                downloaded += 1
            else:
                print(f"Не удалось загрузить: {filename}")
                missing += 1

        print(f"\nСкачано: {downloaded}/{len(dataset['rows'])}")
        print(f"Отсутствует: {missing}")


def main():
    BASE_JSON_PATH = "../data/base_dataset.json"
    ARTICLES_DIR = "../data/articles"

    # Создаем загрузчик
    downloader = ArticleDownloader(BASE_JSON_PATH, ARTICLES_DIR)

    # Проверяем текущий статус
    downloader.check_download_status()

    # Спрашиваем пользователя
    choice = input("\nХотите скачать отсутствующие статьи? (y/n): ")
    if choice.lower() == 'y':
        # Скачиваем все статьи
        downloader.download_all_articles(delay=1)  # Задержка 1 секунда между запросами

        # Снова проверяем статус
        downloader.check_download_status()


if __name__ == "__main__":
    main()