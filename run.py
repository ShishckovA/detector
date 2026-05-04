#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для извлечения всех картинок из HTML-страницы по ссылке.

Извлекает URL изображений из:
- тегов <img> (атрибуты src, srcset, data-src)
- встроенных CSS-правил background-image (атрибут style)
- элементов <picture> и <source> (srcset)

Преобразует относительные пути в абсолютные. При необходимости скачивает все изображения.
"""

import os
import re
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def get_absolute_url(base_url, relative_url):
    """Преобразует относительный URL в абсолютный."""
    if not relative_url:
        return None
    # Пропускаем data: URI и javascript:
    if relative_url.startswith(('data:', 'javascript:', '#')):
        return None
    return urljoin(base_url, relative_url)


def extract_urls_from_srcset(srcset_str, base_url):
    """Извлекает все URL из атрибута srcset."""
    urls = []
    # Формат srcset: url ширина, url 2x и т.д.
    # Разделитель - запятая, но внутри url могут быть запятые? Обычно нет.
    parts = srcset_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Первое слово (до пробела) — это URL
        url_part = part.split()[0]
        if url_part:
            abs_url = get_absolute_url(base_url, url_part)
            if abs_url:
                urls.append(abs_url)
    return urls


def extract_background_images_from_style(element, base_url):
    """Извлекает URL из inline стиля background(-image)."""
    style = element.get('style', '')
    if not style:
        return []
    # Ищем background-image: url(...) или background: url(...)
    pattern = r'background(?:-image)?\s*:\s*url\([\'"]?([^\)\'"]+)[\'"]?\)'
    matches = re.findall(pattern, style, re.IGNORECASE)
    urls = []
    for match in matches:
        abs_url = get_absolute_url(base_url, match)
        if abs_url:
            urls.append(abs_url)
    return urls


def extract_images_from_html(html_content, base_url):
    """
    Извлекает все URL изображений из HTML.
    Возвращает множество уникальных абсолютных URL.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    image_urls = set()

    # 1. Теги <img>
    for img in soup.find_all('img'):
        # Атрибут src
        src = img.get('src')
        if src:
            abs_url = get_absolute_url(base_url, src)
            if abs_url:
                image_urls.add(abs_url)

        # Атрибут srcset (множество URL)
        srcset = img.get('srcset')
        if srcset:
            urls = extract_urls_from_srcset(srcset, base_url)
            image_urls.update(urls)

        # Атрибут data-src (lazy loading)
        data_src = img.get('data-src')
        if data_src:
            abs_url = get_absolute_url(base_url, data_src)
            if abs_url:
                image_urls.add(abs_url)

        # Другие варианты: data-original и т.д.
        for attr in ('data-original', 'data-lazy-src'):
            lazy_src = img.get(attr)
            if lazy_src:
                abs_url = get_absolute_url(base_url, lazy_src)
                if abs_url:
                    image_urls.add(abs_url)

    # 2. Элементы <picture> и <source>
    for source in soup.find_all('source'):
        srcset = source.get('srcset')
        if srcset:
            urls = extract_urls_from_srcset(srcset, base_url)
            image_urls.update(urls)
        # Для старых браузеров
        src = source.get('src')
        if src:
            abs_url = get_absolute_url(base_url, src)
            if abs_url:
                image_urls.add(abs_url)

    # 3. Встроенные стили background-image
    for element in soup.find_all(style=True):
        urls = extract_background_images_from_style(element, base_url)
        image_urls.update(urls)

    # 4. Дополнительно: теги с атрибутом data-background
    for elem in soup.find_all(attrs={'data-background': True}):
        bg = elem['data-background']
        abs_url = get_absolute_url(base_url, bg)
        if abs_url:
            image_urls.add(abs_url)

    return image_urls


def download_image(url, folder, timeout=10):
    """
    Скачивает изображение по URL и сохраняет в указанную папку.
    Возвращает имя сохранённого файла или None в случае ошибки.
    """
    try:
        if not url.endswith(".jpg"):
            return None
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        # Определяем расширение файла
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            # Если это не изображение, лучше пропустить
            return None

        # Извлекаем имя файла из URL или генерируем
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            # Создаём имя, если нет расширения
            ext_map = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/webp': '.webp',
                'image/svg+xml': '.svg'
            }
            ext = ext_map.get(content_type, '.img')
            filename = f"image_{hash(url) & 0xffffffff}{ext}"

        # Убираем опасные символы из имени
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
        filepath = os.path.join(folder, filename)

        # Сохраняем
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Извлечение всех изображений из HTML по ссылке')
    parser.add_argument('url', help='URL страницы для парсинга')
    parser.add_argument('-d', '--download', metavar='DIR', help='Скачать изображения в указанную папку')
    parser.add_argument('--timeout', type=int, default=10, help='Таймаут для запросов (секунды)')
    args = parser.parse_args()

    # Загружаем HTML
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(args.url, headers=headers, timeout=args.timeout)
        response.raise_for_status()
        html = response.text
        base_url = response.url  # окончательный URL после возможных редиректов
    except Exception as e:
        print(f"Не удалось загрузить страницу: {e}")
        return

    # Извлекаем URL изображений
    image_urls = extract_images_from_html(html, base_url)

    if not image_urls:
        print("Изображений не найдено.")
        return

    print(f"Найдено {len(image_urls)} изображений:")
    for url in sorted(image_urls):
        print(url)

    # Скачивание, если указано
    if args.download:
        os.makedirs(args.download, exist_ok=True)
        print(f"\nСкачивание изображений в папку '{args.download}'...")
        success_count = 0
        for url in image_urls:
            filename = download_image(url, args.download, timeout=args.timeout)
            if filename:
                print(f"✓ Сохранено: {filename}")
                success_count += 1
            else:
                print(f"✗ Не удалось: {url}")

        print(f"\nЗавершено. Успешно скачано {success_count} из {len(image_urls)} изображений.")


if __name__ == '__main__':
    main()
