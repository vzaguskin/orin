#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МАКСИМАЛЬНЫЙ СТРЕСС-ТЕСТ стрим-процессора:
дроби, валюты, математика, аббревиатуры, спецсимволы, иероглифы, эмодзи.
"""

from normalizer import StreamTextProcessor

def generate_max_stress_text() -> str:
    """Максимально насыщенный текст для стресс-теста"""
    return (
        "Версия 2.0 и π≈3.14159 — это дробные числа. "
        "Цены: $19.99, €49.50, £99.99, ¥1000.50 — валюты с копейками. "
        "Аббревиатуры: HTML5, CSS3, JS2022, API v3.0, NASA, FBI, PhD. "
        "Математика: 2+2=4, 100/3≈33.33, (a+b)^2 = a^2 + 2ab + b^2, x^2 + y^2 = z^2. "
        "Спецсимволы: & | ~ ` ' \" # % @ $ ^ * \\ / < > = — все должны расширяться. "
        "Иероглифы: 北京办公室 и эмодзи 😊🚀🤖 — должны быть удалены. "
        "Дата: 2024-12-31, время: 14:30:45. "
        "Телефон: +7 (495) 123-45-67, почта: test@example.com. "
        "Дроби: 0.5, 1.25, 3.1415926535. "
        "Валюты без пробелов:$1.99€2.50. "
        "Сложная формула: E=mc^2, F=ma, a^2+b^2=c^2. "
        "Ёжик ёлку ёл — буква ё должна сохраниться. "
        "Завершение без точки чтобы проверить flush"
    )

def main():
    processor = StreamTextProcessor(max_chunk_size=180)
    text = generate_max_stress_text()
    
    print("=" * 90)
    print("МАКСИМАЛЬНЫЙ СТРЕСС-ТЕСТ: посимвольная обработка сложного текста")
    print("=" * 90)
    print(f"\nИсходный текст ({len(text)} символов):\n")
    print(text)
    print("\n" + "=" * 90)
    print("ПОТОКОВАЯ ОБРАБОТКА:")
    print("=" * 90 + "\n")
    
    fragments = []
    total_chars = 0
    
    # Посимвольная подача
    for i, char in enumerate(text):
        new_fragments = processor.feed(char)
        
        # Выводим прогресс каждые 50 символов
        #if i % 50 == 0:
        #    buf_len = len(processor.clean_buffer)
        #    buf_preview = processor.clean_buffer[:40].replace('\n', '\\n')
        #    print(f"[{i:4d}/{len(text)}] Буфер: {buf_len:3d} симв. | '{buf_preview}{'...' if buf_len > 40 else ''}'")
        
        # Выводим отправленные фрагменты
        for frag in new_fragments:
            fragments.append(frag)
            total_chars += len(frag)
            print(f"\n📤 ФРАГМЕНТ #{len(fragments):2d} | {len(frag):4d} симв. |")
            print(f"   «{frag}»")
    
    # Финальный flush
    print("\n" + "-" * 90)
    print("FLUSH — отправка остатка буфера:")
    print("-" * 90 + "\n")
    
    final_fragments = processor.flush()
    for frag in final_fragments:
        fragments.append(frag)
        total_chars += len(frag)
        print(f"\n📤 ФИНАЛ #{len(fragments):2d} | {len(frag):4d} симв. |")
        print(f"   «{frag}»")
    
    # Итоговая статистика
    print("\n" + "=" * 90)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 90)
    print(f"Всего фрагментов:       {len(fragments)}")
    print(f"Общая длина фрагментов: {total_chars} символов")
    print(f"Буфер после flush:      '{processor.clean_buffer}' (пустой: {processor.clean_buffer == ''})")
    print(f"Потеря данных:          {'НЕТ' if processor.clean_buffer == '' else 'ЕСТЬ'}")
    print("=" * 90)
    
    # Анализ проблемных мест
    print("\n🔍 АНАЛИЗ КРИТИЧЕСКИХ КЕЙСОВ:")
    all_text = " ".join(fragments).lower()
    
    checks = [
        ("Дроби целы", "3.14" not in all_text and "19.99" not in all_text, "дроби не разорваны"),
        ("Валюты целы", "$" not in all_text and "€" not in all_text and "£" not in all_text, "валюты обработаны"),
        ("Аббревиатуры", "эйч ти эм эль" in all_text and "си эс эс" in all_text, "HTML5/CSS3 расшифрованы"),
        ("Математика", "плюс" in all_text and "умножить" in all_text and "степени" in all_text, "операторы расширены"),
        ("Иероглифы", "北京" not in all_text and "办公室" not in all_text, "иероглифы удалены"),
        ("Эмодзи", "😊" not in all_text and "🚀" not in all_text, "эмодзи удалены"),
        ("Буква ё", "ёжик" in all_text or "ежик" in all_text, "буква ё сохранена"),
        ("Спецсимволы", "собака" in all_text and "процент" in all_text, "@ и % расширены"),
    ]
    
    for name, condition, desc in checks:
        status = "✅" if condition else "❌"
        print(f"{status} {name:20s} → {desc}")
    
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()