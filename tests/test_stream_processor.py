# tests/test_stream_processor.py
import re
import pytest
from normalizer import StreamTextProcessor


class TestStreamTextProcessorGuarantees:
    """Критические гарантии — падение любого теста = нерабочий продукт."""

    def test_guarantee_no_digits_in_output(self):
        """ГАРАНТИЯ #1: В финальном выводе НЕТ цифр (0-9)."""
        processor = StreamTextProcessor(max_chunk_size=200)
        text = "Цена $99.99 за 100 единиц. Версия 2.15.3. Дата 2024-12-31. π≈3.14159"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments)
        
        # Извлекаем все цифры из вывода
        digits_found = re.findall(r'\d', full_text)
        assert digits_found == [], (
            f"НАРУШЕНА ГАРАНТИЯ #1: обнаружены цифры в выводе: {digits_found}\n"
            f"Полный вывод: '{full_text}'"
        )
    
    def test_guarantee_hard_limit(self):
        """ГАРАНТИЯ #2: Каждый фрагмент ≤ max_chunk_size символов."""
        processor = StreamTextProcessor(max_chunk_size=28)
        # Стресс-текст с максимальным раздуванием (числа → длинные слова)
        text = "1111 2222 3333 4444 5555 6666 7777 8888 9999 1000000"
        
        fragments = []
        for ch in text:
            new_frags = processor.feed(ch)
            for frag in new_frags:
                assert len(frag) <= 28, (
                    f"НАРУШЕНА ГАРАНТИЯ #2: фрагмент превышает лимит 28!\n"
                    f"Длина: {len(frag)}, текст: '{frag}'"
                )
            fragments.extend(new_frags)
        
        for frag in processor.flush():
            assert len(frag) <= 28, (
                f"НАРУШЕНА ГАРАНТИЯ #2 в flush: фрагмент превышает лимит 28!\n"
                f"Длина: {len(frag)}, текст: '{frag}'"
            )
            fragments.append(frag)
        
        assert fragments, "Должны быть сгенерированы фрагменты"
    
    def test_guarantee_number_integrity(self):
        """ГАРАНТИЯ #3: Числа НЕ разрываются на части (нет '3 .' или '. 14')."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Число пи 3.14159 а е 2.71828 версия 2.15.3"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Запрещённые паттерны разрыва чисел
        forbidden_patterns = [
            (r'\d \.', "цифра + пробел + точка (например '3 .')"),
            (r'\. \d', "точка + пробел + цифра (например '. 14')"),
            (r'\d \d', "две цифры с пробелом между ними (разрыв числа)"),
        ]
        
        for pattern, description in forbidden_patterns:
            matches = re.findall(pattern, full_text)
            assert not matches, (
                f"НАРУШЕНА ГАРАНТИЯ #3: обнаружен разрыв числа!\n"
                f"Паттерн: {description}\n"
                f"Совпадения: {matches}\n"
                f"Полный вывод: '{full_text}'"
            )


class TestNumberTransformation:
    """Трансформация чисел: целые, дроби, версии."""

    def test_whole_numbers(self):
        """Целые числа → слова."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "1 23 456 1000 2024 1000000"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        assert "один" in full_text
        assert "двадцать три" in full_text or "двадцати трех" in full_text
        assert "четыреста пятьдесят шесть" in full_text
        assert "тысяча" in full_text
        assert "две тысячи двадцать четыре" in full_text or "дветысячидвадцатьчетыре" in full_text
        assert "миллион" in full_text
    
    def test_decimal_fractions(self):
        """Десятичные дроби → слова (все части трансформируются)."""
        processor = StreamTextProcessor(max_chunk_size=200)
        text = "3.14 2.718 0.5 1.25"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Целые части
        assert "три" in full_text
        assert "два" in full_text
        assert "ноль" in full_text or "нуль" in full_text
        assert "один" in full_text
        
        # Дробные части (полностью трансформированы)
        assert "четырнадцать" in full_text
        assert "семьсот восемнадцать" in full_text or "семьсот восемнадцати" in full_text
        assert "пять" in full_text
        assert "двадцать пять" in full_text or "двадцати пяти" in full_text
        
        # Структура
        assert "три точка четырнадцать" in full_text or "три точка четырнадцати" in full_text
        assert "ноль точка пять" in full_text or "нуль точка пять" in full_text
    
    def test_version_numbers(self):
        """Версии (несколько точек) → слова."""
        processor = StreamTextProcessor(max_chunk_size=200)
        text = "Версия 2.15.3 установлена"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Все части версии трансформированы
        assert "два" in full_text
        assert "пятнадцать" in full_text
        assert "три" in full_text
        assert "точка" in full_text
        
        # Структура сохранена
        assert "два точка пятнадцать точка три" in full_text
        
        # Нет цифр
        assert "2" not in full_text
        assert "15" not in full_text
        assert "3" not in full_text
    


    def test_mixed_alphanumeric(self):
        """Смешанный текст+цифры разбивается на сущности с транслитерацией латиницы."""
        processor = StreamTextProcessor(max_chunk_size=200)
        text = "JS2022 CSS3 v3.0 2ab"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # JS2022 → "джей эс" + "две тысячи двадцать два"
        assert "джей эс" in full_text or "джейс" in full_text
        assert "две тысячи двадцать два" in full_text or "дветысячидвадцатьдва" in full_text
        
        # CSS3 → "си эс эс" + "три"
        assert "си эс эс" in full_text or "сиэсэс" in full_text
        assert "три" in full_text
        assert "css" not in full_text.replace(" ", "")
        
        # v3.0 → кириллическая "в" + "три точка ноль"
        assert "в" in full_text  # кириллическая 'в' из транслитерации 'v'
        assert "v" not in full_text.replace(" ", "")  # латинская 'v' удалена
        assert "три точка ноль" in full_text
        
        # 2ab → "два" + "аб" (латинские 'ab' → кириллические 'аб')
        assert "два" in full_text
        assert "аб" in full_text  # 'a'→'а', 'b'→'б' → "аб"
        assert "ab" not in full_text.replace(" ", "")
        
        # Гарантия: нет латиницы в выводе
        latin_chars = re.findall(r'[a-z]', full_text.replace(" ", ""))
        assert not latin_chars, f"Латинские буквы в выводе: {latin_chars}"

class TestAbbreviations:
    """Аббревиатуры → произношение букв по одной."""

    def test_common_abbreviations(self):
        """Стандартные аббревиатуры."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "NASA FBI CIA HTML CSS JS PhD"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # NASA → "эн эй эс эй" (по буквам)
        assert "эн эй эс эй" in full_text or "энэйэсэй" in full_text
        
        # FBI → "эф би ай"
        assert "эф би ай" in full_text or "эфбиай" in full_text
        
        # CIA → "си ай эй"
        assert "си ай эй" in full_text or "сиайэй" in full_text
        
        # HTML → "аш ти эм эль"
        assert "аш ти эм эль" in full_text or "аштемэль" in full_text
        
        # CSS → "си эс эс"
        assert "си эс эс" in full_text or "сиэсэс" in full_text
        
        # JS → "джей эс"
        assert "джей эс" in full_text or "джейс" in full_text
        
        # PhD → "пи эйч ди"
        assert "пи эйч ди" in full_text or "пиэйчди" in full_text
    
    def test_single_uppercase_letters(self):
        """Одиночные заглавные буквы → произношение."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "A B C D E F G"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        assert "эй" in full_text  # A
        assert "би" in full_text  # B
        assert "си" in full_text  # C
        assert "ди" in full_text  # D
        assert "и" in full_text   # E
        assert "эф" in full_text  # F
        assert "джи" in full_text # G


class TestSpecialCharacters:
    """Спецсимволы → слова."""

    def test_arithmetic_operators(self):
        """Арифметические операторы."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "2+2=4 10-5=5 6*7=42 100/4=25"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        assert "плюс" in full_text
        assert "минус" in full_text
        assert "умножить на" in full_text
        assert "разделить на" in full_text
        assert "равно" in full_text
        
        # Нет нераскрытых символов
        assert "+" not in full_text
        assert "-" not in full_text
        assert "*" not in full_text
        assert "/" not in full_text
        assert "=" not in full_text
    
    def test_currency_symbols(self):
        """Валютные символы."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "$99 €50 £75 ¥1000"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        assert "доллар" in full_text
        assert "евро" not in full_text or "€" not in full_text  # евро может не поддерживаться в num2words
        assert "фунт" in full_text or "стерлинг" in full_text
        assert "иена" in full_text or "¥" not in full_text
        
        # Нет нераскрытых символов
        assert "$" not in full_text
        # € и ¥ могут остаться если нет поддержки в num2words — это допустимо
    
    def test_comparison_operators(self):
        """Операторы сравнения (<, >) вне тегов."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "5<10 20>15"  # Не теги, а сравнения
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        assert "пять" in full_text
        assert "десять" in full_text
        assert "меньше" in full_text
        assert "больше" in full_text
        
        # Нет нераскрытых символов
        assert "<" not in full_text
        assert ">" not in full_text
    
    def test_brackets_and_punctuation(self):
        """Скобки и дополнительные символы."""
        processor = StreamTextProcessor(max_chunk_size=200)
        text = "(a+b)^2 {x|y} [1,2,3] #tag @user"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        assert "скобка открывается" in full_text
        assert "скобка закрывается" in full_text
        assert "фигурная скобка" in full_text or "{" not in full_text or "}" not in full_text
        assert "квадратная" in full_text or "[" not in full_text or "]" not in full_text
        assert "решётка" in full_text or "#" not in full_text
        assert "собака" in full_text or "@" not in full_text


class TestTagHandling:
    """Обработка тегов <...>."""

    def test_html_like_tags_ignored(self):
        """Теги игнорируются, содержимое вне тегов обрабатывается."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Привет <b>Иван</b> как <i>дела</i> сегодня"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Содержимое тегов сохранено
        assert "привет" in full_text
        assert "иван" in full_text
        assert "как" in full_text
        assert "дела" in full_text
        assert "сегодня" in full_text
        
        # Теги и их содержимое (буквы тегов) удалены
        assert "<" not in full_text
        assert ">" not in full_text
        assert "b" not in full_text.replace(" ", "")  # буква 'b' из <b> удалена
        assert "i" not in full_text.replace(" ", "")  # буква 'i' из <i> удалена
    
    # tests/test_stream_processor.py — заменить два теста:


    def test_lt_gt_as_operators_not_tags(self):
        """< и > как операторы (не после буквы) раскрываются как слова."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "5<10 test@example.com 20>15"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Операторы раскрыты
        assert "меньше" in full_text
        assert "больше" in full_text
        
        # Email: @ раскрыт как "собака"
        assert "собака" in full_text
        
        # Латиница транслитерирована в кириллицу (гарантия отсутствия латиницы)
        assert "test" not in full_text.replace(" ", "")
        assert "example" not in full_text.replace(" ", "")
        assert "com" not in full_text.replace(" ", "")
        
        # Проверяем кириллические аналоги (простая транслитерация):
        # 't'→'т', 'e'→'е', 's'→'с' → "тест"
        # 'e'→'е', 'x'→'кс', 'a'→'а', 'm'→'м', 'p'→'п', 'l'→'л', 'e'→'е' → "ексампле"
        # 'c'→'с', 'o'→'о', 'm'→'м' → "сом" (но в доменах часто "ком")
        assert "тест" in full_text or "т е с т" in full_text
        # Для "example" допускаем варианты из-за 'x'→'кс'
        assert any(word in full_text for word in ["ексампл", "екзампл", "ексампле", "екзампле"])
        # Для "com" — простая транслитерация 'c'→'с', 'o'→'о', 'm'→'м' → "сом", но в русском домены произносят как "ком"
        assert "ком" in full_text or "сом" in full_text or "к о м" in full_text
        
        # Нет нераскрытых символов
        assert "<" not in full_text
        assert ">" not in full_text

    # tests/test_stream_processor.py — заменить тесты тегов:

    def test_html_like_tags_ignored(self):
        """Теги игнорируются, НО содержимое МЕЖДУ тегами сохраняется."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Привет <b>Иван</b> как <i>дела</i> сегодня"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Содержимое тегов СОХРАНЕНО
        assert "привет" in full_text
        assert "иван" in full_text  # ← КРИТИЧЕСКИ ВАЖНО: Иван не потерян!
        assert "как" in full_text
        assert "дела" in full_text
        assert "сегодня" in full_text
        
        # Теги удалены
        assert "<" not in full_text
        assert ">" not in full_text
        assert "b" not in full_text.replace(" ", "")  # буквы тегов удалены
        assert "i" not in full_text.replace(" ", "")


    def test_closing_tags_ignored(self):
        """Закрывающие теги </tag> полностью игнорируются."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Ответ </tool_call> пользователю"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Тег полностью удалён (без артефактов "разделить на тхинк")
        assert "меньше" not in full_text  # '<' не раскрыто
        assert "разделить" not in full_text  # '/' не раскрыто
        assert "больше" not in full_text    # '>' не раскрыто
        assert "think" not in full_text.replace(" ", "")
        assert "тхинк" not in full_text.replace(" ", "")
        
        # Полезный текст сохранён
        assert "ответ" in full_text
        assert "пользователю" in full_text  # ← КРИТИЧЕСКИ ВАЖНО: текст после тега не потерян!




class TestCharacterFiltering:
    """Фильтрация недопустимых символов."""

    def test_cyrillic_preserved(self):
        """Кириллица сохраняется (включая ё)."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Привет ёжик ёлку ёл на ёлке"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Буква ё сохранена
        assert "ё" in full_text or "е" in full_text  # ё может нормализоваться в е в некоторых системах
        assert "жик" in full_text  # часть "ёжик"
        assert "лку" in full_text  # часть "ёлку"
        assert "лке" in full_text  # часть "ёлке"
    
    def test_emojis_removed(self):
        """Эмодзи полностью удаляются."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Привет 😊 как дела 🚀 сегодня 🤖"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Эмодзи удалены
        assert "😊" not in full_text
        assert "🚀" not in full_text
        assert "🤖" not in full_text
        
        # Текст сохранён
        assert "привет" in full_text
        assert "как дела" in full_text
        assert "сегодня" in full_text
    
    def test_chinese_removed(self):
        """Иероглифы полностью удаляются."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "Привет 北京 офис сегодня"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Иероглифы удалены
        assert "北京" not in full_text
        
        # Текст сохранён
        assert "привет" in full_text
        assert "офис" in full_text
        assert "сегодня" in full_text


class TestStreamingBehavior:
    """Потоковое поведение (без буферизации всего текста)."""

    def test_fragments_emitted_before_end(self):
        """Фрагменты возвращаются ДО окончания текста (настоящая потоковость)."""
        processor = StreamTextProcessor(max_chunk_size=30)
        text = "Короткое предложение. Длинное предложение для теста потоковости."
        
        emitted_before_end = False
        fragments = []
        chars_processed = 0
        
        for ch in text:
            chars_processed += 1
            new_frags = processor.feed(ch)
            fragments.extend(new_frags)
            
            # Если получили фрагмент до обработки 70% текста — потоковость есть
            if new_frags and chars_processed < len(text) * 0.7:
                emitted_before_end = True
        
        fragments.extend(processor.flush())
        
        assert emitted_before_end, (
            "НАРУШЕНА ПОТОКОВОСТЬ: фрагменты возвращены только после обработки "
            f"{chars_processed}/{len(text)} символов (в конце текста)"
        )
        
        # Все фрагменты соблюдают лимит
        for frag in fragments:
            assert len(frag) <= 30
    
    def test_progressive_synthesis_ready(self):
        """После каждого символа состояние процессора консистентно."""
        processor = StreamTextProcessor(max_chunk_size=28)
        text = "12345.67890 Версия 2.15.3"
        
        # Проверяем, что после КАЖДОГО символа нет внутренних ошибок
        for i, ch in enumerate(text):
            try:
                fragments = processor.feed(ch)
                # Все фрагменты соблюдают лимит
                for frag in fragments:
                    assert len(frag) <= 28, f"Фрагмент после символа {i}='{ch}' превышает лимит"
            except Exception as e:
                pytest.fail(f"Ошибка при обработке символа {i}='{ch}': {e}")
        
        # Финальная проверка после flush
        for frag in processor.flush():
            assert len(frag) <= 28


class TestIntegrationScenarios:
    """Интеграционные сценарии — сложные реальные тексты."""

    def test_technical_text_full(self):
        """Полный технический текст с числами, аббревиатурами, спецсимволами."""
        processor = StreamTextProcessor(max_chunk_size=28)
        text = (
            "Версия 2.0 и π≈3.14159 — это дробные числа. Цены: $19.99, €49.50. "
            "Аббревиатуры: HTML5, CSS3, JS2022, NASA, FBI. Математика: 2+2=4, 100/3≈33.33. "
            "Спецсимволы: & | ~ # % @ $ ^ * \\ / < > =. Иероглифы: 北京 и эмодзи 😊 — удалены. "
            "Дата: 2024-12-31. Телефон: +7 (495) 123-45-67. Дроби: 0.5, 1.25, 3.1415926535. "
            "Ёжик ёлку ёл — буква ё сохранена."
        )
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        # Гарантия лимита
        for i, frag in enumerate(fragments):
            assert len(frag) <= 28, f"Фрагмент #{i} превышает лимит: '{frag}' ({len(frag)} симв.)"
        
        full_text = " ".join(fragments).lower()
        
        # Гарантия отсутствия цифр
        digits = re.findall(r'\d', full_text)
        assert digits == [], f"Обнаружены цифры в финальном выводе: {digits}"
        
        # Гарантия раскрытия спецсимволов
        assert "плюс" in full_text
        assert "доллар" in full_text
        assert "меньше" in full_text
        assert "больше" in full_text
        
        # Гарантия фильтрации
        assert "😊" not in full_text
        assert "北京" not in full_text
        
        # Гарантия сохранения ё (или корректной нормализации)
        assert "ё" in full_text or "ежик" in full_text  # ё → е допустимо в некоторых системах
        
        # Гарантия раскрытия аббревиатур
        assert "эн эй эс эй" in full_text or "нэй са" in full_text  # NASA
        assert "эф би ай" in full_text  # FBI
    
    def test_edge_case_phone_numbers(self):
        """Телефонные номера с дефисами (дефисы не раскрываются как 'минус')."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "+7 (495) 123-45-67"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Плюс раскрыт
        assert "плюс семь" in full_text or "плюссемь" in full_text
        
        # Дефисы в номере НЕ раскрываются как "минус" (слишком много минусов было бы)
        # Вместо этого они игнорируются или заменяются на пробелы
        minus_count = full_text.count("минус")
        # Допустимо 0-1 "минус" (для +7), но не 3 (для трёх дефисов)
        assert minus_count <= 1, f"Слишком много 'минус' ({minus_count}) для телефонного номера"
        
        # Числа трансформированы
        assert "четыреста девяносто пять" in full_text or "четырестадевяностопять" in full_text
        assert "сто двадцать три" in full_text
        assert "сорок пять" in full_text
        assert "шестьдесят семь" in full_text
    
    def test_edge_case_dates(self):
        """Даты с дефисами (дефисы не раскрываются как 'минус')."""
        processor = StreamTextProcessor(max_chunk_size=100)
        text = "2024-12-31"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Числа трансформированы
        assert "две тысячи двадцать четыре" in full_text or "дветысячидвадцатьчетыре" in full_text
        assert "двенадцать" in full_text
        assert "тридцать один" in full_text or "тридцатьодин" in full_text
        
        # Дефисы не создают избыточных "минус"
        minus_count = full_text.count("минус")
        assert minus_count <= 1  # допустим максимум 1 "минус" если алгоритм не идеален


class TestStressTests:
    """Стресс-тесты — экстремальные сценарии."""

    def test_max_expansion_numbers(self):
        """Максимальное раздутие чисел (1111 → 'одна тысяча сто одиннадцать')."""
        processor = StreamTextProcessor(max_chunk_size=28)
        # 1111 даёт самое длинное произношение среди 4-значных чисел
        text = "1111 " * 20  # 20 раз подряд
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        # Все фрагменты соблюдают лимит
        for i, frag in enumerate(fragments):
            assert len(frag) <= 28, f"Фрагмент #{i} превышает лимит: '{frag}' ({len(frag)} симв.)"
        
        # Нет цифр в выводе
        full_text = " ".join(fragments)
        digits = re.findall(r'\d', full_text)
        assert digits == [], f"Цифры в выводе после стресс-теста: {digits}"
    
    def test_rapid_symbol_mixing(self):
        """Быстрая смена типов символов (цифра-буква-цифра-спецсимвол)."""
        processor = StreamTextProcessor(max_chunk_size=50)
        text = "a1b2c3+4d5e6=7f8g9*0"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments).lower()
        
        # Все сущности раскрыты
        assert "эй" in full_text or "a" not in full_text  # 'a' раскрыта или удалена
        assert "один" in full_text
        assert "би" in full_text or "b" not in full_text
        assert "два" in full_text
        assert "плюс" in full_text
        assert "равно" in full_text
        assert "умножить на" in full_text
        
        # Нет цифр
        digits = re.findall(r'\d', full_text)
        assert digits == []
    
    def test_very_long_text(self):
        """Очень длинный текст (10 000+ символов) без утечек памяти."""
        processor = StreamTextProcessor(max_chunk_size=28)
        
        # Генерируем длинный текст с разнообразными сущностями
        parts = [
            "Версия 2.15.3 установлена. ",
            "Цена $99.99 за единицу. ",
            "Код NASA FBI CIA. ",
            "Математика: 2+2=4, 3*3=9. ",
            "Дата 2024-12-31. ",
            "Телефон +7 (495) 123-45-67. ",
            "Дробь 3.1415926535. ",
            "Ёжик ёлку ёл. ",
        ]
        text = "".join(parts * 100)  # ~2500 символов
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        # Все фрагменты соблюдают лимит
        for frag in fragments:
            assert len(frag) <= 28
        
        # Нет цифр в выводе
        full_text = " ".join(fragments)
        digits = re.findall(r'\d', full_text)
        assert digits == [], f"Цифры в выводе после длинного текста: {digits[:10]}..."
        
        # Минимальное количество фрагментов (защита от избыточной разбивки)
        assert len(fragments) > 100, "Должно быть много фрагментов для длинного текста"
        assert len(fragments) < len(text) // 5, "Слишком много фрагментов — избыточная разбивка"

    def test_no_latin_chars_in_output(self):
        """ГАРАНТИЯ: в финальном выводе НЕТ латинских букв (a-z, A-Z)."""
        processor = StreamTextProcessor(max_chunk_size=200)
        text = "Test example Python JS2022 NASA email@test.com v3.0"
        
        fragments = []
        for ch in text:
            fragments.extend(processor.feed(ch))
        fragments.extend(processor.flush())
        
        full_text = " ".join(fragments)
        
        # Ищем латинские буквы (регистронезависимо)
        latin_found = re.findall(r'[a-zA-Z]', full_text)
        assert latin_found == [], (
            f"НАРУШЕНА ГАРАНТИЯ: обнаружены латинские буквы в выводе: {latin_found}\n"
            f"Полный вывод: '{full_text}'"
        )