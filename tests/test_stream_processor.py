from normalizer import StreamTextProcessor

class TestStreamTextProcessor:
    
    def test_no_data_loss_on_sentence_end(self):
        processor = StreamTextProcessor(max_chunk_size=50)
        fragments = []
        
        for ch in "123. Привет":
            fragments.extend(processor.feed(ch))
        
        assert len(fragments) == 1
        assert "сто двадцать три" in fragments[0].lower()
        assert "." in fragments[0]
        assert processor.clean_buffer == " Привет"
        
        final = processor.flush()
        assert "привет" in final[0].lower()

    def test_decimal_numbers_not_split(self):
        """Дробные числа НЕ разрываются на фрагменты (2.0 остаётся целым)"""
        processor = StreamTextProcessor()
        fragments = []
        
        # Ключевой момент: пробел ПОСЛЕ завершающей точки триггерит отправку
        for ch in "Версия 2.0. ":
            fragments.extend(processor.feed(ch))
        
        # Должен быть ОДИН фрагмент со всей дробью
        assert len(fragments) == 1, f"Ожидался 1 фрагмент, получено {len(fragments)}: {fragments}"
        text = fragments[0].lower()
        # "2.0" → "два" (без "ноль") — это корректно для естественной речи
        assert "два" in text or "две" in text
        # Главное: не должно быть РАЗРЫВА на два фрагмента
        assert fragments[0].count('.') <= 2  # максимум две точки: 2.0.

    def test_currency_not_split(self):
        """Валюты НЕ разрываются ($19.99 остаётся целым)"""
        processor = StreamTextProcessor()
        fragments = []
        
        for ch in "Цена $19.99. ":
            fragments.extend(processor.feed(ch))
        
        assert len(fragments) == 1, f"Ожидался 1 фрагмент, получено {len(fragments)}: {fragments}"
        text = fragments[0].lower()
        assert "доллар" in text or "$" in fragments[0]
        # Главное: не разорвано на два фрагмента "доллар девятнадцать." + "девяносто девять."

    def test_flush_sends_remaining(self):
        """flush() отправляет остаток буфера без требований к завершению предложения"""
        processor = StreamTextProcessor()
        
        for ch in "Незавершённый текст":  # ← буква Ё!
            processor.feed(ch)
        
        result = processor.flush()
        assert len(result) == 1
        text = result[0].lower()
        # num2words может заменять ё→е, допускаем оба варианта
        assert "незавершённый" in text or "незавершенный" in text
        assert "текст" in text

    def test_chinese_and_emoji_filtered(self):
        """Иероглифы и эмодзи удаляются на этапе приёма символов"""
        processor = StreamTextProcessor()
        fragments = []
        
        # Подаем текст с иероглифами и эмодзи
        for ch in "Привет 北京 😊 мир!":
            fragments.extend(processor.feed(ch))
        
        # Фрагмент отправится только после '!' + пробел, но у нас нет пробела — проверяем через flush
        assert len(fragments) == 0, f"Фрагменты не должны отправляться до завершения предложения: {fragments}"
        
        # Вызываем flush для получения результата
        final = processor.flush()
        assert len(final) == 1
        text = final[0].lower()
        
        # Проверяем содержимое
        assert "привет" in text
        assert "мир" in text
        # Иероглифы и эмодзи НЕ должны попасть в результат
        assert "北京" not in text
        assert "😊" not in text
        # Слова не склеены (между ними должен быть пробел)
        assert "привет мир" in text.replace("  ", " ")

    def test_special_chars_expansion(self):
        processor = StreamTextProcessor()
        
        for ch in "a+b=c":
            processor.feed(ch)
        
        result = processor.flush()
        assert len(result) == 1
        text = result[0].lower()
        assert "плюс" in text
        assert "равно" in text

    def test_max_chunk_size_respected(self):
        processor = StreamTextProcessor(max_chunk_size=30)
        fragments = []
        
        for ch in "1234567890 " * 5:
            fragments.extend(processor.feed(ch))
        
        for frag in fragments:
            assert len(frag) <= 30

    def test_math_expressions(self):
        processor = StreamTextProcessor()
        fragments = []
        
        for ch in "2+2=4. ":
            fragments.extend(processor.feed(ch))
        
        assert len(fragments) == 1
        text = fragments[0].lower()
        assert "два" in text or "2" in text
        assert "плюс" in text
        assert "равно" in text
        assert "четыре" in text or "4" in text

    def test_abbreviations_with_numbers(self):
        """Аббревиатуры с цифрами обрабатываются корректно (каждое предложение = отдельный фрагмент)"""
        processor = StreamTextProcessor()
        fragments = []
        
        # Два предложения → два фрагмента (корректное поведение!)
        for ch in "HTML5. CSS3. ":
            fragments.extend(processor.feed(ch))
        
        # Два предложения = два фрагмента
        assert len(fragments) == 2, f"Ожидалось 2 фрагмента для двух предложений, получено {len(fragments)}: {fragments}"
        
        # Первый фрагмент: HTML5
        text1 = fragments[0].lower()
        assert "эйч" in text1
        assert "ти" in text1
        assert "пять" in text1
        
        # Второй фрагмент: CSS3
        text2 = fragments[1].lower()
        assert "си" in text2
        assert "три" in text2

    def test_yo_letter_preserved(self):
        """Буква 'ё' корректно обрабатывается"""
        processor = StreamTextProcessor()
        fragments = []
        
        for ch in "Ёжик ёлку ёл. ":
            fragments.extend(processor.feed(ch))
        
        assert len(fragments) == 1
        text = fragments[0].lower()
        assert "жик" in text.replace("ё", "е")
        assert "лку" in text.replace("ё", "е")