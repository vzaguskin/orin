import re
from typing import List, Optional
from trans_map import trans_map  # ← ИМПОРТИРУЕМ ИЗ ОТДЕЛЬНОГО ФАЙЛА

class StreamTextProcessor:
    """
    Потоковый обработчик текста для TTS.
    Обрабатывает символы по одному, выдаёт готовые фрагменты, когда они полны.
    Поддерживает:
    - Удаление тегов <...> (только когда закрыты)
    - Преобразование чисел в слова
    - Транслитерация латинских слов
    - Накопление и безопасная отправка фрагментов
    """

    def __init__(self, max_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.buffer = ""  # Текущий накопленный текст
        self.in_tag = False   # Внутри тега <...>?
        self.tag_start = -1   # Позиция начала тега в buffer
        self.last_sent_pos = 0  # Позиция, до которой мы уже отправили

        # Словарь транслитераций (импортируется из trans_map.py)
        self.trans_map = trans_map

        # Регулярные выражения для поиска чисел и слов
        self.number_pattern = re.compile(r'\b\d+\b')
        self.word_pattern = re.compile(r'\b[A-Za-z]+\b')

        # Теги, которые нужно удалять
        self.tag_patterns = [
            re.compile(r'<\s*([a-zA-Z][^>]*)\s*>'),  # открывающий тег
            re.compile(r'<\s*/\s*([a-zA-Z]+)\s*>'),  # закрывающий тег
        ]

    def feed(self, char: str) -> List[str]:
        """
        Подаёт один символ. Возвращает список готовых фрагментов для отправки.
        Если ничего не готово — возвращает пустой список.
        """
        if not char:
            return []

        self.buffer += char
        fragments = []

        # 1. ПРОВЕРКА ТЕГОВ — В РЕАЛЬНОМ ВРЕМЕНИ
        if not self.in_tag:
            # Проверяем, не начался ли тег
            for pattern in self.tag_patterns:
                match = pattern.search(self.buffer[self.last_sent_pos:])
                if match:
                    start_in_buffer = match.start() + self.last_sent_pos
                    end_in_buffer = match.end() + self.last_sent_pos

                    # Если это закрывающий тег — завершаем тег
                    if pattern.pattern.startswith(r'<\s*/'):
                        tag_name = match.group(1)
                        # Проверяем, был ли открыт такой тег?
                        if self.in_tag and self.tag_start != -1:
                            # Мы закрываем тег — вырезаем всё от начала до конца
                            self.buffer = self.buffer[:self.tag_start] + self.buffer[end_in_buffer:]
                            self.last_sent_pos = self.tag_start  # Сдвигаем начало
                            self.in_tag = False
                            self.tag_start = -1
                            # НЕ отправляем фрагмент — он уже вырезан
                            break
                    else:
                        # Это открывающий тег
                        self.in_tag = True
                        self.tag_start = start_in_buffer
                        # Вырезаем тег из буфера, но не отправляем
                        self.buffer = self.buffer[:start_in_buffer] + self.buffer[end_in_buffer:]
                        self.last_sent_pos = start_in_buffer
                        break
        else:
            # Проверяем, не закрылся ли тег
            for pattern in self.tag_patterns:
                if pattern.pattern.startswith(r'<\s*/'):
                    match = pattern.search(self.buffer[self.last_sent_pos:])
                    if match:
                        end_in_buffer = match.end() + self.last_sent_pos
                        # Вырезаем тег
                        self.buffer = self.buffer[:self.tag_start] + self.buffer[end_in_buffer:]
                        self.last_sent_pos = self.tag_start
                        self.in_tag = False
                        self.tag_start = -1
                        break

        # 2. ПРОВЕРКА ЧИСЕЛ — ПОКА НЕ ЗАВЕРШИЛИСЬ
        # Ищем числа, которые могут быть частично введены
        # Например: "5" → "5" → "5 " → "50" → "50 " → "50 г"
        # Мы не хотим обрабатывать "5" пока не пришёл пробел или знак препинания
        if self.buffer:
            # Проверяем, есть ли число в конце
            last_chars = self.buffer[-5:]  # достаточно 5 символов
            if self.number_pattern.search(last_chars):
                # Проверяем, заканчивается ли число на нецифровой символ
                if len(self.buffer) > 0:
                    last_char = self.buffer[-1]
                    if not last_char.isdigit() and last_char not in '.,!?;: ':
                        # Это не число — просто символ
                        pass
                    elif last_char in '.,!?;: \n\t\r':
                        # Число завершено — обрабатываем его
                        processed = self._process_numbers_in_buffer()
                        if processed:
                            fragments.append(processed)

        # 3. ПРОВЕРКА ТРАНСЛИТА — ПОКА НЕ ЗАВЕРШИЛИСЬ СЛОВА
        # Проверяем, есть ли в конце слова, которые могут быть в trans_map
        if self.buffer:
            # Проверяем последние 10 символов на совпадение с ключами trans_map
            tail = self.buffer[-10:].lower()  # не чувствительный к регистру
            for word in self.trans_map.keys():
                if len(word) <= len(tail):
                    if tail.endswith(word.lower()):
                        # Проверяем, что после слова идёт не буква/цифра — значит, слово закончилось
                        if len(self.buffer) >= len(word):
                            start_idx = len(self.buffer) - len(word)
                            # Проверяем, что это целое слово — до и после не буквы/цифры
                            before = self.buffer[start_idx - 1:start_idx] if start_idx > 0 else ""
                            after = self.buffer[start_idx + len(word):start_idx + len(word) + 1] if start_idx + len(word) < len(self.buffer) else ""

                            if (not before or not before.isalnum()) and (not after or not after.isalnum()):
                                # Слово завершено — заменяем его
                                replacement = self.trans_map[word]
                                self.buffer = self.buffer[:start_idx] + replacement + self.buffer[start_idx + len(word):]
                                # Не отправляем — ждём завершения фрагмента
                                break

        # 4. ПРОВЕРКА НА ГОТОВЫЙ ФРАГМЕНТ — ПО ДЛИНЕ ИЛИ ЗНАКУ
        if len(self.buffer) >= self.max_chunk_size:
            # Отправляем фрагмент
            fragment = self.buffer[:self.max_chunk_size]
            self.buffer = self.buffer[self.max_chunk_size:]
            self.last_sent_pos = 0
            fragments.append(fragment.strip())
        elif len(self.buffer) > 0 and self.buffer[-1] in '.!?':
            # Отправляем, если закончилось предложение
            fragment = self.buffer
            self.buffer = ""
            self.last_sent_pos = 0
            fragments.append(fragment.strip())

        return fragments

    def _process_numbers_in_buffer(self) -> Optional[str]:
        """Обрабатывает числа в буфере и возвращает готовый фрагмент, если есть"""
        if not self.buffer:
            return None

        # Ищем все числа в буфере
        matches = list(self.number_pattern.finditer(self.buffer))
        if not matches:
            return None

        # Берём последнее число
        last_match = matches[-1]
        start, end = last_match.span()

        # Проверяем, что за числом идёт не цифра — значит, число завершено
        if end < len(self.buffer) and self.buffer[end].isdigit():
            return None  # Не завершено — ждём дальше

        # Заменяем число
        num_str = self.buffer[start:end]
        try:
            word = num2words(int(num_str), lang='ru')
        except:
            return None

        # Заменяем в буфере
        self.buffer = self.buffer[:start] + word + self.buffer[end:]

        # Если буфер стал длиннее max_chunk_size — отдаём фрагмент
        if len(self.buffer) >= self.max_chunk_size:
            fragment = self.buffer[:self.max_chunk_size]
            self.buffer = self.buffer[self.max_chunk_size:]
            return fragment.strip()

        return None

    def flush(self) -> List[str]:
        """
        Возвращает оставшийся текст как один фрагмент.
        Вызывается при завершении потока.
        """
        fragments = []
        if self.buffer.strip():
            # Обрабатываем числа и транслит в остатке
            self._process_numbers_in_buffer()
            # Применяем транслит к остатку
            for word, replacement in self.trans_map.items():
                if word.lower() in self.buffer.lower():
                    self.buffer = re.sub(
                        r'\b' + re.escape(word) + r'\b',
                        replacement,
                        self.buffer,
                        flags=re.IGNORECASE
                    )
            fragments.append(self.buffer.strip())
            self.buffer = ""
        return fragments

    def reset(self):
        """Сбрасывает состояние — для нового запроса"""
        self.buffer = ""
        self.in_tag = False
        self.tag_start = -1
        self.last_sent_pos = 0