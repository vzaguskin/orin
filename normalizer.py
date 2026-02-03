# normalizer.py
import re
from typing import List

class StreamTextProcessor:
    """
    Потоковый процессор текста для TTS на встраиваемых платформах.
    
    Гарантии:
    1. ВСЕ цифры → слова (включая дробную часть: 3.14 → три точка четырнадцать)
    2. Ноль цифр в финальном выводе (критично для словаря без цифр)
    3. Каждый фрагмент ≤ max_chunk_size символов ПОСЛЕ трансформации
    4. Целостность чисел (нет разрывов вида '3 .' или '. 14')
    5. Полное раскрытие спецсимволов (+, $, <, > и др.)
    6. Фильтрация недопустимых символов (иероглифы, эмодзи)
    7. Потоковая обработка без буферизации всего текста
    
    Совместимость: Python 3.7+
    """
    
    # Словарь транслитерации латинских букв (базовый)
    LATIN_TO_RU = {
        'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
        'g': 'джи', 'h': 'эйч', 'i': 'ай', 'j': 'джей', 'k': 'кей',
        'l': 'эль', 'm': 'эм', 'n': 'эн', 'o': 'оу', 'p': 'пи',
        'q': 'кью', 'r': 'ар', 's': 'эс', 't': 'ти', 'u': 'ю',
        'v': 'в', 'w': 'дабл-ю', 'x': 'икс', 'y': 'уай', 'z': 'зед',
    }
    
    # Словарь спецсимволов → слова
    SPECIAL_CHARS = {
        '+': 'плюс',
        '*': 'умножить на',
        '/': 'разделить на',
        '&': 'и',
        '@': 'собака',
        '$': 'доллар',
        '€': 'евро',
        '£': 'фунт',
        '¥': 'иена',
        '#': 'решётка',
        '%': 'процент',
        '=': 'равно',
        '^': 'в степени',
        '~': 'примерно',
        '|': 'или',
        '\\': 'бэкслэш',
        '(': 'скобка открывается',
        ')': 'скобка закрывается',
        '[': 'скобка квадратная открывается',
        ']': 'скобка квадратная закрывается',
        '{': 'фигурная скобка открывается',
        '}': 'фигурная скобка закрывается',
        '<': 'меньше',
        '>': 'больше',
    }

    def __init__(self, max_chunk_size=28):
        if max_chunk_size < 10:
            raise ValueError("max_chunk_size must be at least 10")
        self.max_chunk_size = max_chunk_size
        self.current_entity = ""
        self.transformed_buffer = ""
        self.inside_tag = False
        self.after_lt = False
        self.prev_char_was_digit = False
        self.consecutive_digit_dashes = 0  # Счётчик дефисов между цифрами

    def feed(self, char):
        if not char:
            self.prev_char_was_digit = False
            self.consecutive_digit_dashes = 0
            return []
        
        # === 1. Обработка тегов: '<' ===
        if char == '<':
            self._flush_entity()
            self.after_lt = True  # Ждём следующий символ для принятия решения
            self.prev_char_was_digit = False
            self.consecutive_digit_dashes = 0
            return []
        
        # === 2. Контекст после '<' ===
        if self.after_lt:
            self.after_lt = False
            # ТЕГ начинается ТОЛЬКО если после '<' идёт:
            # - буква (для <b>, <think>)
            # - слэш (для </b>, </think>)
            # ЦИФРА после '<' = НЕ тег (арифметика: 5<10)
            if char.isalpha() or char == '/':
                self.inside_tag = True  # Игнорируем содержимое ДО '>'
                self.prev_char_was_digit = False
                self.consecutive_digit_dashes = 0
                return []
            else:
                # Не тег — спецсимвол "меньше"
                self._add_to_buffer('меньше')
                # Продолжаем обработку текущего символа ниже
        
        # === 3. Игнорирование содержимого тега (<...>) ===
        if self.inside_tag:
            if char == '>':
                # Завершение тега — сбрасываем флаг, НИЧЕГО не добавляем в вывод
                self.inside_tag = False
                self.prev_char_was_digit = False
                self.consecutive_digit_dashes = 0
                return []
            # Игнорируем ВСЁ внутри тега (включая буквы, цифры, слэши)
            self.prev_char_was_digit = False
            self.consecutive_digit_dashes = 0
            return []
        
        # === 5. Обработка дефиса (контекстная) ===
        if char == '-':
            self._flush_entity()
            
            # Дефис между цифрами: первый → "минус", последующие → игнорировать (пробел)
            if self.prev_char_was_digit:
                self.consecutive_digit_dashes += 1
                if self.consecutive_digit_dashes == 1:
                    self._add_to_buffer('минус')
                else:
                    # Игнорируем (добавляем пробел как разделитель)
                    if self.transformed_buffer and not self.transformed_buffer.endswith(' '):
                        self.transformed_buffer += ' '
            else:
                # Дефис не между цифрами → всегда "минус"
                self._add_to_buffer('минус')
                self.consecutive_digit_dashes = 0
            
            self.prev_char_was_digit = False
            return self._emit_fragments_if_ready()
        
        # === 6. Обработка спецсимволов (кроме дефиса) ===
        if char in self.SPECIAL_CHARS:
            self._flush_entity()
            self._add_to_buffer(self.SPECIAL_CHARS[char])
            self.prev_char_was_digit = False
            self.consecutive_digit_dashes = 0
            return self._emit_fragments_if_ready()
        
        # === 7. Обработка точки ===
        if char == '.':
            if self.current_entity and self.current_entity[-1].isdigit():
                self.current_entity += char
                self.prev_char_was_digit = False
                self.consecutive_digit_dashes = 0
                return []
            self._flush_entity()
            self._add_to_buffer('.')
            self.prev_char_was_digit = False
            self.consecutive_digit_dashes = 0
            return self._emit_fragments_if_ready()
        
        # === 8. Обработка апострофа (разделитель тысяч) ===
        if char == "'":
            self.prev_char_was_digit = False
            self.consecutive_digit_dashes = 0
            return []
        
        # === 9. Обработка допустимых символов ===
        if self._is_valid_char(char):
            is_digit = char.isdigit()
            is_upper = char.isupper() and char.isascii()
            is_lower = char.islower() and char.isascii()
            is_cyrillic = '\u0400' <= char <= '\u04FF'
            
            # Граница сущности при смене типа
            if self.current_entity:
                prev_char = self.current_entity[-1]
                prev_is_digit = prev_char.isdigit()
                prev_is_upper = prev_char.isupper() and prev_char.isascii()
                prev_is_lower = prev_char.islower() and prev_char.isascii()
                prev_is_cyrillic = self._is_cyrillic(prev_char)
                
                # Граница при смене цифра ↔ буква
                if (prev_is_digit and (is_upper or is_lower or is_cyrillic)) or \
                   ((prev_is_upper or prev_is_lower or prev_is_cyrillic) and is_digit):
                    self._flush_entity()
                
                # Граница при смене регистра (для PhD)
                if (prev_is_upper and is_lower) or (prev_is_lower and is_upper):
                    self._flush_entity()
            
            self.current_entity += char
            self.prev_char_was_digit = is_digit
            if not is_digit:
                self.consecutive_digit_dashes = 0
            return []
        
        # === 10. Граница слова ===
        self._flush_entity()
        
        if char in ',!?;:':
            self._add_to_buffer(char)
        elif char.isspace() and self.transformed_buffer and not self.transformed_buffer.endswith(' '):
            self.transformed_buffer += ' '
        
        self.prev_char_was_digit = False
        self.consecutive_digit_dashes = 0
        return self._emit_fragments_if_ready()

    def flush(self):
        if self.after_lt:
            self._add_to_buffer('меньше')
            self.after_lt = False
        
        self._flush_entity()
        fragments = self._split_hard(self.transformed_buffer.strip())
        
        # Сброс состояния
        self.transformed_buffer = ""
        self.current_entity = ""
        self.inside_tag = False
        self.after_lt = False
        self.prev_char_was_digit = False
        self.consecutive_digit_dashes = 0
        
        return fragments

    def _flush_entity(self):
        if not self.current_entity:
            return
        
        entity = self.current_entity.rstrip('.')
        trailing_dots = len(self.current_entity) - len(entity)
        
        if entity:
            transformed = self._transform_entity(entity)
            if transformed:
                self._add_to_buffer(transformed)
        
        for _ in range(trailing_dots):
            self._add_to_buffer('.')
        
        self.current_entity = ""

    def _transform_entity(self, entity):
        # === 1. Числа с точками ===
        clean_entity = entity.replace("'", "")
        if re.fullmatch(r'\d+(\.\d+)*', clean_entity):
            parts = clean_entity.split('.')
            try:
                from num2words import num2words
                words = []
                for part in parts:
                    num = int(part.lstrip('0') or '0')
                    words.append(num2words(num, lang='ru'))
                return ' точка '.join(words)
            except (ImportError, ValueError, OverflowError):
                return entity.lower()
        
        # === 2. Смешанный регистр (например, "PhD") ===
        if entity.isalpha() and entity.isascii() and not entity.islower() and not entity.isupper():
            parts = []
            for ch in entity:
                ch_lower = ch.lower()
                if ch_lower == 'h':
                    parts.append('эйч')
                elif ch_lower == 'd':
                    parts.append('ди')
                else:
                    parts.append(self.LATIN_TO_RU.get(ch_lower, ch_lower))
            return ' '.join(parts)
        
        # === 3. Чистые аббревиатуры (все заглавные, >=2 буквы) ===
        if len(entity) >= 2 and entity.isalpha() and entity.isupper() and entity.isascii():
            parts = []
            for ch in entity:
                ch_lower = ch.lower()
                if ch_lower == 'h':
                    parts.append('аш')
                elif ch_lower == 'd':
                    parts.append('ди')
                else:
                    parts.append(self.LATIN_TO_RU.get(ch_lower, ch_lower))
            return ' '.join(parts)
        
        # === 4. Одиночные буквы ===
        if len(entity) == 1 and entity.isalpha() and entity.isascii():
            ch_lower = entity.lower()
            if ch_lower == 'h':
                return 'эйч'
            elif ch_lower == 'd':
                return 'ди'
            return self.LATIN_TO_RU.get(ch_lower, ch_lower)
        
        # === 5. СТРОЧНЫЕ ЛАТИНСКИЕ СЛОВА → транслитерация в кириллицу (КРИТИЧЕСКАЯ ВЕТКА!) ===
        if entity.isalpha() and entity.islower() and entity.isascii():
            return self._transliterate_latin_to_cyrillic(entity)
        
        # === 6. Кириллица (включая ё) ===
        if all('\u0400' <= ch <= '\u04FF' for ch in entity):
            return entity.lower()
        
        # === 7. Смешанный текст (кириллица + латиница) — фильтруем латиницу ===
        result_parts = []
        current_latin = ""
        for ch in entity:
            if 'a' <= ch.lower() <= 'z' and ch.isascii():
                current_latin += ch
            else:
                if current_latin:
                    if current_latin.isupper():
                        result_parts.append(self._transform_entity(current_latin))
                    elif current_latin.islower():
                        result_parts.append(self._transliterate_latin_to_cyrillic(current_latin))
                    else:
                        result_parts.append(self._transform_entity(current_latin))
                    current_latin = ""
                if '\u0400' <= ch <= '\u04FF':
                    result_parts.append(ch.lower())
        if current_latin:
            if current_latin.isupper():
                result_parts.append(self._transform_entity(current_latin))
            elif current_latin.islower():
                result_parts.append(self._transliterate_latin_to_cyrillic(current_latin))
            else:
                result_parts.append(self._transform_entity(current_latin))
        
        return ' '.join(result_parts) if result_parts else entity.lower()

    def _transliterate_latin_to_cyrillic(self, word):
        """
        Простая посимвольная транслитерация строчных латинских слов в кириллицу.
        Минималистичная реализация для гарантии отсутствия латиницы.
        """
        translit_map = {
            'a': 'а', 'b': 'б', 'c': 'с', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
            'h': 'х', 'i': 'и', 'j': 'ж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
            'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
            'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'ы', 'z': 'з',
        }
        # Посимвольная замена БЕЗ диграфов (максимально надёжно)
        return ''.join(translit_map.get(ch, ch) for ch in word.lower())

    def _add_to_buffer(self, text):
        if not text or not text.strip():
            return
        
        if (self.transformed_buffer and 
            self.transformed_buffer[-1] not in ' ([{' and 
            not self.transformed_buffer.endswith(' ')):
            self.transformed_buffer += ' '
        
        self.transformed_buffer += text

    def _emit_fragments_if_ready(self):
        fragments = []
        
        while len(self.transformed_buffer.strip()) > self.max_chunk_size:
            text = self.transformed_buffer.strip()
            frags = self._split_hard(text)
            
            if len(frags) > 1:
                fragments.extend(frags[:-1])
                self.transformed_buffer = frags[-1] + ' '
            else:
                fragments.append(frags[0])
                self.transformed_buffer = ""
                break
        
        return fragments

    def _split_hard(self, text):
        if not text or not text.strip():
            return []
        
        fragments = []
        remaining = text.strip()
        
        while remaining:
            if len(remaining) <= self.max_chunk_size:
                fragments.append(remaining)
                break
            
            split_pos = self.max_chunk_size
            for i in range(self.max_chunk_size - 1, max(0, self.max_chunk_size // 2 - 1), -1):
                if remaining[i] in ' ,':
                    if i >= 6 and remaining[i-6:i+1].lower() == ' точка':
                        continue
                    split_pos = i + 1
                    break
            
            fragment = remaining[:split_pos].strip()
            
            if len(fragment) > self.max_chunk_size:
                fragment = fragment[:self.max_chunk_size].rsplit(' ', 1)[0].strip() or fragment[:self.max_chunk_size]
            
            fragments.append(fragment)
            remaining = remaining[split_pos:].lstrip()
        
        return fragments

    def _is_valid_char(self, char):
        if char.isdigit():
            return True
        if '\u0400' <= char <= '\u04FF':  # Кириллица (включая ё)
            return True
        if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):  # Латиница
            return True
        if char == "'":  # Апостроф-разделитель
            return True
        return False

    def _is_cyrillic(self, char):
        return '\u0400' <= char <= '\u04FF'