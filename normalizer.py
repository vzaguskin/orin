import re
from typing import List, Optional
from num2words import num2words

# Импорт словаря транслитераций
from trans_map import trans_map

# Спецсимволы → произношение
SPECIAL_CHARS = {
    '+': ' плюс ', '-': ' минус ', '*': ' умножить на ', '/': ' разделить на ',
    '&': ' и ', '@': ' собака ', '$': ' доллар ', '#': ' решётка ', '%': ' процент ',
    '=': ' равно ', '<': ' меньше ', '>': ' больше ', '^': ' в степени ',
    '~': ' примерно ', '|': ' или ', '\\': ' обратный слэш ', '`': ' гравис ',
    '"': ' кавычки ', "'": ' апостроф ', '(': ' скобка открывается ',
    ')': ' скобка закрывается ', '≈': ' примерно равно ', '–': ' тире ', '—': ' длинное тире ',
    '€': ' евро ', '£': ' фунт ', '¥': ' иена ', '®': ' зарегистрировано ', '©': ' копирайт ',
}

# Транслит латинских букв
LATIN_TO_RU = {
    'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
    'g': 'джи', 'h': 'эйч', 'i': 'ай', 'j': 'джей', 'k': 'кей',
    'l': 'эль', 'm': 'эм', 'n': 'эн', 'o': 'оу', 'p': 'пи',
    'q': 'кью', 'r': 'ар', 's': 'эс', 't': 'ти', 'u': 'ю',
    'v': 'ви', 'w': 'дабл-ю', 'x': 'икс', 'y': 'уай', 'z': 'зед',
}

class StreamTextProcessor:
    def __init__(self, max_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.clean_buffer = ""
        self.inside_tag = False

    def _is_allowed_char(self, char: str) -> bool:
        """
        Проверяет, разрешён ли символ для TTS.
        Разрешаем: кириллицу (включая ё/Ё), латиницу, цифры, пробелы,
        знаки препинания и спецсимволы из маппинга.
        """
        if char.isspace():
            return True
        if char in SPECIAL_CHARS:
            return True
        if char in '.!?,;:':
            return True
        # Кириллица: явная проверка всех букв включая ё/Ё
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ':
            return True
        # Латиница
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            return True
        if char.isdigit():
            return True
        return False

    def feed(self, char: str) -> List[str]:
        if not char:
            return []

        # === 1. Фильтрация тегов ===
        if char == '<':
            self.inside_tag = True
            return []
        elif char == '>' and self.inside_tag:
            self.inside_tag = False
            return []
        elif self.inside_tag:
            return []

        # === 2. Фильтрация "мусора" для TTS (иероглифы, эмодзи, неизвестные символы) ===
        if not self._is_allowed_char(char):
            # Заменяем на пробел, чтобы не склеивать слова
            char = ' '

        # === 3. Добавление символа в буфер ===
        self.clean_buffer += char
        fragments = []

        # === 4. Отправка по завершённым предложениям (ТОЛЬКО .!? + пробел) ===
        while True:
            end_pos = self._find_safe_sentence_end_in_raw(self.clean_buffer)
            if end_pos is None:
                break
            
            raw_fragment = self.clean_buffer[:end_pos]
            self.clean_buffer = self.clean_buffer[end_pos:]
            
            processed = self._transform(raw_fragment)
            if processed.strip():
                fragments.append(processed.strip())

        # === 5. Аварийная отсечка по длине ===
        if len(self.clean_buffer) > self.max_chunk_size * 2:
            cutoff = self._find_safe_cutoff_in_raw(self.clean_buffer, self.max_chunk_size)
            if cutoff > 0:
                raw_fragment = self.clean_buffer[:cutoff]
                self.clean_buffer = self.clean_buffer[cutoff:]
                
                processed = self._transform(raw_fragment)
                if processed.strip():
                    fragments.append(processed.strip())

        return fragments

    def _find_safe_sentence_end_in_raw(self, text: str) -> Optional[int]:
        """
        Отправляем фрагмент ТОЛЬКО при: знак препинания (.!?) + пробел/перенос.
        Это гарантирует целостность дробей и валют без сложных эвристик.
        """
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '.!?':
                # Требуем пробел/перенос СРАЗУ после знака препинания
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1  # включаем знак препинания, но не пробел
        return None

    def _find_safe_cutoff_in_raw(self, text: str, max_len: int) -> int:
        """Безопасная отсечка по пробелам/знакам препинания"""
        if len(text) <= max_len:
            return len(text)
        
        for i in range(min(max_len, len(text)) - 1, -1, -1):
            if text[i].isspace() or text[i] in '.!?,;:':
                return i + 1
        
        return max_len

    def _transform(self, text: str) -> str:
        if not text:
            return text

        # 1. Спецсимволы
        for char, repl in SPECIAL_CHARS.items():
            text = text.replace(char, repl)

        # 2. Денежные суммы с дробями: $19.99 → "девятнадцать долларов девяносто девять центов"
        def replace_currency(match):
            currency = match.group(1)
            whole = match.group(2)
            frac = match.group(3)
            try:
                w_num = int(whole)
                w = num2words(w_num, lang='ru')
                f = num2words(int(frac), lang='ru')
                curr_map = {'$': 'доллар', '€': 'евро', '£': 'фунт'}
                curr_word = curr_map.get(currency, 'валюта')
                
                # Склонение валюты
                if w_num % 10 == 1 and w_num % 100 != 11:
                    curr_form = curr_word  # 1 доллар
                elif 2 <= w_num % 10 <= 4 and not (12 <= w_num % 100 <= 14):
                    curr_form = curr_word + 'а'  # 2-4 доллара
                else:
                    curr_form = curr_word + 'ов'  # 5+ долларов
                
                return f"{w} {curr_form} {f} центов"
            except:
                return match.group()
        
        text = re.sub(r'([\$€£])(\d+)\.(\d{2})', replace_currency, text)

        # 3. Десятичные дроби: 2.0 → "два" (дробная часть 0 игнорируется для естественности)
        def replace_decimal(match):
            try:
                num_str = match.group()
                if '.' not in num_str:
                    return num2words(int(num_str), lang='ru')
                
                whole_part, frac_part = num_str.split('.')
                whole = int(whole_part)
                frac = int(frac_part)
                
                # Особый случай: 2.0 → "два" (без "ноль десятых" — неестественно для речи)
                if frac == 0:
                    return num2words(whole, lang='ru')
                
                whole_word = num2words(whole, lang='ru')
                frac_word = num2words(frac, lang='ru')
                
                # Название дробной части
                frac_len = len(frac_part.lstrip('0') or '0')
                if frac_len == 1:
                    frac_name = 'десятых'
                elif frac_len == 2:
                    frac_name = 'сотых'
                else:
                    frac_name = 'тысячных'
                
                return f"{whole_word} целых {frac_word} {frac_name}"
            except:
                return match.group()
        
        text = re.sub(r'\b\d+\.\d+\b', replace_decimal, text)

        # 4. Целые числа
        def replace_number(match):
            try:
                num = int(match.group())
                return num2words(num, lang='ru')
            except:
                return match.group()
        text = re.sub(r'\b\d+\b', replace_number, text)

        # 5. Аббревиатуры (только ЗАГЛАВНЫЕ буквы, 2+ символов, опционально цифры)
        def replace_abbreviation(match):
            word = match.group()
            letters = ''.join(ch for ch in word if ch.isalpha())
            digits = ''.join(ch for ch in word if ch.isdigit())
            
            if not letters:
                return word
            
            # Транслит по буквам
            translit = ' '.join(LATIN_TO_RU.get(ch.lower(), ch) for ch in letters)
            if digits:
                try:
                    digit_words = num2words(int(digits), lang='ru')
                    return f"{translit} {digit_words}"
                except:
                    return f"{translit} {digits}"
            return translit
        
        text = re.sub(r'\b([A-Z]{2,}[0-9]*)\b', replace_abbreviation, text)

        # 6. Остальные латинские слова
        def replace_latin_word(match):
            word = match.group()
            lower_word = word.lower()
            
            if lower_word in trans_map:
                return trans_map[lower_word]
            
            # Транслит по буквам, сохраняя только буквы
            return ' '.join(
                LATIN_TO_RU.get(ch.lower(), ch) 
                for ch in word 
                if ch.isalpha()
            )
        
        text = re.sub(r'\b[a-zA-Z]+\b', replace_latin_word, text)

        # 7. Финальная очистка пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def flush(self) -> List[str]:
        if not self.clean_buffer:
            return []
        
        processed = self._transform(self.clean_buffer)
        self.clean_buffer = ""
        result = processed.strip()
        return [result] if result else []

    def reset(self):
        self.clean_buffer = ""
        self.inside_tag = False