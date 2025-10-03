import re
from typing import List, Optional
from num2words import num2words

# Импорт словаря транслитераций
from trans_map import trans_map

# Спецсимволы → произношение
SPECIAL_CHARS = {
    '+': ' плюс ',
    '-': ' минус ',
    '*': ' умножить на ',
    '/': ' разделить на ',
    '&': ' и ',
    '@': ' собака ',
    '$': ' доллар ',
    '#': ' решётка ',
    '%': ' процент ',
    '=': ' равно ',
    '<': ' меньше ',
    '>': ' больше ',
    '^': ' в степени ',
    '~': ' примерно ',
    '|': ' или ',
    '\\': ' обратный слэш ',
    '`': ' гравис ',
    '"': ' кавычки ',
    "'": ' апостроф ',
    '(': ' скобка открывается ',
    ')': ' скобка закрывается ',
}

# Транслит латинских букв (для аббревиатур и неизвестных слов)
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
        self.clean_buffer = ""  # Только текст вне тегов
        self.inside_tag = False

    def feed(self, char: str) -> List[str]:
        if not char:
            return []

        # === 1. Удаление тегов в реальном времени ===
        if char == '<':
            self.inside_tag = True
        elif char == '>' and self.inside_tag:
            self.inside_tag = False
            return []  # тег завершён — ничего не добавляем
        elif self.inside_tag:
            return []  # внутри тега — игнорируем символ
        else:
            # Вне тега — добавляем символ
            self.clean_buffer += char

        # === 2. Полное преобразование буфера ===
        processed = self._transform(self.clean_buffer)
        fragments = []

        # === 3. Проверка: есть ли завершённое предложение? ===
        end_pos = self._find_safe_sentence_end(processed)
        if end_pos is not None:
            fragment = processed[:end_pos].strip()
            if fragment:
                fragments.append(fragment)
            self.clean_buffer = ""
            return fragments

        # === 4. Аварийная отправка по длине (только если сильно превышен лимит) ===
        if len(processed) >= self.max_chunk_size:
            cutoff = self._find_safe_cutoff(processed, self.max_chunk_size)
            fragment = processed[:cutoff].strip()
            if fragment:
                fragments.append(fragment)
            # После отправки по длине — сбрасываем буфер (небольшая потеря контекста допустима)
            self.clean_buffer = ""

        return fragments

    def _transform(self, text: str) -> str:
        if not text:
            return text

        # Спецсимволы
        for char, repl in SPECIAL_CHARS.items():
            text = text.replace(char, repl)

        # Числа → слова
        def replace_number(match):
            try:
                num = int(match.group())
                return num2words(num, lang='ru')
            except (ValueError, OverflowError):
                return match.group()
        text = re.sub(r'\b\d+\b', replace_number, text)

        # Латинские слова
        def replace_latin_word(match):
            word = match.group()
            lower_word = word.lower()

            # Аббревиатура: все заглавные, только буквы, длина >=2
            if len(word) >= 2 and word.isalpha() and word.isupper():
                return ' '.join(LATIN_TO_RU.get(ch.lower(), ch) for ch in word)

            # Есть в словаре?
            if lower_word in trans_map:
                return trans_map[lower_word]

            # Иначе — транслит по буквам
            return ''.join(LATIN_TO_RU.get(ch.lower(), ch) for ch in word)

        text = re.sub(r'\b[A-Za-z]+\b', replace_latin_word, text)

        # Очистка лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _find_safe_sentence_end(self, text: str) -> Optional[int]:
        """
        Ищет позицию ПОСЛЕ последнего завершённого предложения.
        Завершённое = [.!?] за которым следует пробел/перенос/конец строки.
        Возвращает индекс для среза (например, 10 → text[:10]).
        """
        # Идём с конца
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '.!?':
                # Если это конец строки — считаем завершённым (для flush)
                if i == len(text) - 1:
                    return len(text)
                # Иначе проверяем следующий символ
                next_char = text[i + 1]
                if next_char in ' \n\t\r':
                    return i + 1  # включаем точку, но не пробел
        return None

    def _find_safe_cutoff(self, text: str, max_len: int) -> int:
        """Находит безопасную позицию для обрезки (не посередине слова)"""
        if len(text) <= max_len:
            return len(text)
        # Ищем последний пробел до max_len
        cutoff = max_len
        while cutoff > 0 and not text[cutoff - 1].isspace():
            cutoff -= 1
        if cutoff == 0:
            cutoff = max_len  # не нашли — режем по длине
        return cutoff

    def flush(self) -> List[str]:
        """Вызывается в конце потока — отправляем всё, что осталось"""
        if not self.clean_buffer:
            return []
        processed = self._transform(self.clean_buffer)
        # В flush разрешаем отправку даже без пробела после точки
        self.clean_buffer = ""
        result = processed.strip()
        return [result] if result else []

    def reset(self):
        """Сброс состояния (для нового диалога)"""
        self.clean_buffer = ""
        self.inside_tag = False