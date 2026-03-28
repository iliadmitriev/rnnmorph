# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Функции для обработки тегов.


def process_gram_tag(gram: str):
    """
    Выкинуть лишние грамматические категории и отсортировать их в составе значения.
    
    This function normalizes grammeme tags to match the original RNNMorph training setup.
    Categories dropped:
    - Animacy, Aspect, NumType (original drops)
    - PronType, Poss, Reflex (not in original model output vectorizer)
    - Abbr, ExtPos, Foreign, InflClass, NameType, Polarity, Typo (rare/UD-specific)
    """
    gram = gram.strip().split("|")
    
    # Original dropped categories
    dropped = ["Animacy", "Aspect", "NumType"]
    
    # Additional categories not in original RNNMorph model
    # These appear in UD/OpenCorpora but were not used in training
    additional_dropped = [
        "PronType", "Poss", "Reflex",  # Pronominal features
        "Abbr", "ExtPos", "Foreign",    # Special tags
        "InflClass", "NameType",        # Morphological features
        "Polarity", "Typo"              # Error/variation tags
    ]
    
    all_dropped = dropped + additional_dropped
    
    gram = [grammem for grammem in gram if sum([drop in grammem for drop in all_dropped]) == 0]
    return "|".join(sorted(gram)) if gram else "_"


def convert_from_opencorpora_tag(to_ud, tag: str, text: str):
    """
    Конвертировать теги их формата OpenCorpora в Universal Dependencies

    :param to_ud: конвертер.
    :param tag: тег в OpenCorpora.
    :param text: токен.
    :return: тег в UD.
    """
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1]
    return pos, gram
