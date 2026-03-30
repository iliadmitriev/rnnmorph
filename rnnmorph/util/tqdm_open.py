# -*- coding: utf-8 -*-
# Авторы: Анастасьев Даниил
# Описание: Обертка открытия больших файлов в счетчик tqdm.

from contextlib import contextmanager
from os.path import getsize, basename
from tqdm import tqdm


@contextmanager
def tqdm_open(filename, encoding='utf8'):
    """
    Открытие файла, обёрнутое в tqdm
    """
    total = getsize(filename)

    def wrapped_line_iterator(fd):
        with tqdm(total=total, unit="B", unit_scale=True, desc=basename(filename), miniters=1) as pb:
            processed_bytes = 0
            try:
                for line in fd:
                    processed_bytes += len(line.encode('utf-8'))
                    if processed_bytes >= 1024 * 1024:
                        pb.update(processed_bytes)
                        processed_bytes = 0
                    yield line
            finally:
                # Ensure progress bar reaches 100%
                pb.update(processed_bytes)
                pb.refresh()  # Force display update
                pb.close()  # Properly close the bar

    with open(filename, encoding=encoding) as fd:
        yield wrapped_line_iterator(fd)
