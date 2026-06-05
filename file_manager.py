import csv
from contextlib import contextmanager


@contextmanager
def csv_writer(file_path, mode):
    file = open(file_path, mode, newline="")
    writer = csv.writer(file)
    yield writer
    file.close()
