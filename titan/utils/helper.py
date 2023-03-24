"""Useful Helper Functions"""
import itertools


def chunks(iterable, size):
    """Iterate over in chunks"""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, size))
