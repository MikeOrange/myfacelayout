from string import ascii_lowercase

LETTERS = set(ascii_lowercase)


def is_pangram(s):
    return set(s.lower()) >= LETTERS

