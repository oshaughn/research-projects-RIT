"""Condor submit argument quoting used by runtime subdag/frame builders."""
import shlex


def parse_submit_arguments(value):
    value = value.strip()
    if not value.startswith('"'):
        return shlex.split(value)
    if not value.endswith('"'):
        raise ValueError('Unterminated Condor arguments string')
    text = value[1:-1]
    words, word = [], []
    quoted, active = False, False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            if index+1 >= len(text) or text[index+1] != '"':
                raise ValueError('Literal double quotes must be doubled in Condor arguments')
            word.append('"'); active = True; index += 2; continue
        if char == "'":
            if quoted and index+1 < len(text) and text[index+1] == "'":
                word.append("'"); index += 2; continue
            quoted = not quoted; active = True
        elif char.isspace() and not quoted:
            if active:
                words.append(''.join(word)); word = []; active = False
        else:
            word.append(char); active = True
        index += 1
    if quoted:
        raise ValueError('Unterminated single-quoted Condor argument')
    if active:
        words.append(''.join(word))
    return words


def format_submit_arguments(argv):
    words = []
    for value in argv:
        value = value.replace('"', '""')
        if not value or any(char.isspace() for char in value) or "'" in value:
            value = "'" + value.replace("'", "''") + "'"
        words.append(value)
    return '"' + ' '.join(words) + '"'
