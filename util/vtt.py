from typing import NamedTuple


def format_time(seconds):
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'


def parse_time(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s


class Segment(NamedTuple):
    start: float
    end: float
    text: str


def load_vtt(s):
    if isinstance(s, str):
        lines = s.split('\n')
    else:
        lines = s

    data = []
    i = 1
    while i < len(lines):
        line = lines[i]
        if '-->' in line:
            start, end = line.split('-->')
            start = parse_time(start.strip())
            end = parse_time(end.strip())
            text = lines[i + 1]
            i += 2
            data.append(Segment(start, end, text.strip()))
        else:
            i += 1
    return data


def write_vtt(filename, lines):
    with open(filename, 'w') as fp:
        fp.write('WEBVTT\n\n')

        for i, l in enumerate(lines):
            fp.write(f'{format_time(l.start)} --> {format_time(l.end)}\n{l.text}')
            if i < len(lines) - 1:
                fp.write('\n\n')
            else:
                fp.write('\n')