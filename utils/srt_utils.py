import srt
from typing import List
def read_srt(path: str):
    with open(path,'r', encoding='utf-8') as f:
        return list(srt.parse(f.read()))
def srt_lines(subs):
    lines = []
    for sub in subs:
        text = sub.content.replace('\n', ', ').strip()
        if text: lines.append(text)
    return lines
