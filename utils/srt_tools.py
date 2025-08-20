import srt, datetime, re
def segments_to_srt(segments, out_path):
    subs = []
    for i, seg in enumerate(segments, start=1):
        start = datetime.timedelta(seconds=float(seg.get('start', 0)))
        end = datetime.timedelta(seconds=float(seg.get('end', start.total_seconds()+1.0)))
        text = seg.get('text', '').strip().replace('\r','').replace('\n',' ').strip()
        subs.append(srt.Subtitle(index=i, start=start, end=end, content=text))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))
def smart_segment_text_to_srt(text, out_path):
    parts = [p.strip() for p in re.split('([。！？!?\.\n])', text) if p.strip()]
    sentences=[]; buf=''
    for p in parts:
        if re.match('[。！？!?\.\n]', p):
            buf = (buf + p).strip(); sentences.append(buf); buf=''
        else:
            buf += (' ' + p) if buf else p
    if buf.strip(): sentences.append(buf.strip())
    subs=[]; import datetime
    t = datetime.timedelta(0); idx=1
    for s in sentences:
        sec = max(1.2, min(6.0, len(s.split())/3.0))
        start=t; end=t+datetime.timedelta(seconds=sec)
        subs.append(srt.Subtitle(index=idx, start=start, end=end, content=s))
        t = end + datetime.timedelta(milliseconds=120); idx+=1
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))
