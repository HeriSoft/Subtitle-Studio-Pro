import subprocess, json, os, re
def duration_of(path, ffprobe='./ffprobe.exe'):
    try:
        cmd = [ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            j = json.loads(proc.stdout)
            return float(j['format']['duration'])
    except Exception:
        pass
    # fallback parse ffmpeg -i
    try:
        cmd = [ffprobe.replace('ffprobe.exe','ffmpeg.exe'), '-i', path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stderr
        m = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', out)
        if m:
            h,mi,s = m.groups(); return int(h)*3600 + int(mi)*60 + float(s)
    except Exception:
        pass
    return None
