import subprocess, re, time
def run_ffmpeg_with_progress(args, total_seconds=None, progress_cb=None, cancel_flag=None):
    # ensure -progress pipe:1 present
    if '-progress' not in args:
        args = args + ['-progress','pipe:1','-nostats']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        while True:
            if cancel_flag and cancel_flag():
                p.kill(); return False
            line = p.stdout.readline()
            if not line:
                if p.poll() is not None:
                    break
                time.sleep(0.1); continue
            m = re.search(r'out_time_ms=(\d+)', line)
            if m and total_seconds and progress_cb:
                out_ms = int(m.group(1)); pct = min(100.0, out_ms/1000.0/total_seconds*100.0); progress_cb(pct)
        p.wait()
        return p.returncode == 0
    finally:
        try: p.kill()
        except: pass
