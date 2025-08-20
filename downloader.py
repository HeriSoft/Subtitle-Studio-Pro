import subprocess, os, re, shutil

def download_media(url, out_dir, quality='best', ffmpeg_path='./ffmpeg.exe', progress_cb=None, cancel_flag=None, use_bbdown=False, bbdown_path='BBDown.exe'):
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(out_dir, exist_ok=True)
    out_tpl = os.path.join(out_dir, '%(title).80s-%(id)s.%(ext)s')

    # Kiểm tra và trích xuất mã BV cho Bilibili
    is_bilibili = bool(re.match(r'.*(bilibili\.com/video/(BV|av)|b23\.tv/(BV|av))', url))
    if is_bilibili and use_bbdown:
        if not os.path.exists(bbdown_path):
            raise FileNotFoundError(f'BBDown.exe not found at {bbdown_path}')
        # Trích xuất mã BV (ví dụ: BV1kE88z2EDB)
        bv_match = re.search(r'(BV|av)[0-9a-zA-Z]+', url)
        if not bv_match:
            raise ValueError('Invalid Bilibili URL: BV or AV code not found')
        bv_code = bv_match.group(0)
        cmd = [bbdown_path, bv_code, '-tv']
        # Chạy BBDown.exe
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='gbk', errors='replace')
        output_lines = []
        try:
            for line in p.stdout:
                if cancel_flag and cancel_flag():
                    p.kill()
                    if progress_cb:
                        progress_cb(0)
                    return None
                if progress_cb:
                    progress_cb(line.strip())
                output_lines.append(line.strip())
            p.wait()
            if p.returncode != 0:
                stdout, stderr = p.communicate()
                raise RuntimeError(f'BBDown.exe failed: {stderr or "Unknown error"}')
        finally:
            pass
        # Tìm tên file gốc từ output hoặc file .mp4 mới nhất
        src_file = None
        for line in output_lines:
            if line.endswith('.mp4') and not line.startswith('[2025'):
                possible_file = line.split()[-1]
                if os.path.exists(possible_file):
                    src_file = possible_file
                    break
        if not src_file:
            current_dir = os.getcwd()
            files = sorted([os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith('.mp4')],
                           key=lambda p: os.path.getmtime(p), reverse=True)
            src_file = files[0] if files else None
        if not src_file or not os.path.exists(src_file):
            raise FileNotFoundError('No video file found after BBDown download')
        src_filename = os.path.basename(src_file)
        # Tạo tên file đích trong out_dir
        dest_file = os.path.join(out_dir, src_filename)
        # Di chuyển file
        if progress_cb:
            progress_cb(f'Đã tải: {src_filename}')
            progress_cb(f'Đang di chuyển vào thư mục chỉ định {out_dir}')
        try:
            shutil.move(src_file, dest_file)
            if progress_cb:
                progress_cb('Di chuyển thành công!')
        except Exception as e:
            raise RuntimeError(f'Failed to move file to {out_dir}: {e}')
        return dest_file

    # Logic yt-dlp cho các URL khác
    fmt = 'bestvideo+bestaudio/best' if quality == 'best' else quality
    cmd = ['yt-dlp', '-o', out_tpl, '-f', fmt, '--merge-output-format', 'mp4', '--newline', '--ffmpeg-location', os.path.abspath(os.path.dirname(ffmpeg_path)), url]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace')
    try:
        for line in p.stdout:
            if cancel_flag and cancel_flag():
                p.kill()
                if progress_cb:
                    progress_cb(0)
                return None
            m = re.search(r'(\d{1,3}\.\d)%', line)
            if m and progress_cb:
                try:
                    progress_cb(float(m.group(1)))
                except:
                    pass
            elif progress_cb:
                progress_cb(line.strip())
        p.wait()
        if p.returncode != 0:
            stdout, stderr = p.communicate()
            raise RuntimeError(f'yt-dlp failed: {stderr or "Unknown error"}')
    finally:
        pass
    files = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir)], key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0] if files else None