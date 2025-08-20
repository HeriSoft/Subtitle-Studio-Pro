import subprocess
import os
import tempfile
import json
from typing import List
import shlex
from pathlib import Path
from ffmpeg_runner import run_ffmpeg_with_progress

def concat_audios_ffmpeg(ffmpeg_path: str, input_paths: List[str], out_path: str, sample_rate: int = 24000, progress_cb=None, cancel_flag=None, logger=None, copy_codec: bool=False):
    if not input_paths:
        raise ValueError("Danh sách file âm thanh đầu vào rỗng")
    if not os.path.isfile(ffmpeg_path) or 'ffmpeg' not in os.path.basename(ffmpeg_path).lower():
        raise FileNotFoundError(f"Đường dẫn ffmpeg không hợp lệ: {ffmpeg_path}")
    for p in input_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File âm thanh không tồn tại: {p}")

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as f:
        for p in input_paths:
            # Escape đường dẫn cho ffmpeg
            safe_path = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")
        list_path = f.name
    try:
        args = [ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-ar', str(sample_rate)]
        if copy_codec:
            args.extend(['-c:a', 'copy'])
        else:
            args.extend(['-c:a', 'aac'])
        args.append(out_path)

        total_seconds = sum(get_media_duration(ffmpeg_path, p, logger=logger) or 0 for p in input_paths)
        if total_seconds == 0:
            raise RuntimeError("Không thể lấy thời lượng của các file âm thanh")
        success = run_ffmpeg_with_progress(args, total_seconds=total_seconds, progress_cb=progress_cb, cancel_flag=cancel_flag)
        if not success:
            raise RuntimeError("FFmpeg concat failed")
    finally:
        try:
            os.remove(list_path)
        except Exception as e:
            if logger:
                logger.emit(f"Lỗi khi xóa file tạm {list_path}: {str(e)}")

def mux_voiceover(ffmpeg_path, video_in, voice_audio, out_path, srt_path=None, mute_original=False, progress_cb=None, cancel_flag=None, logger=None):
    for input_file in [video_in, voice_audio] + ([srt_path] if srt_path else []):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File không tồn tại: {input_file}")
    if not os.path.isfile(ffmpeg_path) or 'ffmpeg' not in os.path.basename(ffmpeg_path).lower():
        raise FileNotFoundError(f"Đường dẫn ffmpeg không hợp lệ: {ffmpeg_path}")

    cmd = [ffmpeg_path, "-i", video_in, "-i", voice_audio]
    if srt_path:
        # Escape đường dẫn phụ đề để tránh lỗi Unicode / dấu cách
        safe_srt = Path(srt_path).as_posix()
        cmd.extend(["-vf", f"subtitles='{safe_srt}'"])
    if mute_original:
        cmd.extend(["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac"])
    else:
        cmd.extend(["-map", "0:v"])
        # Thêm audio gốc nếu có
        try:
            if get_media_duration(ffmpeg_path, video_in, logger=logger):
                cmd.extend(["-map", "0:a"])
        except:
            pass
        cmd.extend(["-map", "1:a", "-c:v", "copy", "-c:a", "aac"])
    # Không dùng -shortest để tránh cắt mất phụ đề
    cmd.extend(["-y", out_path])

    total_seconds = get_media_duration(ffmpeg_path, video_in, logger=logger) or 100
    success = run_ffmpeg_with_progress(cmd, total_seconds=total_seconds, progress_cb=progress_cb, cancel_flag=cancel_flag)
    if not success:
        raise RuntimeError("FFmpeg mux failed")

def get_media_duration(ffmpeg_path: str, input_file: str, logger=None) -> float:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File không tồn tại: {input_file}")
    ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
    if not os.path.isfile(ffprobe_path):
        raise FileNotFoundError(f"ffprobe không tồn tại tại: {ffprobe_path}")
    try:
        cmd = [ffprobe_path, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', input_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        j = json.loads(result.stdout)
        return float(j['format']['duration'])
    except Exception as e:
        if logger:
            logger.emit(f"Lỗi lấy thời lượng: {str(e)}")
        return None
