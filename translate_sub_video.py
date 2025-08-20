# translate_sub_video.py
import asyncio
import os
from datetime import timedelta
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

# Đảm bảo file openai_provider.py nằm trong cùng thư mục hoặc trong một gói được định nghĩa
from providers.openai_provider import OpenAIProvider

# Định nghĩa đường dẫn và chi tiết mô hình Whisper
MODEL_PATH = "./models/BELLE-2/Belle-whisper-large-v3-zh"
REPO_ID = "BELLE-2/Belle-whisper-large-v3-zh"

def format_timestamp(milliseconds: float) -> str:
    """Định dạng thời gian từ mili giây sang định dạng SRT HH:MM:SS,mmm"""
    td = timedelta(milliseconds=milliseconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

async def translate_video_subtitles(
    video_path: str,
    output_srt_path: str,
    whisper_model_path: str = MODEL_PATH,
    whisper_repo_id: str = REPO_ID,
    source_lang: str = "zh", # Ngôn ngữ nguồn là tiếng Trung
    target_lang: str = "vi"  # Ngôn ngữ đích là tiếng Việt
):
    """
    Trích xuất âm thanh từ video, chuyển đổi giọng nói thành văn bản bằng Whisper,
    và dịch phụ đề sang ngôn ngữ đích bằng OpenAI, sau đó lưu vào file .srt.
    """
    print(f"Bắt đầu dịch phụ đề cho video: {video_path}")

    # 1. Trích xuất âm thanh từ video
    audio_path = "temp_audio_for_transcription.wav"
    try:
        print("Đang trích xuất âm thanh...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None) # Tắt nhật ký của moviepy
        print(f"Âm thanh đã được trích xuất tới {audio_path}")
    except Exception as e:
        print(f"Lỗi khi trích xuất âm thanh: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise

    # 2. Chuyển đổi giọng nói thành văn bản bằng mô hình Belle-Whisper
    print("Đang tải mô hình Whisper...")
    # Sử dụng GPU nếu có, nếu không thì dùng CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Sử dụng kiểu dữ liệu float16 trên GPU để tiết kiệm bộ nhớ và tăng tốc
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Kiểm tra xem mô hình có tồn tại cục bộ không, nếu không thì tải từ Hugging Face Hub
    if os.path.exists(whisper_model_path):
        model_name_or_path = whisper_model_path
        print(f"Đang tải mô hình cục bộ từ: {whisper_model_path}")
    else:
        model_name_or_path = whisper_repo_id
        print(f"Đang tải mô hình từ Hugging Face Hub: {whisper_repo_id}")

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    # Khởi tạo pipeline nhận dạng giọng nói tự động
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30, # Chia nhỏ âm thanh thành các đoạn 30 giây để xử lý
        batch_size=8,
        return_timestamps=True, # Trả về cả dấu thời gian
        torch_dtype=torch_dtype,
        device=device,
    )

    print("Đang chuyển đổi âm thanh thành văn bản...")
    try:
        # Chuyển đổi và đảm bảo ngôn ngữ là tiếng Trung
        transcription_result = pipe(audio_path, generate_kwargs={"language": "chinese"})
        segments = transcription_result.get("chunks", [])
        print(f"Chuyển đổi hoàn tất. Tìm thấy {len(segments)} đoạn.")
    except Exception as e:
        print(f"Lỗi trong quá trình chuyển đổi: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise

    # 3. Dịch các đoạn văn bản đã chuyển đổi bằng OpenAI
    openai_provider = OpenAIProvider()
    translated_subtitles = []
    print("Đang dịch phụ đề bằng OpenAI...")
    for i, segment in enumerate(segments):
        original_text = segment['text'].strip()
        # Chuyển đổi thời gian từ giây sang mili giây
        start_time_ms = int(segment['timestamp'][0] * 1000) if segment['timestamp'][0] is not None else 0
        end_time_ms = int(segment['timestamp'][1] * 1000) if segment['timestamp'][1] is not None else start_time_ms + 1000 # Thời lượng mặc định 1 giây nếu không có

        if not original_text:
            continue # Bỏ qua các đoạn trống

        print(f"  Đang dịch đoạn {i+1}/{len(segments)}: '{original_text}'")
        try:
            translated_text = await openai_provider.translate(
                original_text,
                source_lang=source_lang,
                target_lang=target_lang,
                chinese_archaic=False # Đặt True nếu nội dung là kiếm hiệp/tiên hiệp
            )
            translated_subtitles.append({
                "index": i + 1,
                "start": format_timestamp(start_time_ms),
                "end": format_timestamp(end_time_ms),
                "text": translated_text.strip()
            })
        except Exception as e:
            print(f"    Lỗi khi dịch đoạn: '{original_text}' - {e}")
            # Thêm văn bản gốc hoặc một placeholder nếu dịch lỗi
            translated_subtitles.append({
                "index": i + 1,
                "start": format_timestamp(start_time_ms),
                "end": format_timestamp(end_time_ms),
                "text": f"[Lỗi dịch] {original_text}"
            })

    # 4. Lưu vào file .srt
    print(f"Đang lưu phụ đề đã dịch vào {output_srt_path}")
    try:
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            for sub in translated_subtitles:
                f.write(f"{sub['index']}\n")
                f.write(f"{sub['start']} --> {sub['end']}\n")
                f.write(f"{sub['text']}\n\n")
        print("File SRT đã được lưu thành công.")
    except Exception as e:
        print(f"Lỗi khi lưu file SRT: {e}")
        raise

    # Dọn dẹp file âm thanh tạm thời
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Đã dọn dẹp file âm thanh tạm thời: {audio_path}")

async def main():
    # --- CẤU HÌNH SỬ DỤNG ---
    # Thay đổi đường dẫn này thành file video của bạn
    video_file = "E:/SubtitleStudio/SubtitleStudio_Pro/downloads/c.mp4"
    # Tên file SRT đầu ra mong muốn
    output_srt_file = "output_translated_subtitles.srt"

    # Kiểm tra xem file video có tồn tại không
    if not os.path.exists(video_file):
        print(f"Lỗi: Không tìm thấy file video tại '{video_file}'")
        print("Vui lòng thay thế 'path/to/your/video.mp4' bằng đường dẫn thực tới file video của bạn.")
        return

    try:
        await translate_video_subtitles(video_file, output_srt_file)
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    print("Bắt đầu script dịch phụ đề video.")
    print("Đảm bảo bạn đã có file `settings.py` với `openai_api_key`.")
    print("Đảm bảo đã cài đặt tất cả các thư viện cần thiết: `pip install moviepy pydub transformers accelerate openai torch`")
    print("Ngoài ra, `ffmpeg` phải được cài đặt trên hệ thống của bạn và có thể truy cập được (thêm vào PATH).")

    # Tạo thư mục models nếu chưa tồn tại (để lưu mô hình Whisper nếu tải cục bộ)
    os.makedirs("./models/BELLE-2/", exist_ok=True)

    asyncio.run(main())