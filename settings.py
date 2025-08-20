import json
import os
from cryptography.fernet import Fernet
from pathlib import Path

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")
DEFAULTS = {
    "openai_api_key": "",
    "deepseek_api_key": "",
    "ausynclab_api_key": "",
    "max_tokens": 2048,
    "ffmpeg_path": "ffmpeg",
    "ytdlp_path": "yt-dlp",
    "bbdown_path": "BBDown.exe",
    "hf_auto_download": False,
    "font_family": "Arial",
    "burn_font_family": "Arial",
    "burn_font_size": 24,
    "burn_position": "bottom",
    "burn_color": "&H00FFFFFF",
    "burn_outline": 1,
    "burn_shadow": 1,
    "burn_margin_v": 10,
    "out_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), 'downloads')),  # Cập nhật
    "theme": "Light",
    "language": "Vietnamese",
    "convert_type": "Video",
    "convert_format": "mp4",
    "use_wuxia_style": False,
    'user_name': '',
    'user_age': 0,
    'user_gender': 'Không tiết lộ',
    'user_job': '',
    'user_education': 'Đại học',
    'user_skills': '',
    'user_interests': '',
    'user_dislikes': '',
    'communication_style': 'Trực tiếp',
    'auto_share_personal': True
}

# Tạo key mã hóa (lưu ý: trong thực tế nên dùng key management system)
def _get_encryption_key():
    key_path = Path(__file__).parent / ".encryption_key"
    if not key_path.exists():
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
    else:
        with open(key_path, "rb") as f:
            key = f.read()
    return key

def _encrypt_data(data: str) -> bytes:
    """Mã hóa chuỗi nhạy cảm"""
    fernet = Fernet(_get_encryption_key())
    return fernet.encrypt(data.encode())

def _decrypt_data(encrypted_data: bytes) -> str:
    """Giải mã dữ liệu"""
    fernet = Fernet(_get_encryption_key())
    return fernet.decrypt(encrypted_data).decode()

def save_settings(data: dict):
    """Lưu cấu hình với mã hóa thông tin nhạy cảm"""
    sensitive_fields = [
        'user_name', 'user_job', 
        'user_skills', 'user_interests',
        'user_dislikes'
    ]
    
    encrypted_data = {}
    for k, v in data.items():
        if k in sensitive_fields and v:
            encrypted_data[k] = _encrypt_data(v).decode()
        else:
            encrypted_data[k] = v
    
    with open(SETTINGS_PATH, "w", encoding='utf-8') as f:
        json.dump(encrypted_data, f, indent=2)

def load_settings():
    """Đọc cấu hình với giải mã thông tin"""
    try:
        with open(SETTINGS_PATH, "r", encoding='utf-8') as f:
            encrypted_data = json.load(f)
    except:
        return DEFAULTS
    
    decrypted_data = {}
    for k, v in encrypted_data.items():
        if k in DEFAULTS and isinstance(DEFAULTS[k], str) and isinstance(v, str):
            try:
                decrypted_data[k] = _decrypt_data(v.encode())
            except:
                decrypted_data[k] = v
        else:
            decrypted_data[k] = v
    
    return {**DEFAULTS, **decrypted_data}