import os
import time
from huggingface_hub import snapshot_download

def ensure_hf_model(model_name: str, log_cb=None):
    model_dir = os.path.join('models', model_name)
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        if log_cb:
            log_cb(f'Đang tải mô hình {model_name}...')
        for attempt in range(3):
            try:
                snapshot_download(
                    repo_id=f'openai/whisper-{model_name}',
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                break
            except Exception as e:
                if attempt < 2:
                    if log_cb:
                        log_cb(f'Thử lại tải mô hình {model_name} ({attempt + 1}/3)...')
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f'Tải mô hình thất bại: {e}')
        if log_cb:
            log_cb(f'Đã tải mô hình {model_name} vào {model_dir}')
    return model_dir