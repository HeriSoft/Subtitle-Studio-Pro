import requests
import time

DEEPSEEK_BASE = 'https://api.deepseek.com/v1'

class DeepSeekTranslator:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError('DeepSeek API key không được cung cấp')
        # Kiểm tra API key hợp lệ
        try:
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            r = requests.get(f"{DEEPSEEK_BASE}/models", headers=headers, timeout=10)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 401:
                raise ValueError('DeepSeek API key không hợp lệ')
            raise Exception(f'Lỗi kiểm tra API DeepSeek: {str(e)}')

    def translate(self, text: str, target_lang: str, chinese_archaic: bool = False):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Cải thiện system prompt cho từng ngôn ngữ đích
        system = (
            "You are a professional subtitle translator specialized in Asian languages. "
            "Follow these rules strictly:\n"
            "1. Keep translations concise and natural for subtitles\n"
            "2. Preserve proper nouns and technical terms\n"
            "3. Maintain original line breaks when possible\n"
            "4. Never combine multiple lines into one"
        )
        
        # Thêm hướng dẫn đặc biệt cho dịch Trung-Việt
        if target_lang.lower() in ['zh', 'chinese'] and chinese_archaic:
            system += (
                "\n\nTranslate into Classical-flavored Chinese for wuxia/xianxia, "
                "using archaic pronouns (兄, 妹, 余/吾, 汝, 娘子, 家父, 家母), elegant tone."
            )
        elif target_lang.lower() == 'vi':
            system += (
                "\n\nWhen translating to Vietnamese:\n"
                "- Use natural Vietnamese colloquial speech\n"
                "- Keep sentence structures simple\n"
                "- Preserve Chinese names in their original form\n"
                "- For wuxia terms, use common Vietnamese translations (e.g. 'võ lâm' for 武林)"
            )
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': (
                    f"Translate this exactly line by line to {target_lang}. "
                    f"Never combine lines. Keep line breaks:\n\n{text}"
                )}
            ],
            'temperature': 0.15,
            'max_tokens': 2000  # Giới hạn token để tránh response quá dài
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{DEEPSEEK_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                translated_text = response.json()['choices'][0]['message']['content'].strip()
                
                # Xử lý hậu kỳ để đảm bảo giữ nguyên số dòng
                translated_lines = translated_text.split('\n')
                original_lines = text.split('\n')
                
                # Nếu số dòng khớp hoặc ít hơn bản gốc
                if len(translated_lines) <= len(original_lines):
                    return '\n'.join(translated_lines)
                
                # Nếu nhiều dòng hơn, cắt bớt cho khớp
                return '\n'.join(translated_lines[:len(original_lines)])
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    raise ValueError('DeepSeek API key không hợp lệ')
                raise Exception(f'API DeepSeek lỗi: {str(e)}')
            except requests.exceptions.RequestException as e:
                raise Exception(f'Lỗi kết nối DeepSeek: {str(e)}')
        
        raise Exception('Failed after 3 retries')