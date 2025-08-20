import aiohttp
import asyncio

BASE_TTS = 'https://api.ausynclab.io/api/v1/speech'
BASE_VOICES = 'https://api.ausynclab.io/api/v1/voices'

class AusyncLabClient:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.session = None

    def _headers(self):
        return {'X-API-Key': self.api_key, 'accept': 'application/json'}

    async def __aenter__(self):
        """Khởi tạo session khi sử dụng async with."""
        self.session = aiohttp.ClientSession(headers=self._headers())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Đóng session khi thoát khỏi async with."""
        if self.session:
            await self.session.close()
            self.session = None

    async def list_voices(self):
        url = f"{BASE_VOICES}/list"
        async with aiohttp.ClientSession(headers=self._headers()) as sess:
            async with sess.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get('result', [])

    async def tts_create(self, *, text: str, voice_id: int, language: str = 'vi', speed: float = 1.0, model_name: str = 'myna-1', audio_name: str = 'Subtitle Line', callback_url: str = None):
        url = f"{BASE_TTS}/text-to-speech"
        payload = {
            'audio_name': audio_name,
            'text': text,
            'voice_id': voice_id,
            'speed': speed,
            'model_name': model_name,
            'language': language
        }
        if callback_url:
            payload['callback_url'] = callback_url
        async with aiohttp.ClientSession(headers={**self._headers(), 'Content-Type': 'application/json'}) as sess:
            async with sess.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return int(data['result']['audio_id'])

    async def tts_get(self, audio_id: int):
        url = f"{BASE_TTS}/{audio_id}"
        async with aiohttp.ClientSession(headers=self._headers()) as sess:
            async with sess.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def tts_poll_until_ready(self, audio_id: int, interval: float = 0.8, timeout: float = 300.0, progress_cb=None, cancel_flag=None):
        deadline = asyncio.get_event_loop().time() + timeout
        elapsed = 0
        while True:
            if cancel_flag and cancel_flag():
                raise asyncio.CancelledError(f'TTS job {audio_id} cancelled')
            data = await self.tts_get(audio_id)
            result = data.get('result') or {}
            state = (result.get('state') or '').upper()
            if state in ('SUCCEED', 'SUCCEEDED') and result.get('audio_url'):
                if progress_cb:
                    progress_cb(100)
                return result
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f'TTS job {audio_id} timed out with state={state}')
            elapsed += interval
            if progress_cb:
                progress_cb(min(100, (elapsed / timeout * 100)))
            await asyncio.sleep(interval)