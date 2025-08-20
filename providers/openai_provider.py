# providers/openai_provider.py
import asyncio
from openai import AsyncOpenAI
from settings import load_settings

class OpenAIProvider:
    def __init__(self):
        self.cfg = load_settings()
        self.api_key = self.cfg.get('openai_api_key', '')
        if not self.api_key:
            raise ValueError("OpenAI API key is missing")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def text_to_speech(self, text: str, voice: str, speed: float = 1.0, model: str = 'tts-1', output_path: str = None):
        try:
            if not text.strip():
                raise ValueError("Text cannot be empty")
            if len(text) > 4096:
                raise ValueError("Text exceeds OpenAI's 4096 character limit")
            if voice not in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']:
                raise ValueError(f"Invalid voice: {voice}")
            if model not in ['tts-1', 'tts-1-hd']:
                raise ValueError(f"Invalid model: {model}")
            if speed < 0.25 or speed > 4.0:
                raise ValueError(f"Speed must be between 0.25 and 4.0, got {speed}")

            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                speed=speed,
                response_format='mp3'
            )

            if not output_path:
                output_path = 'output.mp3'

            # Ghi trực tiếp dữ liệu bytes vào file
            with open(output_path, 'wb') as f:
                f.write(response.content)  # response.content là bytes

            return output_path
        except Exception as e:
            raise RuntimeError(f"OpenAI TTS failed: {str(e)}") from e

    async def translate(self, text, source_lang, target_lang, chinese_archaic=False, retries=3):
        for attempt in range(retries):
            try:
                if not text.strip():
                    return text

                system = "You are a professional subtitle translator. Rules:\n"
                system += "- Translate each line exactly, preserving line breaks\n"
                system += "- Keep translations concise (under 42 characters per line)\n"
                system += "- Preserve proper nouns and technical terms\n"
                system += "- Never combine multiple lines into one\n"
                
                if source_lang == 'zh' and target_lang == 'vi':
                    system += "- For Chinese-Vietnamese: Keep names/terms unchanged\n"
                
                if chinese_archaic:
                    system += "- Use Classical Chinese style for wuxia/xianxia in Vietnamese\n"

                # Split text into lines
                lines = text.split('\n')
                translated_lines = []

                # Translate each line individually
                for line in lines:
                    if not line.strip():
                        translated_lines.append(line)
                        continue
                    max_chunk_size = 500
                    if len(line) > max_chunk_size:
                        chunks = [line[i:i+max_chunk_size] for i in range(0, len(line), max_chunk_size)]
                        translated_chunks = []
                        for chunk in chunks:
                            translated = await self._call_openai_api(chunk, system, source_lang, target_lang, attempt)
                            translated_chunks.append(translated)
                        translated_lines.append("\n".join(translated_chunks))
                    else:
                        translated = await self._call_openai_api(line, system, source_lang, target_lang, attempt)
                        translated_lines.append(translated)

                # Ensure the number of lines matches the input
                return '\n'.join(translated_lines[:len(lines)])

            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                raise RuntimeError(f"OpenAI translation failed after {retries} retries: {str(e)}") from e

    async def _call_openai_api(self, text, system_prompt, source_lang, target_lang, attempt):
        timeout = 30 * (attempt + 1)
        response = await self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate this to {target_lang}, keeping the exact text structure and line breaks:\n{text}"}
            ],
            temperature=0.15,
            max_tokens=1000,
            timeout=timeout
        )
        return response.choices[0].message.content.strip()