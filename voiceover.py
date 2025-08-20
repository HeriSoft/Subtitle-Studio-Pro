import asyncio
import tempfile
import os
from providers.ausynclab import AusyncLabClient
from providers.openai_provider import OpenAIProvider
from utils.ffmpeg_tools import concat_audios_ffmpeg, mux_voiceover
from utils.srt_utils import read_srt, srt_lines
import aiohttp

async def download_file(url: str, path: str):
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            resp.raise_for_status()
            with open(path, 'wb') as f:
                while True:
                    chunk = await resp.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)

async def tts_for_text(api_key: str, text: str, *, voice_id: str, output_path: str, language: str = 'vi', model_name: str = 'myna-1', progress_cb=None, cancel_flag=None, provider: str = 'AusyncLab', openai_model: str = 'tts-1'):
    try:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if len(text) > 4096 and provider == 'OpenAI':
            raise ValueError("Text exceeds OpenAI's 4096 character limit")
        if not os.path.isdir(os.path.dirname(output_path)):
            raise ValueError(f"Output directory {os.path.dirname(output_path)} does not exist")

        if provider == 'OpenAI':
            provider_instance = OpenAIProvider()
            audio_path = await provider_instance.text_to_speech(
                text=text,
                voice=voice_id,
                speed=1.0,
                model=openai_model,
                output_path=output_path
            )
            if progress_cb:
                progress_cb(100)
            return audio_path
        else:
            if not isinstance(voice_id, int) or voice_id < 0:
                raise ValueError("Invalid voice_id")
            client = AusyncLabClient(api_key)
            async with client:
                audio_id = await client.tts_create(
                    text=text,
                    voice_id=voice_id,
                    language=language,
                    speed=1.0,
                    model_name=model_name,  # Use model_name from input
                    audio_name=os.path.basename(output_path)
                )
                result = await client.tts_poll_until_ready(
                    audio_id,
                    interval=0.8,
                    timeout=300.0,
                    progress_cb=progress_cb,
                    cancel_flag=cancel_flag
                )
                if not result.get('audio_url'):
                    raise RuntimeError("No audio URL returned")
                await download_file(result['audio_url'], output_path)
                return output_path
    except Exception as e:
        raise RuntimeError(f"TTS processing failed: {str(e)}") from e

async def tts_for_srt(api_key: str, srt_path: str, *, voice_id: str, workdir: str, ffmpeg_path: str, progress_cb=None, cancel_flag=None, language: str = 'vi', model_name: str = 'myna-1', provider: str = 'AusyncLab', openai_model: str = 'tts-1'):
    try:
        subs = read_srt(srt_path)
        texts = srt_lines(subs)
        if not texts:
            raise ValueError("SRT file is empty or invalid")
        
        os.makedirs(workdir, exist_ok=True)
        
        if provider == 'OpenAI':
            provider_instance = OpenAIProvider()
            sem = asyncio.Semaphore(4)
            
            async def process_line(idx, text):
                try:
                    async with sem:
                        if cancel_flag and cancel_flag():
                            return None
                        if not text.strip():
                            return None
                        out_path = os.path.join(workdir, f'line_{idx:05d}.mp3')
                        await provider_instance.text_to_speech(
                            text=text,
                            voice=voice_id,
                            speed=1.0,
                            model=openai_model,
                            output_path=out_path
                        )
                        if progress_cb:
                            progress_cb(int((idx / len(texts)) * 90))
                        return out_path
                except Exception as e:
                    raise RuntimeError(f"Failed to process line {idx}: {str(e)}")
        else:
            voice_id = int(voice_id.split(":")[-1] if ":" in str(voice_id) else voice_id)
            if voice_id < 0:
                raise ValueError("Invalid voice_id")
            client = AusyncLabClient(api_key)
            sem = asyncio.Semaphore(6)
            
            async def process_line(idx, text):
                try:
                    async with sem:
                        if cancel_flag and cancel_flag():
                            return None
                        if not text.strip():
                            return None
                        async with client:
                            audio_id = await client.tts_create(
                                text=text,
                                voice_id=voice_id,
                                language=language,
                                speed=1.0,
                                model_name=model_name,  # Use model_name from input
                                audio_name=f'line-{idx}'
                            )
                            result = await client.tts_poll_until_ready(
                                audio_id,
                                interval=0.8,
                                timeout=300.0,
                                progress_cb=lambda x: progress_cb(int((idx / len(texts)) * 90)) if progress_cb else None,
                                cancel_flag=cancel_flag
                            )
                        out_path = os.path.join(workdir, f'line_{idx:05d}.m4a')
                        await download_file(result['audio_url'], out_path)
                        return out_path
                except Exception as e:
                    raise RuntimeError(f"Failed to process line {idx}: {str(e)}")
        
        tasks = [process_line(i, t) for i, t in enumerate(texts, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        audio_files = []
        for idx, result in enumerate(results, 1):
            if isinstance(result, Exception):
                raise result
            if result:
                audio_files.append(result)
        
        if cancel_flag and cancel_flag():
            return None
            
        audio_files.sort()
        out_voice = os.path.join(workdir, 'voice_concat.m4a')
        concat_audios_ffmpeg(
            ffmpeg_path,
            audio_files,
            out_voice,
            sample_rate=24000,
            progress_cb=lambda x: progress_cb(90 + int(x * 0.1)) if progress_cb else None,
            cancel_flag=cancel_flag
        )
        
        return out_voice

    except Exception as e:
        raise RuntimeError(f"TTS processing failed: {str(e)}") from e

async def make_voiceover(ausync_key: str, srt_path: str, video_in: str, out_path: str, *, voice_id: str, language: str = 'vi', speed: float = 1.0, model_name: str = 'myna-1', mute_original: bool = False, ffmpeg_path: str, progress_cb=None, cancel_flag=None, provider: str = 'AusyncLab', openai_model: str = 'tts-1'):
    try:
        with tempfile.TemporaryDirectory() as td:
            voice_audio = await tts_for_srt(
                ausync_key, srt_path, 
                voice_id=voice_id, 
                language=language, 
                speed=speed, 
                model_name=model_name,  # Use model_name from input
                workdir=td, 
                ffmpeg_path=ffmpeg_path, 
                progress_cb=progress_cb, 
                cancel_flag=cancel_flag,
                provider=provider,
                openai_model=openai_model
            )
            
            if not voice_audio:
                if cancel_flag and cancel_flag():
                    raise RuntimeError("Operation cancelled by user")
                raise RuntimeError("Failed to generate voice audio")
                
            if not os.path.exists(voice_audio):
                raise RuntimeError(f"Generated voice audio not found at {voice_audio}")
                
            mux_voiceover(
                ffmpeg_path, 
                video_in, 
                voice_audio, 
                out_path, 
                mute_original=mute_original, 
                progress_cb=progress_cb, 
                cancel_flag=cancel_flag
            )
            
            if not os.path.exists(out_path):
                raise RuntimeError(f"Final output not created at {out_path}")
                
    except Exception as e:
        raise RuntimeError(f"Voiceover creation failed: {str(e)}") from e