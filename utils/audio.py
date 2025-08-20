import asyncio
from playsound import playsound

async def play_audio(audio_path):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, playsound, audio_path)