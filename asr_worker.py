import threading, os, logging
logger = logging.getLogger(__name__)
class ASRWorker(threading.Thread):
    def __init__(self, audio_path, out_srt, model='large-v3', device='cuda', progress_cb=None, cancel_flag=None):
        super().__init__(); self.audio_path=audio_path; self.out_srt=out_srt; self.model=model; self.device=device; self.progress_cb=progress_cb; self.cancel_flag=cancel_flag
    def run(self):
        try:
            from faster_whisper import WhisperModel
            model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')
            os.makedirs(model_dir, exist_ok=True)
            model = WhisperModel(self.model, device=self.device, download_root=model_dir)
            segments, info = model.transcribe(self.audio_path)
            import srt, datetime
            subs = []
            count = 0
            for seg in segments:
                if self.cancel_flag and self.cancel_flag():
                    logger.info('ASR cancelled'); return
                subs.append(srt.Subtitle(index=len(subs)+1, start=datetime.timedelta(seconds=seg.start), end=datetime.timedelta(seconds=seg.end), content=seg.text.strip()))
                count += 1
                if self.progress_cb:
                    self.progress_cb(min(99, count))
            with open(self.out_srt, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subs))
            if self.progress_cb: self.progress_cb(100)
        except Exception as e:
            logger.exception('ASR error: %s', e); raise
