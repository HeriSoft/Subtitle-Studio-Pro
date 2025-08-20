import os
import asyncio
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QMessageBox, QProgressBar, QGroupBox, QDoubleSpinBox, QApplication, QSizePolicy
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon
from settings import load_settings, save_settings
from providers.ausynclab import AusyncLabClient
from voiceover import make_voiceover, tts_for_text
from providers.openai_provider import OpenAIProvider

class VoiceOverWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(str, str)

    def __init__(self, video_path=None, srt_path=None, text_input=None, out_path=None, voice_id=None, cfg=None, is_text_mode=False, provider='AusyncLab', openai_model=None, ausynclab_model='myna-1', parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.srt_path = srt_path
        self.text_input = text_input
        self.out_path = out_path
        self.voice_id = voice_id
        self.cfg = cfg
        self.is_text_mode = is_text_mode
        self.provider = provider
        self.openai_model = openai_model
        self.ausynclab_model = ausynclab_model
        self.cancel_flag = False

    def cancel(self):
        self.cancel_flag = True

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if self.is_text_mode:
                if self.provider == 'OpenAI':
                    if not self.cfg.get('openai_api_key'):
                        self.done.emit('error', 'Thiếu OpenAI API key')
                        return
                    audio_path = loop.run_until_complete(
                        tts_for_text(
                            api_key=self.cfg['openai_api_key'],
                            text=self.text_input,
                            voice_id=self.voice_id,
                            output_path=self.out_path,
                            provider='OpenAI',
                            openai_model=self.openai_model or 'tts-1',
                            progress_cb=lambda x: self.progress.emit(x),
                            cancel_flag=lambda: self.cancel_flag
                        )
                    )
                    self.done.emit('success', audio_path)
                else:
                    if not self.cfg.get('ausynclab_api_key'):
                        self.done.emit('error', 'Thiếu AusyncLab API key')
                        return
                    audio_path = loop.run_until_complete(
                        tts_for_text(
                            api_key=self.cfg['ausynclab_api_key'],
                            text=self.text_input,
                            voice_id=self.voice_id,
                            language='vi',
                            model_name=self.ausynclab_model,
                            output_path=self.out_path,
                            provider='AusyncLab',
                            progress_cb=lambda x: self.progress.emit(x),
                            cancel_flag=lambda: self.cancel_flag
                        )
                    )
                    self.done.emit('success', audio_path)
            else:
                if not self.cfg.get('ausynclab_api_key'):
                    self.done.emit('error', 'Thiếu AusyncLab API key')
                    return
                loop.run_until_complete(
                    make_voiceover(
                        self.cfg['ausynclab_api_key'],
                        self.srt_path,
                        self.video_path,
                        self.out_path,
                        voice_id=self.voice_id,
                        language='vi',
                        speed=1.0,
                        model_name=self.ausynclab_model,
                        mute_original=False,
                        ffmpeg_path=self.cfg['ffmpeg_path'],
                        progress_cb=lambda x: self.progress.emit(x),
                        cancel_flag=lambda: self.cancel_flag,
                        provider=self.provider,
                        openai_model=self.openai_model or 'tts-1'
                    )
                )
                self.done.emit('success', self.out_path)
        except Exception as e:
            self.log.emit(f'Lỗi: {str(e)}')
            self.done.emit('error', str(e))
        finally:
            loop.close()

class TestTTSWorker(QThread):
    log = Signal(str)
    done = Signal(str, str)

    def __init__(self, text, voice_id, cfg, provider, openai_model=None, ausynclab_model='myna-1', parent=None):
        super().__init__(parent)
        self.text = text
        self.voice_id = voice_id
        self.cfg = cfg
        self.provider = provider
        self.openai_model = openai_model
        self.ausynclab_model = ausynclab_model

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if self.provider == 'AusyncLab':
                if not self.cfg.get('ausynclab_api_key'):
                    self.done.emit('error', 'Thiếu AusyncLab API key')
                    return
                client = AusyncLabClient(self.cfg['ausynclab_api_key'])
                async def run_ausynclab():
                    async with client:
                        audio_id = await client.tts_create(
                            text=self.text,
                            voice_id=self.voice_id,
                            language='vi',
                            speed=1.0,
                            model_name=self.ausynclab_model,
                            audio_name='test'
                        )
                        result = await client.tts_poll_until_ready(audio_id, interval=0.8, timeout=300.0)
                        return result['audio_url']
                audio_url = loop.run_until_complete(run_ausynclab())
                loop.run_until_complete(play_audio(audio_url))
            else:
                if not self.cfg.get('openai_api_key'):
                    self.done.emit('error', 'Thiếu OpenAI API key')
                    return
                provider = OpenAIProvider()
                audio_path = loop.run_until_complete(
                    provider.text_to_speech(
                        text=self.text,
                        voice=self.voice_id,
                        speed=1.0,
                        model=self.openai_model or 'tts-1'
                    )
                )
                loop.run_until_complete(play_audio(audio_path))
            self.done.emit('success', 'Phát thử thành công')
        except Exception as e:
            self.log.emit(f'❌ Lỗi: {str(e)}')
            self.done.emit('error', str(e))
        finally:
            loop.close()

class VoiceOverTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.cached_voices = None
        self.update_tts_ui()
        self.worker = None
        self.test_worker = None

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # File Selection Group
        file_group = QGroupBox("File Selection (for Voiceover)")
        file_layout = QVBoxLayout(file_group)
        
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video:"))
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Select video file...")
        btn_video = QPushButton()
        btn_video.setIcon(QIcon.fromTheme("folder"))
        btn_video.clicked.connect(self.pick_video)
        video_layout.addWidget(self.video_path, 4)
        video_layout.addWidget(btn_video, 1)
        file_layout.addLayout(video_layout)

        srt_layout = QHBoxLayout()
        srt_layout.addWidget(QLabel("SRT:"))
        self.srt_path = QLineEdit()
        self.srt_path.setPlaceholderText("Select SRT file...")
        btn_srt = QPushButton()
        btn_srt.setIcon(QIcon.fromTheme("folder"))
        btn_srt.clicked.connect(self.pick_srt)
        srt_layout.addWidget(self.srt_path, 4)
        srt_layout.addWidget(btn_srt, 1)
        file_layout.addLayout(srt_layout)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output:"))
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("Select output file...")
        btn_out = QPushButton()
        btn_out.setIcon(QIcon.fromTheme("folder"))
        btn_out.clicked.connect(self.pick_out)
        out_layout.addWidget(self.out_path, 4)
        out_layout.addWidget(btn_out, 1)
        file_layout.addLayout(out_layout)

        main_layout.addWidget(file_group)

        # TTS Settings Group
        tts_group = QGroupBox("TTS Settings")
        tts_layout = QVBoxLayout(tts_group)
        
        # Provider and Model selection on the same row
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Provider:"))
        self.tts_model_combo = QComboBox()
        self.tts_model_combo.addItems(["AusyncLab", "OpenAI"])
        self.tts_model_combo.currentTextChanged.connect(self.update_tts_ui)
        model_layout.addWidget(self.tts_model_combo, 2)
        
        self.ausynclab_model_label = QLabel("AusyncLab Model:")
        self.ausynclab_model_label.setVisible(True)
        model_layout.addWidget(self.ausynclab_model_label)
        self.ausynclab_model_combo = QComboBox()
        self.ausynclab_model_combo.addItems(["myna-1", "myna-2"])
        self.ausynclab_model_combo.setVisible(True)
        model_layout.addWidget(self.ausynclab_model_combo, 1)
        
        self.openai_model_label = QLabel("OpenAI Model:")
        self.openai_model_label.setVisible(False)
        model_layout.addWidget(self.openai_model_label)
        self.openai_model_combo = QComboBox()
        self.openai_model_combo.addItems(["tts-1", "tts-1-hd"])
        self.openai_model_combo.setVisible(False)
        model_layout.addWidget(self.openai_model_combo, 1)
        tts_layout.addLayout(model_layout)

        # Voice selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.addItem("Loading voices...")
        voice_layout.addWidget(self.tts_voice_combo, 2)
        tts_layout.addLayout(voice_layout)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.tts_speed = QDoubleSpinBox()
        self.tts_speed.setRange(0.25, 4.0)
        self.tts_speed.setSingleStep(0.1)
        self.tts_speed.setValue(1.0)
        self.tts_speed.setEnabled(False)
        speed_layout.addWidget(self.tts_speed)
        tts_layout.addLayout(speed_layout)

        # Text input for TTS
        text_layout = QHBoxLayout()
        text_layout.addWidget(QLabel("Text for TTS:"))
        self.tts_text_input = QTextEdit()
        self.tts_text_input.setPlaceholderText("Enter text to convert to speech...")
        self.tts_text_input.setMaximumHeight(60)
        text_layout.addWidget(self.tts_text_input)
        tts_layout.addLayout(text_layout)

        # Output file for TTS
        tts_out_layout = QHBoxLayout()
        tts_out_layout.addWidget(QLabel("TTS Output:"))
        self.tts_out_path = QLineEdit()
        self.tts_out_path.setPlaceholderText("Select output audio file...")
        btn_tts_out = QPushButton()
        btn_tts_out.setIcon(QIcon.fromTheme("folder"))
        btn_tts_out.clicked.connect(self.pick_tts_out)
        tts_out_layout.addWidget(self.tts_out_path, 4)
        tts_out_layout.addWidget(btn_tts_out, 1)
        tts_layout.addLayout(tts_out_layout)

        # Test and Generate buttons
        tts_button_layout = QHBoxLayout()
        self.tts_test_btn = QPushButton("Test Voice")
        self.tts_test_btn.setIcon(QIcon.fromTheme("audio-volume-high"))
        self.tts_test_btn.clicked.connect(self.start_test_tts)
        self.tts_generate_btn = QPushButton("Generate TTS")
        self.tts_generate_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.tts_generate_btn.clicked.connect(self.generate_tts)
        tts_button_layout.addWidget(self.tts_test_btn)
        tts_button_layout.addWidget(self.tts_generate_btn)
        tts_layout.addLayout(tts_button_layout)

        main_layout.addWidget(tts_group)

        # Action Buttons for Voiceover
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Generate Voiceover")
        self.start_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_btn.clicked.connect(self.start_voiceover)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setIcon(QIcon.fromTheme("process-stop"))
        self.cancel_btn.clicked.connect(self.cancel_voiceover)
        self.cancel_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Status Label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # Compact Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(self.log)

        # Style buttons
        button_style = """
            QPushButton {
                padding: 8px;
                border-radius: 5px;
                border: 1px solid #d0d0d0;
                background: #4CAF50;
                color: white;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background: #45a049;
            }
            QPushButton:pressed {
                background: #3d8b40;
            }
            QPushButton:disabled {
                background: #cccccc;
                color: #666666;
            }
        """
        for btn in [btn_video, btn_srt, btn_out, btn_tts_out, self.tts_test_btn, self.tts_generate_btn, self.start_btn, self.cancel_btn]:
            btn.setStyleSheet(button_style)

    def update_tts_ui(self):
        model = self.tts_model_combo.currentText()
        self.tts_voice_combo.clear()
        self.openai_model_label.setVisible(model == "OpenAI")
        self.openai_model_combo.setVisible(model == "OpenAI")
        self.ausynclab_model_label.setVisible(model == "AusyncLab")
        self.ausynclab_model_combo.setVisible(model == "AusyncLab")
        self.tts_speed.setEnabled(model == "OpenAI")
        
        if model == "AusyncLab":
            self.load_ausynclab_voices()
        else:
            self.load_openai_voices()

    def load_ausynclab_voices(self):
        try:
            if hasattr(self, 'cached_voices') and self.cached_voices:
                self._populate_voice_combo(self.cached_voices)
                return
                
            cfg = load_settings()
            if not cfg.get('ausynclab_api_key'):
                self.log.append('⚠️ Thiếu AusyncLab API key')
                return

            self.tts_voice_combo.clear()
            self.tts_voice_combo.addItem("Đang tải giọng...")
            QApplication.processEvents()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _load():
                client = AusyncLabClient(cfg['ausynclab_api_key'])
                async with client:
                    return await client.list_voices()
                    
            voices = loop.run_until_complete(_load())
            loop.close()
            
            self.cached_voices = voices
            self._populate_voice_combo(voices)
            
        except Exception as e:
            self.log.append(f'❌ Lỗi tải giọng: {str(e)}')
            self.tts_voice_combo.clear()
            self.tts_voice_combo.addItem("Lỗi khi tải giọng")

    def _populate_voice_combo(self, voices):
        self.tts_voice_combo.clear()
        if not voices:
            self.tts_voice_combo.addItem("Không có giọng nào")
            return
            
        self.voice_id_map = {i: v['id'] for i, v in enumerate(voices)}
        self.tts_voice_combo.addItems([
            f"{v['name']} ({v['language']})" 
            for v in sorted(voices, key=lambda x: x['name'])
        ])

    def load_openai_voices(self):
        voices = [
            ("Alloy", "neutral"),
            ("Echo", "friendly"),
            ("Fable", "storytelling"),
            ("Onyx", "serious"),
            ("Nova", "energetic"),
            ("Shimmer", "soft")
        ]
        self.tts_voice_combo.addItems([f"{name} ({style})" for name, style in voices])
        self.voice_id_map = {i: name.lower() for i, (name, _) in enumerate(voices)}

    @Slot()
    def start_test_tts(self):
        text = self.tts_text_input.toPlainText() or "Xin chào, đây là bản xem trước giọng đọc của tôi."
        model = self.tts_model_combo.currentText()
        voice = self.tts_voice_combo.currentText().split(" ")[0]
        
        cfg = load_settings()
        voice_index = self.tts_voice_combo.currentIndex()
        voice_id = self.voice_id_map.get(voice_index, voice)
        
        self.status_label.setText("Đang tạo âm thanh thử...")
        self.test_worker = TestTTSWorker(
            text=text,
            voice_id=voice_id,
            cfg=cfg,
            provider=model,
            openai_model=self.openai_model_combo.currentText() if model == "OpenAI" else None,
            ausynclab_model=self.ausynclab_model_combo.currentText() if model == "AusyncLab" else 'myna-1'
        )
        self.test_worker.log.connect(self.append_log)
        self.test_worker.done.connect(self.on_test_tts_done)
        self.test_worker.start()

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Chọn video', '.', 'Video files (*.mp4 *.mkv *.avi)')
        if p:
            self.video_path.setText(p)

    def pick_srt(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Chọn file SRT', '.', 'SRT files (*.srt)')
        if p:
            self.srt_path.setText(p)

    def pick_out(self):
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
        p, _ = QFileDialog.getSaveFileName(self, 'Chọn file đầu ra', default_dir, 'Video files (*.mp4)')
        if p:
            self.out_path.setText(p)

    def pick_tts_out(self):
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
        p, _ = QFileDialog.getSaveFileName(self, 'Chọn file âm thanh đầu ra', default_dir, 'Audio files (*.mp3 *.m4a)')
        if p:
            self.tts_out_path.setText(p)

    @Slot()
    def cancel_voiceover(self):
        if self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.log.append('Đã hủy lồng tiếng')
            self.progress_bar.setValue(0)
            self.status_label.setText("Đã hủy")
        if self.test_worker:
            self.test_worker.terminate()
            self.log.append('Đã hủy phát thử')
            self.status_label.setText("Đã hủy")

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def append_log(self, msg):
        self.log.append(msg)

    @Slot(str, str)
    def on_test_tts_done(self, status, msg):
        self.test_worker = None
        if status == 'success':
            self.status_label.setText("Phát thử thành công")
        else:
            self.status_label.setText("Lỗi khi phát thử")
            QMessageBox.critical(self, "Lỗi TTS", msg)

    @Slot(str, str)
    def on_voiceover_done(self, status, msg):
        self.cancel_btn.setEnabled(False)
        if status == 'success':
            self.status_label.setText("Hoàn tất")
            QMessageBox.information(self, 'Xong', f'Đã tạo: {msg}')
        else:
            self.status_label.setText("Lỗi")
            QMessageBox.critical(self, 'Lỗi', msg)
        self.worker = None

    @Slot()
    def start_voiceover(self):
        model = self.tts_model_combo.currentText()
        voice = self.tts_voice_combo.currentText().split(" ")[0]
        speed = self.tts_speed.value() if model == "OpenAI" else 1.0
        
        if not self.video_path.text():
            QMessageBox.warning(self, 'Thiếu video', 'Vui lòng chọn file video')
            return
        if not self.srt_path.text():
            QMessageBox.warning(self, 'Thiếu SRT', 'Vui lòng chọn file SRT')
            return
        if not self.out_path.text():
            QMessageBox.warning(self, 'Thiếu đầu ra', 'Vui lòng chọn file đầu ra')
            return
            
        cfg = load_settings()
        if model == "AusyncLab" and not cfg.get('ausynclab_api_key'):
            QMessageBox.warning(self, 'Lỗi', 'Thiếu AusyncLab API key')
            return
        if model == "OpenAI" and not cfg.get('openai_api_key'):
            QMessageBox.warning(self, 'Lỗi', 'Thiếu OpenAI API key')
            return
        if not os.path.exists(cfg.get('ffmpeg_path', 'ffmpeg')):
            QMessageBox.warning(self, 'Lỗi', 'FFmpeg không tìm thấy')
            return
        if not os.access(os.path.dirname(self.out_path.text()), os.W_OK):
            QMessageBox.warning(self, 'Lỗi thư mục', f'Không có quyền ghi vào {os.path.dirname(self.out_path.text())}')
            return
            
        cfg['out_dir'] = os.path.dirname(self.out_path.text())
        save_settings(cfg)
        
        self.log.clear()
        self.progress_bar.setValue(0)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Đang tạo lồng tiếng...")
        
        voice_index = self.tts_voice_combo.currentIndex()
        voice_id = self.voice_id_map.get(voice_index, voice)
        
        self.worker = VoiceOverWorker(
            video_path=self.video_path.text(),
            srt_path=self.srt_path.text(),
            out_path=self.out_path.text(),
            voice_id=voice_id,
            cfg=cfg,
            is_text_mode=False,
            provider=model,
            openai_model=self.openai_model_combo.currentText() if model == "OpenAI" else None,
            ausynclab_model=self.ausynclab_model_combo.currentText() if model == "AusyncLab" else 'myna-1'
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_voiceover_done)
        self.worker.start()

    @Slot()
    def generate_tts(self):
        model = self.tts_model_combo.currentText()
        voice = self.tts_voice_combo.currentText().split(" ")[0]
        text = self.tts_text_input.toPlainText()
        speed = self.tts_speed.value() if model == "OpenAI" else 1.0
        
        if not text:
            QMessageBox.warning(self, 'Thiếu văn bản', 'Vui lòng nhập văn bản để tạo TTS')
            return
        if not self.tts_out_path.text():
            QMessageBox.warning(self, 'Thiếu đầu ra', 'Vui lòng chọn file âm thanh đầu ra')
            return
            
        cfg = load_settings()
        if model == "AusyncLab" and not cfg.get('ausynclab_api_key'):
            QMessageBox.warning(self, 'Lỗi', 'Thiếu AusyncLab API key')
            return
        if model == "OpenAI" and not cfg.get('openai_api_key'):
            QMessageBox.warning(self, 'Lỗi', 'Thiếu OpenAI API key')
            return
            
        self.log.clear()
        self.progress_bar.setValue(0)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Đang tạo âm thanh...")
        
        voice_index = self.tts_voice_combo.currentIndex()
        voice_id = self.voice_id_map.get(voice_index, voice)
        
        self.worker = VoiceOverWorker(
            text_input=text,
            out_path=self.tts_out_path.text(),
            voice_id=voice_id,
            cfg=cfg,
            is_text_mode=True,
            provider=model,
            openai_model=self.openai_model_combo.currentText() if model == "OpenAI" else None,
            ausynclab_model=self.ausynclab_model_combo.currentText() if model == "AusyncLab" else 'myna-1'
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_voiceover_done)
        self.worker.start()