import os
import re
import torch
import gc
import json
import soundfile as sf
import asyncio
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QTextEdit, QMessageBox, QProgressBar, QCheckBox, QSplitter, QGroupBox, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from settings import load_settings, save_settings
import srt
import subprocess
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor, WhisperTimeStampLogitsProcessor
from huggingface_hub import snapshot_download
from providers.openai_provider import OpenAIProvider
from providers.deepseek_provider import DeepSeekTranslator

class TranslateWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(str, str, str)

    def __init__(self, video_path, output_dir, engine, lang_from, lang_to, use_wuxia_style, cfg, parent=None):
        super().__init__(parent)
        self.audio_start_time = 0
        self.video_path = video_path
        self.output_dir = output_dir
        self.engine = engine
        self.lang_from = lang_from
        self.lang_to = lang_to
        self.use_wuxia_style = use_wuxia_style
        self.cfg = cfg
        self.cancel_flag = False
        self.whisper_model = None
        self.whisper_processor = None

    def extract_audio(self, video_path, output_audio="temp_audio.wav"):
        try:
            ffmpeg_path = self.cfg.get("ffmpeg_path", "ffmpeg")
            
            # Reset thời gian bắt đầu
            self.audio_start_time = 0
            
            # Trích xuất âm thanh với tham số chính xác hơn
            cmd = [
                ffmpeg_path, "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                "-ss", "0", "-async", "1", output_audio, "-y"
            ]
            subprocess.run(cmd, check=True)
            return output_audio
        except Exception as e:
            raise Exception(f"Lỗi trích xuất âm thanh: {str(e)}")
    
    def detect_audio_start(self, audio_path):
        """Phát hiện thời điểm bắt đầu có âm thanh với xử lý lỗi chi tiết"""
        try:
            cmd = [
                self.cfg.get("ffmpeg_path", "ffmpeg"),
                "-i", audio_path,
                "-af", "silencedetect=noise=-30dB:d=0.5",
                "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.stderr:
                for line in result.stderr.splitlines():
                    if "silence_end" in line:
                        # Xử lý dòng có dạng: "[silencedetect @ 0x7f] silence_end: 31.90517 | silence_duration: 2.23123"
                        parts = line.split("silence_end: ")
                        if len(parts) > 1:
                            time_part = parts[1].split()[0]  # Lấy phần trước ký tự '|' hoặc khoảng trắng
                            try:
                                return float(time_part)
                            except ValueError:
                                self.log.emit(f"Cảnh báo: Không thể chuyển đổi thời gian từ '{time_part}'")
                                continue
            
            self.log.emit("Không phát hiện được thời điểm âm thanh bắt đầu, sử dụng mặc định 0s")
            return 0.0
            
        except Exception as e:
            self.log.emit(f"Cảnh báo: Lỗi khi phát hiện âm thanh - {str(e)}")
            return 0.0

    def load_whisper_model(self):
        model_name = "openai/whisper-large-v3"
        model_dir = Path("./models/large-v3")
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=True
            )
            self.whisper_processor = WhisperProcessor.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=True
            )
        except Exception:
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=model_dir
            )
            self.whisper_processor = WhisperProcessor.from_pretrained(
                model_name,
                cache_dir=model_dir
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.to(device)
        return self.whisper_model, self.whisper_processor

    def transcribe_audio(self):
        import os, re, gc, torch
        from transformers import pipeline, WhisperTimeStampLogitsProcessor
        from huggingface_hub import snapshot_download

        def split_segments_by_rules(segments, max_duration=6, max_chars=80, min_duration=0.6):
            new_segments = []
            seen_texts = set()

            for seg in segments:
                text = seg["text"].strip()
                start, end = seg["start"], seg["end"]

                if text in seen_texts:
                    continue
                seen_texts.add(text)

                # Đặc biệt xử lý cho tiếng Trung: tách theo dấu câu tiếng Trung
                if any(char in text for char in ['。', '！', '？', '，']):
                    sentences = re.split(r'([。！？])', text)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    # Kết hợp dấu câu với câu trước đó
                    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) and len(sentences[i+1]) == 1 else '') 
                                for i in range(0, len(sentences), 2)]
                else:
                    sentences = [text]

                total_time = end - start
                if total_time <= 0:
                    continue

                time_per_char = total_time / max(1, len(text))
                t0 = start

                for sent in sentences:
                    if not sent:
                        continue
                        
                    # Tính thời gian dựa trên độ dài câu
                    seg_dur = min(len(sent) * time_per_char, max_duration)
                    seg_dur = max(seg_dur, min_duration)
                    
                    # Nếu câu quá dài, chia nhỏ
                    if len(sent) > max_chars:
                        parts = [sent[i:i+max_chars] for i in range(0, len(sent), max_chars)]
                        part_dur = seg_dur / len(parts)
                        for p in parts:
                            if p.strip():
                                new_segments.append({
                                    "start": t0,
                                    "end": t0 + part_dur,
                                    "text": p.strip()
                                })
                                t0 += part_dur
                    else:
                        new_segments.append({
                            "start": t0,
                            "end": t0 + seg_dur,
                            "text": sent.strip()
                        })
                        t0 += seg_dur

            return new_segments

        try:
            self.log.emit("Đang xử lý âm thanh...")
            audio_path = self.extract_audio(self.video_path)
            self.audio_start_time = 0
            self.log.emit(f"Phát hiện âm thanh bắt đầu tại: {self.audio_start_time:.2f}s")

            if self.lang_from == "zh" or self.lang_to == "zh":
                self.log.emit("Sử dụng BELLE-Whisper để phiên âm tiếng Trung...")
                torch.cuda.empty_cache()
                model_path = "./models/BELLE-2/Belle-whisper-large-v3-zh"
                repo_id = "BELLE-2/Belle-whisper-large-v3-zh"
                required_files = ["config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"]
                all_files_present = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
                if not os.path.exists(model_path) or not all_files_present:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir="./models/",
                        repo_type="model",
                        resume_download=True,
                        ignore_patterns=["*.md", "*.txt"]
                    )
                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1,
                    model_kwargs={"use_safetensors": True},
                    chunk_length_s=20,
                    stride_length_s=5,
                    ignore_warning=True
                )
                result = asr_pipeline(
                    audio_path,
                    return_timestamps="segment",
                    generate_kwargs={"language": "zh", "task": "transcribe"}
                )
            else:
                self.log.emit("Sử dụng Whisper để phiên âm...")
                if not self.whisper_model or not self.whisper_processor:
                    self.whisper_model, self.whisper_processor = self.load_whisper_model()
                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.whisper_model,
                    tokenizer=self.whisper_processor.tokenizer,
                    feature_extractor=self.whisper_processor.feature_extractor,
                    device=0 if torch.cuda.is_available() else -1,
                    chunk_length_s=20,
                    stride_length_s=5,
                    ignore_warning=True
                )
                result = asr_pipeline(
                    audio_path,
                    return_timestamps="segment",
                    generate_kwargs={
                        "language": self.lang_from,
                        "task": "transcribe",
                        "do_sample": False,
                        "num_beams": 1,
                        "logits_processor": [WhisperTimeStampLogitsProcessor(
                            generate_config=self.whisper_model.generation_config,
                            processor=self.whisper_processor,
                            begin_index=self.whisper_processor.tokenizer.sot_ids[0]
                        )]
                    }
                )

            segments = []
            for seg in result.get("chunks", []):
                start, end = seg.get("timestamp", (None, None))
                text = seg.get("text", "").strip()
                if start is None or end is None or not text:
                    self.log.emit(f"Bỏ qua đoạn không hợp lệ: {text[:50]}...")
                    continue
                # Điều chỉnh thời gian theo audio_start_time
                adjusted_start = max(0, float(start) + self.audio_start_time)
                adjusted_end = max(adjusted_start + 0.1, float(end) + self.audio_start_time)
                segments.append({
                    "start": adjusted_start,
                    "end": adjusted_end, 
                    "text": text
                })

            # Log raw segments
            self.log.emit(f"Tổng số đoạn thô từ BELLE-Whisper: {len(segments)}")
            for i, seg in enumerate(segments):
                self.log.emit(f"Đoạn thô {i+1}: {seg['start']} --> {seg['end']}: {seg['text'][:50]}...")

            segments = split_segments_by_rules(
                segments,
                max_duration=6.0,
                max_chars=80,
                min_duration=0.6
            )

            try:
                os.remove(audio_path)
            except:
                pass
            try:
                del asr_pipeline
            except:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            gc.collect()

            if not segments:
                raise Exception("Không tìm thấy đoạn văn bản nào trong âm thanh")

            return segments
        except Exception as e:
            self.log.emit(f"Lỗi khi phiên âm: {str(e)}")
    
    def create_subtitles(self, segments):
        """Tạo file SRT bắt đầu từ 00:00:00"""
        subs = []
        for i, seg in enumerate(segments):
            # Bắt đầu luôn từ 0 và giữ nguyên khoảng cách giữa các đoạn
            start_time = max(0, seg["start"])
            end_time = max(start_time + 0.1, seg["end"])
            
            subs.append(srt.Subtitle(
                index=i+1,
                start=srt.timedelta(seconds=start_time),
                end=srt.timedelta(seconds=end_time),
                content=seg["text"].strip()
            ))
        
        # Sắp xếp lại các đoạn phụ đề theo thời gian
        subs.sort(key=lambda x: x.start.total_seconds())
        
        # Đánh lại số thứ tự từ 1
        for i, sub in enumerate(subs):
            sub.index = i + 1
            
        return subs

    def _format_timestamp(self, seconds):
        """Chuyển seconds sang định dạng SRT (HH:MM:SS,MSMS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    async def translate_subtitles(self, subs):
        """Dịch phụ đề sử dụng engine đã chọn với Semaphore để giới hạn request đồng thời"""
        try:
            subtitle_texts = [sub.content for sub in subs]
            translated_lines = []
            initial_semaphore_limit = 10
            sem = asyncio.Semaphore(initial_semaphore_limit)
            retry_delay = 5

            if self.cancel_flag:
                raise Exception("Translation canceled by user")

            provider = OpenAIProvider() if self.engine == "OpenAI" else DeepSeekTranslator(self.cfg["deepseek_api_key"])

            async def translate_with_semaphore(index, text):
                nonlocal sem
                max_retries = 3
                retries = 0

                while retries < max_retries:
                    async with sem:
                        try:
                            if len(text) > 1000:
                                text = text[:1000] + "... [TRUNCATED]"
                                self.log.emit(f"Cắt ngắn dòng {index+1} do quá dài: {text[:50]}...")
                            
                            self.log.emit(f"Đang dịch dòng {index+1}/{len(subtitle_texts)}: {text[:50]}...")
                            
                            if self.engine == "OpenAI":
                                result = await provider.translate(
                                    text,
                                    self.lang_from,
                                    self.lang_to,
                                    self.use_wuxia_style
                                )
                            else:
                                result = await provider.translate(
                                    text,
                                    self.lang_to,
                                    self.use_wuxia_style
                                )
                            self.log.emit(f"Kết quả dòng {index+1}: {result[:50]}...")
                            return index, result.strip()

                        except Exception as e:
                            if "rate limit" in str(e).lower() or "429" in str(e):
                                self.log.emit(f"Lỗi rate limit tại dòng {index+1}: {str(e)}")
                                if sem._value == initial_semaphore_limit:
                                    self.log.emit(f"Giảm semaphore từ {initial_semaphore_limit} xuống 5")
                                    sem = asyncio.Semaphore(5)
                                retries += 1
                                if retries < max_retries:
                                    self.log.emit(f"Thử lại dòng {index+1} sau {retry_delay} giây...")
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                continue
                            else:
                                self.log.emit(f"Lỗi dịch dòng {index+1}: {str(e)}")
                                return index, f"[ERROR: {text}]" if self.engine == "OpenAI" else text

                self.log.emit(f"Bỏ qua dòng {index+1} sau {max_retries} lần thử")
                return index, f"[ERROR: {text}]" if self.engine == "OpenAI" else text

            tasks = [translate_with_semaphore(i, text) for i, text in enumerate(subtitle_texts)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = sorted(results, key=lambda x: x[0])
            translated_lines = [result[1] for result in results]

            for i in range(len(subtitle_texts)):
                if self.cancel_flag:
                    raise Exception("Translation canceled by user")
                self.progress.emit(50 + int((i + 1) / len(subtitle_texts) * 50))

            translated_subs = []
            for i, sub in enumerate(subs):
                translated_subs.append(srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=translated_lines[i]
                ))

            return translated_subs

        except Exception as e:
            self.log.emit(f"Lỗi khi dịch phụ đề: {str(e)}")
            raise

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.log.emit("Bắt đầu quá trình dịch phụ đề")
            self.audio_start_time = 0
            self.progress.emit(5)
            
            # 1. Trích xuất âm thanh và phát hiện thời điểm bắt đầu
            audio_path = self.extract_audio(self.video_path)
            self.progress.emit(15)
            
            # 2. Phiên âm âm thanh
            segments = self.transcribe_audio()
            if not segments:
                raise Exception("Không thể phiên âm - không có đoạn văn bản nào được tạo")
            self.progress.emit(40)
            
            # 3. Tạo phụ đề
            subtitles = self.create_subtitles(segments)
            self.progress.emit(50)
            
            # 4. Dịch phụ đề
            translated_subtitles = loop.run_until_complete(
                self.translate_subtitles(subtitles)
            )
            self.progress.emit(80)
            
            # 5. Lưu file
            video_name = Path(self.video_path).stem
            source_srt = os.path.join(self.output_dir, f"{video_name}_{self.lang_from}.srt")
            target_srt = os.path.join(self.output_dir, f"{video_name}_{self.lang_to}.srt")
            
            with open(source_srt, "w", encoding="utf-8") as f:
                f.write(srt.compose(subtitles))
            
            with open(target_srt, "w", encoding="utf-8") as f:
                f.write(srt.compose(translated_subtitles))
            
            self.progress.emit(95)
            
            # Dọn dẹp
            if os.path.exists(audio_path):
                os.remove(audio_path)
            torch.cuda.empty_cache()
            gc.collect()
            
            self.progress.emit(100)
            self.done.emit("success", "Dịch phụ đề hoàn tất", target_srt)
        
        except Exception as e:
            self.log.emit(f"Lỗi: {str(e)}")
            self.done.emit("error", str(e), "")
        
        finally:
            loop.close()

class TranslateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = load_settings()
        self.source_srt_path = None
        self.target_srt_path = None
        self.init_ui()
        self.worker = None

    def init_ui(self):
        # Layout chính
        main_layout = QVBoxLayout(self)

        # --- Phần điều khiển trên cùng ---
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Video input
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video:"))
        self.video_input = QLineEdit()
        video_layout.addWidget(self.video_input)
        btn_video = QPushButton("Chọn...")
        btn_video.clicked.connect(self.select_video)
        video_layout.addWidget(btn_video)
        control_layout.addLayout(video_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Thư mục đầu ra:"))
        self.output_dir_input = QLineEdit(self.cfg.get("out_dir", str(Path(__file__).parent / "subtitles")))
        output_layout.addWidget(self.output_dir_input)
        btn_output = QPushButton("Chọn...")
        btn_output.clicked.connect(self.select_output_dir)
        output_layout.addWidget(btn_output)
        control_layout.addLayout(output_layout)
        
        # Translation settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["OpenAI", "DeepSeek"])
        settings_layout.addWidget(self.engine_combo)
        
        settings_layout.addWidget(QLabel("Từ:"))
        self.lang_from_combo = QComboBox()
        self.lang_from_combo.addItems(["en", "vi", "zh", "ja", "ko"])
        settings_layout.addWidget(self.lang_from_combo)
        
        settings_layout.addWidget(QLabel("Sang:"))
        self.lang_to_combo = QComboBox()
        self.lang_to_combo.addItems(["en", "vi", "zh", "ja", "ko"])
        settings_layout.addWidget(self.lang_to_combo)
        
        self.wuxia_check = QCheckBox("Phong cách kiếm hiệp (Trung-Việt)")
        self.wuxia_check.setEnabled(False)
        settings_layout.addWidget(self.wuxia_check)
        control_layout.addLayout(settings_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Bắt đầu dịch")
        self.start_btn.clicked.connect(self.start_translation)
        btn_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Hủy bỏ")
        self.cancel_btn.clicked.connect(self.cancel_translation)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        control_layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        control_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(control_widget)

         # --- Phần hiển thị SRT ---
        self.srt_splitter = QSplitter(Qt.Horizontal)
        
        # Khung SRT nguồn (cho phép chỉnh sửa)
        source_group = QGroupBox("Phụ đề nguồn")
        source_layout = QVBoxLayout(source_group)
        self.source_table = QTableWidget()
        self.source_table.setColumnCount(3)
        self.source_table.setHorizontalHeaderLabels(["Start", "End", "Text"])
        self.source_table.horizontalHeader().setStretchLastSection(True)
        self.source_table.verticalHeader().setVisible(False)
        self.source_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        source_layout.addWidget(self.source_table)
        
        # Nút lưu và thêm nút chỉnh sửa thời gian
        btn_layout_source = QHBoxLayout()
        self.save_source_btn = QPushButton("Lưu phụ đề nguồn")
        self.save_source_btn.clicked.connect(self.save_source_srt)
        btn_layout_source.addWidget(self.save_source_btn)
        
        self.edit_time_btn_source = QPushButton("Chỉnh thời gian")
        self.edit_time_btn_source.clicked.connect(lambda: self.show_time_editor(self.source_table))
        btn_layout_source.addWidget(self.edit_time_btn_source)
        source_layout.addLayout(btn_layout_source)
        
        # Khung SRT đích (cho phép chỉnh sửa)
        target_group = QGroupBox("Phụ đề đích")
        target_layout = QVBoxLayout(target_group)
        self.target_table = QTableWidget()
        self.target_table.setColumnCount(3)
        self.target_table.setHorizontalHeaderLabels(["Start", "End", "Text"])
        self.target_table.horizontalHeader().setStretchLastSection(True)
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        target_layout.addWidget(self.target_table)
        
        # Nút lưu và thêm nút chỉnh sửa thời gian
        btn_layout_target = QHBoxLayout()
        self.save_target_btn = QPushButton("Lưu phụ đề đích")
        self.save_target_btn.clicked.connect(self.save_target_srt)
        btn_layout_target.addWidget(self.save_target_btn)
        
        self.edit_time_btn_target = QPushButton("Chỉnh thời gian")
        self.edit_time_btn_target.clicked.connect(lambda: self.show_time_editor(self.target_table))
        btn_layout_target.addWidget(self.edit_time_btn_target)
        target_layout.addLayout(btn_layout_target)
        
        self.srt_splitter.addWidget(source_group)
        self.srt_splitter.addWidget(target_group)
        self.srt_splitter.setSizes([400, 400])
        
        main_layout.addWidget(self.srt_splitter, 1)

        # --- Phần log (đơn giản) ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)  # Giới hạn chiều cao log
        main_layout.addWidget(self.log_output)

        # Kết nối signal
        self.lang_from_combo.currentTextChanged.connect(self.update_wuxia_option)
        self.lang_to_combo.currentTextChanged.connect(self.update_wuxia_option)
        self.update_wuxia_option()
    
    def show_time_editor(self, table):
        """Hiển thị dialog chỉnh sửa thời gian cho dòng được chọn"""
        selected_row = table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một dòng để chỉnh sửa")
            return
        
        start_time = table.item(selected_row, 0).text()
        end_time = table.item(selected_row, 1).text()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Chỉnh sửa thời gian")
        layout = QVBoxLayout(dialog)
        
        # Ô nhập thời gian bắt đầu
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Thời gian bắt đầu:"))
        start_edit = QLineEdit(start_time)
        start_layout.addWidget(start_edit)
        
        # Ô nhập thời gian kết thúc
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("Thời gian kết thúc:"))
        end_edit = QLineEdit(end_time)
        end_layout.addWidget(end_edit)
        
        # Nút xác nhận
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        
        layout.addLayout(start_layout)
        layout.addLayout(end_layout)
        layout.addWidget(btn_box)
        
        if dialog.exec() == QDialog.Accepted:
            table.setItem(selected_row, 0, QTableWidgetItem(start_edit.text()))
            table.setItem(selected_row, 1, QTableWidgetItem(end_edit.text()))
    
    def load_srt_to_table(self, table, file_path):
        """Tải nội dung SRT vào QTableWidget với kiểm tra lỗi"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File không tồn tại: {file_path}")
                
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            table.setRowCount(0)  # Xóa dữ liệu cũ
            
            # Sử dụng parser của thư viện srt thay vì regex
            subs = list(srt.parse(content))
            table.setRowCount(len(subs))
            
            for row, sub in enumerate(subs):
                table.setItem(row, 0, QTableWidgetItem(str(sub.start)))
                table.setItem(row, 1, QTableWidgetItem(str(sub.end)))
                table.setItem(row, 2, QTableWidgetItem(sub.content))
                
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể đọc SRT: {str(e)}")
    
    def save_table_to_srt(self, table, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for row in range(table.rowCount()):
                start = table.item(row, 0).text() if table.item(row, 0) else ""
                end = table.item(row, 1).text() if table.item(row, 1) else ""
                text = table.item(row, 2).text() if table.item(row, 2) else ""
                f.write(f"{row+1}\n{start} --> {end}\n{text}\n\n")
        print(f"💾 Saved SRT: {file_path}")

    def toggle_log_visibility(self, visible):
        """Ẩn/hiện phần log"""
        if visible:
            self.log_output.setMaximumHeight(100)
        else:
            self.log_output.setMaximumHeight(0)
    
    def load_srt_files(self, source_path, target_path):
        """Tải nội dung SRT vào các khung xem"""
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                self.source_srt_edit.setPlainText(f.read())
            self.source_srt_path = source_path
            
            if target_path and os.path.exists(target_path):
                with open(target_path, 'r', encoding='utf-8') as f:
                    self.target_srt_edit.setPlainText(f.read())
                self.target_srt_path = target_path
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể tải file SRT: {str(e)}")

    def save_source_srt(self):
        """Lưu phụ đề nguồn"""
        if not hasattr(self, 'source_srt_path') or not self.source_srt_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Lưu phụ đề nguồn", "",
                "Subtitle Files (*.srt)"
            )
            if not path:
                return
            self.source_srt_path = path
        
        self.save_table_to_srt(self.source_table, self.source_srt_path)

    def save_target_srt(self):
        """Lưu phụ đề đích"""
        if not hasattr(self, 'target_srt_path') or not self.target_srt_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Lưu phụ đề đích", "",
                "Subtitle Files (*.srt)"
            )
            if not path:
                return
            self.target_srt_path = path
        
        self.save_table_to_srt(self.target_table, self.target_srt_path)

    # Sửa hàm on_translation_done để tải SRT khi hoàn thành
    def on_translation_done(self, status, message, output_path):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if status == "success":
            video_name = Path(self.video_input.text()).stem
            output_dir = self.output_dir_input.text()
            
            # Đường dẫn file SRT nguồn và đích
            source_srt = Path(output_dir) / f"{video_name}_{self.lang_from_combo.currentText()}.srt"
            target_srt = Path(output_dir) / f"{video_name}_{self.lang_to_combo.currentText()}.srt"
            
            # Tải nội dung vào khung xem
            self.load_srt_files(str(source_srt), str(target_srt))
            
            QMessageBox.information(
                self, 
                "Thành công", 
                f"{message}\nFile đã lưu tại: {output_path}"
            )
        else:
            QMessageBox.critical(self, "Lỗi", message)
    
    def add_find_replace_toolbars(self):
        """Thêm thanh công cụ Find & Replace cho cả 2 khung SRT"""
        # Toolbar cho khung nguồn
        source_toolbar = QHBoxLayout()
        source_toolbar.addWidget(QLabel("Tìm kiếm:"))
        self.source_find_edit = QLineEdit()
        self.source_find_edit.setPlaceholderText("Nhập từ cần tìm...")
        source_toolbar.addWidget(self.source_find_edit)
        
        self.source_replace_edit = QLineEdit()
        self.source_replace_edit.setPlaceholderText("Nhập từ thay thế...")
        source_toolbar.addWidget(self.source_replace_edit)
        
        self.source_find_btn = QPushButton("Tìm")
        self.source_find_btn.clicked.connect(lambda: self.find_text(self.source_srt_edit, self.source_find_edit))
        source_toolbar.addWidget(self.source_find_btn)
        
        self.source_replace_btn = QPushButton("Thay thế")
        self.source_replace_btn.clicked.connect(lambda: self.replace_text(self.source_srt_edit, self.source_find_edit, self.source_replace_edit))
        source_toolbar.addWidget(self.source_replace_btn)
        
        self.source_replace_all_btn = QPushButton("Thay tất cả")
        self.source_replace_all_btn.clicked.connect(lambda: self.replace_all_text(self.source_srt_edit, self.source_find_edit, self.source_replace_edit))
        source_toolbar.addWidget(self.source_replace_all_btn)
        
        # Thêm vào layout khung nguồn
        self.source_srt_widget.layout().insertLayout(1, source_toolbar)
        
        # Toolbar tương tự cho khung đích
        target_toolbar = QHBoxLayout()
        target_toolbar.addWidget(QLabel("Tìm kiếm:"))
        self.target_find_edit = QLineEdit()
        self.target_find_edit.setPlaceholderText("Nhập từ cần tìm...")
        target_toolbar.addWidget(self.target_find_edit)
        
        self.target_replace_edit = QLineEdit()
        self.target_replace_edit.setPlaceholderText("Nhập từ thay thế...")
        target_toolbar.addWidget(self.target_replace_edit)
        
        self.target_find_btn = QPushButton("Tìm")
        self.target_find_btn.clicked.connect(lambda: self.find_text(self.target_srt_edit, self.target_find_edit))
        target_toolbar.addWidget(self.target_find_btn)
        
        self.target_replace_btn = QPushButton("Thay thế")
        self.target_replace_btn.clicked.connect(lambda: self.replace_text(self.target_srt_edit, self.target_find_edit, self.target_replace_edit))
        target_toolbar.addWidget(self.target_replace_btn)
        
        self.target_replace_all_btn = QPushButton("Thay tất cả")
        self.target_replace_all_btn.clicked.connect(lambda: self.replace_all_text(self.target_srt_edit, self.target_find_edit, self.target_replace_edit))
        target_toolbar.addWidget(self.target_replace_all_btn)
        
        # Thêm vào layout khung đích
        self.target_srt_widget.layout().insertLayout(1, target_toolbar)
    
    def find_text(self, text_edit, find_edit):
        """Tìm kiếm text trong QTextEdit với highlight"""
        text_to_find = find_edit.text()
        if not text_to_find:
            return
            
        # Xóa highlight trước đó
        self.clear_highlight(text_edit)
        
        document = text_edit.document()
        cursor = text_edit.textCursor()
        format = QTextCharFormat()
        format.setBackground(QColor("yellow"))
        
        # Tìm tất cả các kết quả và highlight
        count = 0
        cursor = document.find(text_to_find, 0)
        while not cursor.isNull():
            cursor.mergeCharFormat(format)
            count += 1
            cursor = document.find(text_to_find, cursor)
            
        # Di chuyển đến kết quả đầu tiên
        cursor = document.find(text_to_find, 0)
        if not cursor.isNull():
            text_edit.setTextCursor(cursor)
            text_edit.setFocus()
        
        QMessageBox.information(self, "Thông báo", f"Tìm thấy {count} kết quả")
    
    def clear_highlight(self, text_edit):
        """Xóa tất cả highlight"""
        cursor = text_edit.textCursor()
        cursor.movePosition(QTextCursor.Start)
        
        format = QTextCharFormat()
        format.setBackground(QColor("white"))
        
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(format)
        text_edit.setFocus()
    
    def replace_text(self, text_edit, find_edit, replace_edit):
        """Thay thế text hiện tại được tìm thấy"""
        cursor = text_edit.textCursor()
        if cursor.hasSelection():
            cursor.insertText(replace_edit.text())
        self.find_text(text_edit, find_edit)  # Tìm tiếp
    
    def replace_all_text(self, text_edit, find_edit, replace_edit):
        """Thay thế tất cả text phù hợp"""
        text_to_find = find_edit.text()
        if not text_to_find:
            return
            
        # Lấy toàn bộ text
        text = text_edit.toPlainText()
        replaced_text = text.replace(text_to_find, replace_edit.text())
        
        # Cập nhật text mới
        text_edit.setPlainText(replaced_text)
        
        QMessageBox.information(self, "Thông báo", f"Đã thay thế {text.count(text_to_find)} lần xuất hiện")

    def update_wuxia_option(self):
        """Cập nhật trạng thái checkbox wuxia"""
        from_zh = self.lang_from_combo.currentText() == "zh"
        to_vi = self.lang_to_combo.currentText() == "vi"
        self.wuxia_check.setEnabled(from_zh and to_vi)
        if not (from_zh and to_vi):
            self.wuxia_check.setChecked(False)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "", 
            "Video Files (*.mp4 *.mkv *.avi *.mov)"
        )
        if path:
            self.video_input.setText(path)

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, 
            "Chọn thư mục đầu ra", 
            self.output_dir_input.text()
        )
        if path:
            self.output_dir_input.setText(path)

    def start_translation(self):
        # Validate inputs
        video_path = self.video_input.text().strip()
        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file video hợp lệ")
            return
            
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn thư mục đầu ra")
            return
            
        # Kiểm tra và tạo thư mục đầu ra
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo thư mục: {str(e)}")
            return
            
        # Kiểm tra API keys
        engine = self.engine_combo.currentText()
        if engine == "OpenAI" and not self.cfg.get("openai_api_key"):
            QMessageBox.warning(self, "Lỗi", "Thiếu OpenAI API key")
            return
        elif engine == "DeepSeek" and not self.cfg.get("deepseek_api_key"):
            QMessageBox.warning(self, "Lỗi", "Thiếu DeepSeek API key")
            return
            
        # Lưu cài đặt
        self.cfg.update({
            "out_dir": output_dir,
            "use_wuxia_style": self.wuxia_check.isChecked()
        })
        save_settings(self.cfg)
        
        # Chuẩn bị UI
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Khởi tạo worker
        self.worker = TranslateWorker(
            video_path=video_path,
            output_dir=output_dir,
            engine=engine,
            lang_from=self.lang_from_combo.currentText(),
            lang_to=self.lang_to_combo.currentText(),
            use_wuxia_style=self.wuxia_check.isChecked(),
            cfg=self.cfg
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log_output.append)
        self.worker.done.connect(self.on_translation_done)
        self.worker.start()

    def cancel_translation(self):
        if self.worker:
            self.worker.cancel()
            self.log_output.append("Đang hủy quá trình...")
            self.cancel_btn.setEnabled(False)

    def on_translation_done(self, status, message, output_path):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if status == "success":
            QMessageBox.information(
                self, 
                "Thành công", 
                f"{message}\nFile đã lưu tại: {output_path}"
            )
        else:
            QMessageBox.critical(self, "Lỗi", message)