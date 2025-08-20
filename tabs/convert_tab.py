import os
import subprocess
import re
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QLineEdit, QPushButton, QFileDialog, QComboBox, 
                              QTextEdit, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from settings import load_settings, save_settings

class ConvertWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, input_path, output_path, format_type, ffmpeg_path, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.format_type = format_type
        self.ffmpeg_path = ffmpeg_path
        self.cancel_flag = False
        self.last_progress = 0

    def cancel(self):
        self.cancel_flag = True

    def get_input_codecs(self, input_path):
        """Kiểm tra codec của file đầu vào"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-i', input_path],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            video_codec = audio_codec = video_profile = None
            video_bit_depth = 8
            
            for line in result.stderr.split('\n'):
                if 'Stream #' in line and 'Video:' in line:
                    codec_info = line.lower()
                    if 'h264' in codec_info:
                        video_codec = 'h264'
                    elif 'hevc' in codec_info or 'h265' in codec_info:
                        video_codec = 'hevc'
                    elif 'vp9' in codec_info:
                        video_codec = 'vp9'
                    elif 'av1' in codec_info:
                        video_codec = 'av1'
                    elif 'mpeg4' in codec_info:
                        video_codec = 'mpeg4'
                    elif 'mpeg2' in codec_info:
                        video_codec = 'mpeg2'
                    
                    if 'profile=' in codec_info:
                        profile_match = re.search(r'profile=([^\s,]+)', codec_info)
                        if profile_match:
                            video_profile = profile_match.group(1)
                    
                    if 'yuv420p10le' in codec_info:
                        video_bit_depth = 10
                    elif 'yuv420p12le' in codec_info:
                        video_bit_depth = 12
                
                if 'Stream #' in line and 'Audio:' in line:
                    audio_info = line.lower()
                    if 'aac' in audio_info:
                        audio_codec = 'aac'
                    elif 'mp3' in audio_info:
                        audio_codec = 'mp3'
                    elif 'opus' in audio_info:
                        audio_codec = 'opus'
                    elif 'vorbis' in audio_info:
                        audio_codec = 'vorbis'
                    elif 'pcm' in audio_info:
                        audio_codec = 'pcm'
                    elif 'flac' in audio_info:
                        audio_codec = 'flac'
                    elif 'ac3' in audio_info:
                        audio_codec = 'ac3'
                    elif 'eac3' in audio_info:
                        audio_codec = 'eac3'
            
            return video_codec, audio_codec, video_profile, video_bit_depth
        except Exception as e:
            self.log.emit(f'Lỗi kiểm tra codec đầu vào: {str(e)}')
            return None, None, None, 8

    def get_video_duration(self, video_path):
        """Lấy thời lượng video chính xác"""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-hide_banner',
                '-f', 'null',
                '-'
            ]
            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            duration_match = re.search(
                r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})',
                result.stderr
            )
            if duration_match:
                hours, minutes, seconds = map(float, duration_match.groups())
                return hours * 3600 + minutes * 60 + seconds
            
            return None
        except Exception as e:
            self.log.emit(f'Lỗi lấy thời lượng: {str(e)}')
            return None

    def validate_output(self):
        """Kiểm tra tính toàn vẹn của file output"""
        try:
            cmd = [
                self.ffmpeg_path,
                '-v', 'error',
                '-i', self.output_path,
                '-f', 'null',
                '-'
            ]
            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            self.log.emit(f'Lỗi kiểm tra file đầu ra: {str(e)}')
            return False

    def run_ffmpeg_process(self, cmd, duration):
        """Chạy FFmpeg và theo dõi tiến trình"""
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        
        while True:
            if self.cancel_flag:
                process.terminate()
                self.log.emit('Đã hủy quá trình')
                self.done.emit(False, 'Hủy bỏ')
                return
                
            line = process.stderr.readline()
            if not line:
                break
                
            self.log.emit(line.strip())
            
            # Cập nhật progress
            if duration != float('inf'):
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
                if time_match:
                    hours, minutes, seconds = map(float, time_match.groups())
                    current_time = hours * 3600 + minutes * 60 + seconds
                    progress = min(100, int((current_time / duration) * 100))
                    
                    # Chỉ emit khi progress thay đổi đáng kể
                    if progress > self.last_progress:
                        self.progress.emit(progress)
                        self.last_progress = progress
        
        # Đảm bảo process kết thúc
        process.wait()
        
        # Kiểm tra kết quả
        if process.returncode != 0:
            msg = f'Lỗi FFmpeg (mã {process.returncode})'
            self.log.emit(msg)
            self.done.emit(False, msg)
            return
            
        if not os.path.exists(self.output_path):
            msg = f'File đầu ra không được tạo: {self.output_path}'
            self.log.emit(msg)
            self.done.emit(False, msg)
            return
            
        if not self.validate_output():
            msg = 'File đầu ra không hợp lệ hoặc không đầy đủ'
            self.log.emit(msg)
            self.done.emit(False, msg)
            return
            
        # Hoàn thành
        self.progress.emit(100)
        self.log.emit(f'Chuyển đổi thành công: {self.output_path}')
        self.done.emit(True, self.output_path)

    def build_ffmpeg_command(self):
        """Xây dựng lệnh FFmpeg phù hợp"""
        video_codec, audio_codec, video_profile, video_bit_depth = self.get_input_codecs(self.input_path)
        cmd = [self.ffmpeg_path, '-i', self.input_path, '-map', '0:v:0?', '-map', '0:a:0?', '-threads', '0', '-y']
        
        pixel_format = 'yuv420p' if video_bit_depth == 8 else f'yuv420p{video_bit_depth}le'
        
        if self.format_type == 'ts':
            if video_codec in ['h264', 'hevc', 'mpeg2'] and audio_codec in ['aac', 'mp3', 'ac3', 'eac3']:
                cmd.extend(['-c:v', 'copy', '-c:a', 'copy', '-bsf:v', f'{video_codec}_mp4toannexb'])
            else:
                cmd.extend([
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '256k', '-vf', 'scale=iw:ih', 
                    '-pix_fmt', pixel_format, '-profile:v', 'high' if video_bit_depth == 8 else 'main10',
                    '-level', '4.1'
                ])
            cmd.extend(['-fflags', '+genpts'])
            
        elif self.format_type == 'mp4':
            if video_codec in ['h264', 'hevc', 'av1'] and audio_codec in ['aac', 'mp3', 'opus']:
                cmd.extend(['-c:v', 'copy', '-c:a', 'copy'])
            else:
                cmd.extend([
                    '-c:v', 'libx265', '-preset', 'medium', '-crf', '20',
                    '-c:a', 'libopus', '-b:a', '192k', '-vf', 'scale=iw:ih',
                    '-pix_fmt', pixel_format, '-profile:v', 'main' if video_bit_depth == 8 else 'main10',
                    '-tag:v', 'hvc1'
                ])
                
        elif self.format_type == 'mkv':
            if video_codec and audio_codec:
                cmd.extend(['-c:v', 'copy', '-c:a', 'copy'])
            else:
                cmd.extend([
                    '-c:v', 'libx265', '-preset', 'medium', '-crf', '20',
                    '-c:a', 'flac' if audio_codec == 'flac' else 'libopus',
                    '-b:a', '0' if audio_codec == 'flac' else '192k',
                    '-vf', 'scale=iw:ih', '-pix_fmt', pixel_format
                ])
                
        elif self.format_type == 'webm':
            if video_codec in ['vp9', 'av1'] and audio_codec in ['opus', 'vorbis']:
                cmd.extend(['-c:v', 'copy', '-c:a', 'copy'])
            else:
                cmd.extend([
                    '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0',
                    '-row-mt', '1', '-quality', 'good', '-speed', '2',
                    '-c:a', 'libopus', '-b:a', '192k', '-vf', 'scale=iw:ih',
                    '-pix_fmt', pixel_format
                ])
                
        elif self.format_type in ['mp3', 'aac', 'flac', 'wav', 'ogg']:
            cmd = [self.ffmpeg_path, '-i', self.input_path, '-vn', '-y']
            if self.format_type == 'mp3':
                cmd.extend(['-c:a', 'libmp3lame', '-q:a', '2'])
            elif self.format_type == 'aac':
                cmd.extend(['-c:a', 'aac', '-b:a', '256k'])
            elif self.format_type == 'flac':
                cmd.extend(['-c:a', 'flac', '-compression_level', '8'])
            elif self.format_type == 'wav':
                cmd.extend(['-c:a', 'pcm_s24le'])
            elif self.format_type == 'ogg':
                cmd.extend(['-c:a', 'libvorbis', '-b:a', '192k'])
                
        cmd.append(self.output_path)
        return cmd

    def run(self):
        self.log.emit(f'Bắt đầu chuyển đổi: {self.input_path} -> {self.output_path}')
        
        try:
            # Kiểm tra đầu vào
            if not os.path.exists(self.ffmpeg_path):
                msg = f'FFmpeg không tìm thấy tại: {self.ffmpeg_path}'
                self.log.emit(msg)
                self.done.emit(False, msg)
                return
                
            if not os.path.exists(self.input_path):
                msg = f'File đầu vào không tồn tại: {self.input_path}'
                self.log.emit(msg)
                self.done.emit(False, msg)
                return
                
            # Lấy thời lượng video
            duration = self.get_video_duration(self.input_path)
            if duration is None:
                self.log.emit('Không thể xác định thời lượng video')
                duration = float('inf')
            else:
                self.log.emit(f'Thời lượng video: {duration:.2f} giây')
                
            # Xây dựng và chạy lệnh FFmpeg
            cmd = self.build_ffmpeg_command()
            self.log.emit(f'Lệnh FFmpeg: {" ".join(cmd)}')
            self.run_ffmpeg_process(cmd, duration)
            
        except Exception as e:
            self.log.emit(f'Lỗi không mong muốn: {str(e)}')
            self.done.emit(False, str(e))

class ConvertTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.load_settings()
        self.worker = None

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Input file selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel('File đầu vào:'))
        self.input_path = QLineEdit()
        input_layout.addWidget(self.input_path)
        btn_input = QPushButton('Chọn...')
        btn_input.clicked.connect(self.pick_input)
        input_layout.addWidget(btn_input)
        layout.addLayout(input_layout)
        
        # Output file selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel('File đầu ra:'))
        self.output_path = QLineEdit()
        output_layout.addWidget(self.output_path)
        btn_output = QPushButton('Chọn...')
        btn_output.clicked.connect(self.pick_output)
        output_layout.addWidget(btn_output)
        layout.addLayout(output_layout)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel('Loại:'))
        self.type_combo = QComboBox()
        self.type_combo.addItems(['Video', 'Audio'])
        self.type_combo.currentTextChanged.connect(self.update_format_options)
        format_layout.addWidget(self.type_combo)
        
        format_layout.addWidget(QLabel('Định dạng:'))
        self.format_combo = QComboBox()
        format_layout.addWidget(self.format_combo)
        
        format_layout.addWidget(QLabel('Chất lượng:'))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(['Tốt nhất', 'Cân bằng', 'Nhanh'])
        format_layout.addWidget(self.quality_combo)
        layout.addLayout(format_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton('Bắt đầu')
        self.start_btn.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton('Hủy')
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        self.update_format_options()

    def load_settings(self):
        cfg = load_settings()
        self.format_combo.setCurrentText(cfg.get('convert_format', 'mp4'))
        self.quality_combo.setCurrentText(cfg.get('convert_quality', 'Cân bằng'))

    def update_format_options(self):
        current_type = self.type_combo.currentText()
        self.format_combo.clear()
        
        if current_type == 'Video':
            self.format_combo.addItems(['mp4', 'mkv', 'mov', 'webm', 'ts', 'avi'])
        else:
            self.format_combo.addItems(['mp3', 'aac', 'flac', 'wav', 'ogg'])
        
        self.update_output_path()

    def update_output_path(self):
        if not self.input_path.text():
            return
            
        base_name = os.path.splitext(os.path.basename(self.input_path.text()))[0]
        output_dir = os.path.dirname(self.input_path.text())
        output_ext = self.format_combo.currentText()
        self.output_path.setText(os.path.join(output_dir, f"{base_name}.{output_ext}"))

    def pick_input(self):
        file_filter = "Media files (*.mp4 *.mkv *.avi *.mov *.webm *.ts *.mp3 *.aac *.flac *.wav *.ogg)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file đầu vào", "", file_filter)
        
        if file_path:
            self.input_path.setText(file_path)
            self.update_output_path()
            self.log_display.append(f"Đã chọn file đầu vào: {file_path}")

    def pick_output(self):
        if not self.input_path.text():
            return
            
        output_ext = self.format_combo.currentText()
        default_dir = os.path.dirname(self.input_path.text())
        default_name = os.path.splitext(os.path.basename(self.input_path.text()))[0] + f".{output_ext}"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Chọn file đầu ra",
            os.path.join(default_dir, default_name),
            f"{output_ext.upper()} files (*.{output_ext})"
        )
        
        if file_path:
            self.output_path.setText(file_path)
            self.log_display.append(f"Đã chọn file đầu ra: {file_path}")

    def start_conversion(self):
        if not self.input_path.text() or not self.output_path.text():
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file đầu vào và đầu ra")
            return
            
        cfg = load_settings()
        ffmpeg_path = cfg.get('ffmpeg_path', 'ffmpeg')
        
        if not os.path.exists(ffmpeg_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy FFmpeg tại: {ffmpeg_path}")
            return
            
        # Lưu cài đặt
        cfg['convert_format'] = self.format_combo.currentText()
        cfg['convert_quality'] = self.quality_combo.currentText()
        save_settings(cfg)
        
        # Chuẩn bị cho quá trình convert
        self.log_display.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        self.worker = ConvertWorker(
            self.input_path.text(),
            self.output_path.text(),
            self.format_combo.currentText(),
            ffmpeg_path
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.conversion_finished)
        self.worker.start()

    def cancel_conversion(self):
        if self.worker:
            self.worker.cancel()
            self.append_log("Đã yêu cầu hủy bỏ...")
            self.cancel_btn.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value % 10 == 0:  # Chỉ log mỗi 10%
            self.append_log(f"Tiến trình: {value}%")

    def append_log(self, message):
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def conversion_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if success:
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Hoàn thành", f"Chuyển đổi thành công:\n{message}")
        else:
            QMessageBox.critical(self, "Lỗi", f"Chuyển đổi thất bại:\n{message}")
        
        self.worker = None