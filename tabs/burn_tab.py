from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, QMessageBox, QSpinBox, QColorDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSlider, QTextEdit, QProgressBar
from PySide6.QtGui import QFontDatabase, QPixmap
from PySide6.QtCore import Qt, QThread, Signal
from settings import load_settings, save_settings
import os
import subprocess
import tempfile
import re
import srt  # Ensure you have the srt module installed

class BurnWorker(QThread):
    log = Signal(str)
    done = Signal(bool, str)
    progress = Signal(int)  # Thêm tín hiệu tiến trình

    def __init__(self, cfg, preview=False, timestamp=0, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.preview = preview
        self.timestamp = timestamp
        self.cancel_flag = False

    def cancel(self):
        self.cancel_flag = True

    def run(self):
        try:
            if self.cancel_flag:
                self.log.emit('Đã hủy quá trình')
                self.done.emit(False, 'Đã hủy')
                return
            
            video = self.cfg['video']
            srt_file = self.cfg['srt'].replace('\\', '/')  # Đổi tên biến từ srt -> srt_file
            font = self.cfg['font']
            font_size = self.cfg['font_size']
            position = self.cfg['position']
            color = self.cfg['color']
            outline = self.cfg['outline']
            shadow = self.cfg['shadow']
            margin_v = self.cfg['margin_v']
            ffmpeg = self.cfg['ffmpeg']
            out_dir = os.path.dirname(srt_file)
            
            style = f"FontName={font},FontSize={font_size},PrimaryColour={color},Outline={outline},Shadow={shadow},MarginV={margin_v},Alignment={self.get_alignment(position)}"
            srt_escaped = srt_file.replace("'", "\\'").replace(":", "\\:")

            # Kiểm tra file SRT (đã sửa)
            try:
                with open(srt_file, 'r', encoding='utf-8') as f:
                    list(srt.parse(f.read()))  # Giờ đã có thể sử dụng module srt
                self.log.emit("File SRT hợp lệ")
            except Exception as e:
                raise Exception(f"File SRT không hợp lệ: {str(e)}")

            # Lấy thời lượng video
            duration = self.get_video_duration(video, ffmpeg)
            if duration is None:
                self.log.emit("Cảnh báo: Không lấy được thời lượng video, sử dụng mặc định 100 giây")
                duration = 100

            if self.preview:
                temp_file = os.path.join(tempfile.gettempdir(), 'preview.jpg').replace('\\', '/')
                cmd = [
                    ffmpeg, '-ss', str(self.timestamp), '-i', video, '-vf',
                    f"subtitles='{srt_escaped}:force_style={style}'",
                    '-vframes', '1', '-y', temp_file
                ]
                self.log.emit('Đang tạo khung hình xem trước...')
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                if process.returncode != 0:
                    raise Exception(f'FFmpeg lỗi: {process.stderr}')
                if self.cancel_flag:
                    self.log.emit('Đã hủy quá trình')
                    self.done.emit(False, 'Đã hủy')
                    return
                self.log.emit(f'Đã tạo khung hình xem trước: {temp_file}')
                self.done.emit(True, temp_file)
            else:
                out_file = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(video))[0]}_burned.mp4").replace('\\', '/')
                cmd = [
                    ffmpeg, '-i', video,
                    '-vf', f"subtitles='{srt_escaped}':force_style='{style}':sync=0",
                    '-c:v', 'libx264', '-c:a', 'copy', '-y', out_file
                ]
                self.log.emit('Đang ghi phụ đề vào video...')
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                if process.returncode != 0:
                    raise Exception(f'FFmpeg lỗi: {process.stderr}')
                if self.cancel_flag:
                    self.log.emit('Đã hủy quá trình')
                    self.done.emit(False, 'Đã hủy')
                    return
                self.progress.emit(100)
                self.log.emit(f'Đã ghi phụ đề vào video: {out_file}')
                self.done.emit(True, out_file)
        except Exception as e:
            self.log.emit(f'Lỗi: {str(e)}')
            self.done.emit(False, str(e))

    def get_alignment(self, position):
        return {'top': '10', 'center': '5', 'bottom': '2'}.get(position, '2')

    def get_video_duration(self, video, ffmpeg):
        try:
            cmd = [ffmpeg, '-i', video]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', process.stderr)
            if match:
                hours, minutes, seconds = map(float, match.groups())
                return hours * 3600 + minutes * 60 + seconds
            return None
        except Exception:
            return None

class BurnTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_lay = QVBoxLayout(self)
        cfg = load_settings()

        # Video input
        self.video = QLineEdit()
        btn_video = QPushButton('Chọn video')
        btn_video.clicked.connect(self.pick_video)
        video_lay = QHBoxLayout()
        video_lay.addWidget(QLabel('Video'))
        video_lay.addWidget(self.video)
        video_lay.addWidget(btn_video)
        main_lay.addLayout(video_lay)

        # SRT input
        self.srt = QLineEdit()
        btn_srt = QPushButton('Chọn SRT')
        btn_srt.clicked.connect(self.pick_srt)
        srt_lay = QHBoxLayout()
        srt_lay.addWidget(QLabel('Tệp SRT'))
        srt_lay.addWidget(self.srt)
        srt_lay.addWidget(btn_srt)
        main_lay.addLayout(srt_lay)

        # Font selection
        self.font_combo = QComboBox()
        font_db = QFontDatabase()
        self.font_combo.addItems(font_db.families())
        self.font_combo.setCurrentText(cfg.get('burn_font_family', 'Arial'))
        font_lay = QHBoxLayout()
        font_lay.addWidget(QLabel('Font phụ đề'))
        font_lay.addWidget(self.font_combo)
        main_lay.addLayout(font_lay)

        # Font size
        self.font_size = QSpinBox()
        self.font_size.setRange(10, 72)
        self.font_size.setValue(cfg.get('burn_font_size', 24))
        font_size_lay = QHBoxLayout()
        font_size_lay.addWidget(QLabel('Kích thước font'))
        font_size_lay.addWidget(self.font_size)
        main_lay.addLayout(font_size_lay)

        # Position
        self.position_combo = QComboBox()
        self.position_combo.addItems(['Top', 'Center', 'Bottom'])
        self.position_combo.setCurrentText(cfg.get('burn_position', 'Bottom').capitalize())
        pos_lay = QHBoxLayout()
        pos_lay.addWidget(QLabel('Vị trí phụ đề'))
        pos_lay.addWidget(self.position_combo)
        main_lay.addLayout(pos_lay)

        # Vertical margin
        self.margin_v = QSpinBox()
        self.margin_v.setRange(0, 200)
        self.margin_v.setValue(cfg.get('burn_margin_v', 10))
        margin_lay = QHBoxLayout()
        margin_lay.addWidget(QLabel('Khoảng cách dọc (px)'))
        margin_lay.addWidget(self.margin_v)
        main_lay.addLayout(margin_lay)

        # Outline
        self.outline = QSpinBox()
        self.outline.setRange(0, 5)
        self.outline.setValue(cfg.get('burn_outline', 1))
        outline_lay = QHBoxLayout()
        outline_lay.addWidget(QLabel('Độ dày viền (px)'))
        outline_lay.addWidget(self.outline)
        main_lay.addLayout(outline_lay)

        # Shadow
        self.shadow = QSpinBox()
        self.shadow.setRange(0, 5)
        self.shadow.setValue(cfg.get('burn_shadow', 1))
        shadow_lay = QHBoxLayout()
        shadow_lay.addWidget(QLabel('Độ dày bóng (px)'))
        shadow_lay.addWidget(self.shadow)
        main_lay.addLayout(shadow_lay)

        # Color
        self.color_btn = QPushButton('Chọn màu')
        self.color_btn.clicked.connect(self.pick_color)
        self.color = cfg.get('burn_color', '&H00FFFFFF')
        color_lay = QHBoxLayout()
        color_lay.addWidget(QLabel('Màu phụ đề'))
        color_lay.addWidget(self.color_btn)
        main_lay.addLayout(color_lay)

        # Timestamp for preview
        self.timestamp = QSlider(Qt.Horizontal)
        self.timestamp.setRange(0, 100)
        self.timestamp.setValue(0)
        self.timestamp.valueChanged.connect(self.update_timestamp_label)
        self.timestamp_label = QLabel('Thời điểm xem trước: 0s')
        timestamp_lay = QHBoxLayout()
        timestamp_lay.addWidget(QLabel('Thời điểm xem trước'))
        timestamp_lay.addWidget(self.timestamp)
        timestamp_lay.addWidget(self.timestamp_label)
        main_lay.addLayout(timestamp_lay)

        # Split layout for logs and preview
        split_lay = QHBoxLayout()
        
        # Logs section
        logs_lay = QVBoxLayout()
        logs_lay.addWidget(QLabel('Logs'))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumWidth(300)
        logs_lay.addWidget(self.log)
        split_lay.addLayout(logs_lay)

        # Preview section
        preview_lay = QVBoxLayout()
        preview_lay.addWidget(QLabel('Xem trước phụ đề'))
        self.preview_view = QGraphicsView()
        self.preview_scene = QGraphicsScene()
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setMinimumSize(600, 400)
        preview_lay.addWidget(self.preview_view)
        split_lay.addLayout(preview_lay)
        
        main_lay.addLayout(split_lay)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_lay.addWidget(self.progress_bar)

        # Buttons
        self.preview_btn = QPushButton('Xem trước')
        self.preview_btn.clicked.connect(self.preview_subtitles)
        self.start = QPushButton('Ghi phụ đề vào video')
        self.start.clicked.connect(self.start_process)
        self.cancel = QPushButton('Hủy')
        self.cancel.clicked.connect(self.cancel_process)
        btn_lay = QHBoxLayout()
        btn_lay.addWidget(self.preview_btn)
        btn_lay.addWidget(self.start)
        btn_lay.addWidget(self.cancel)
        main_lay.addLayout(btn_lay)

        main_lay.addStretch()
        self.worker = None

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Chọn video', '.', 'Media (*.mp4 *.mkv *.mov)')
        if p:
            self.video.setText(p)
            try:
                cmd = [load_settings().get('ffmpeg_path', 'ffmpeg'), '-i', p]
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                import re
                match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', process.stderr)
                if match:
                    hours, minutes, seconds = map(float, match.groups())
                    duration = hours * 3600 + minutes * 60 + seconds
                    self.timestamp.setMaximum(int(duration))
                    self.update_timestamp_label(self.timestamp.value())
            except Exception:
                pass

    def pick_srt(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Chọn SRT', '.', 'Subtitles (*.srt)')
        if p:
            self.srt.setText(p)

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = f"&H00{color.blue():02X}{color.green():02X}{color.red():02X}"
            self.color_btn.setStyleSheet(f"background-color: {color.name()}")

    def update_timestamp_label(self, value):
        self.timestamp_label.setText(f'Thời điểm xem trước: {value}s')

    def preview_subtitles(self):
        if not os.path.exists(self.video.text().strip()):
            QMessageBox.warning(self, 'Thiếu file', 'Chọn file video trên PC')
            return
        if not os.path.exists(self.srt.text().strip()):
            QMessageBox.warning(self, 'Thiếu file', 'Chọn file SRT')
            return
        cfg = load_settings()
        cfg.update({
            'video': self.video.text().strip(),
            'srt': self.srt.text().strip(),
            'font': self.font_combo.currentText(),
            'font_size': self.font_size.value(),
            'position': self.position_combo.currentText().lower(),
            'color': self.color,
            'outline': self.outline.value(),
            'shadow': self.shadow.value(),
            'margin_v': self.margin_v.value(),
            'ffmpeg': cfg.get('ffmpeg_path', 'ffmpeg')
        })
        save_settings(cfg)
        self.log.clear()
        self.progress_bar.setValue(0)
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.preview_btn.setEnabled(False)
        self.worker = BurnWorker(cfg, preview=True, timestamp=self.timestamp.value())
        self.worker.log.connect(self.log.append)
        self.worker.done.connect(self.on_preview_done)
        self.worker.start()

    def start_process(self):
        if not os.path.exists(self.video.text().strip()):
            QMessageBox.warning(self, 'Thiếu file', 'Chọn file video trên PC')
            return
        if not os.path.exists(self.srt.text().strip()):
            QMessageBox.warning(self, 'Thiếu file', 'Chọn file SRT')
            return
        cfg = load_settings()
        cfg.update({
            'video': self.video.text().strip(),
            'srt': self.srt.text().strip(),
            'font': self.font_combo.currentText(),
            'font_size': self.font_size.value(),
            'position': self.position_combo.currentText().lower(),
            'color': self.color,
            'outline': self.outline.value(),
            'shadow': self.shadow.value(),
            'margin_v': self.margin_v.value(),
            'ffmpeg': cfg.get('ffmpeg_path', 'ffmpeg')
        })
        save_settings(cfg)
        self.log.clear()
        self.progress_bar.setValue(0)
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.preview_btn.setEnabled(False)
        self.worker = BurnWorker(cfg, preview=False)
        self.worker.log.connect(self.log.append)
        self.worker.done.connect(self.on_done)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()

    def cancel_process(self):
        if self.worker:
            self.worker.cancel()
            self.log.append('Đã hủy quá trình')
            self.start.setEnabled(True)
            self.cancel.setEnabled(False)
            self.preview_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.worker = None

    def on_done(self, ok, msg):
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        if ok:
            QMessageBox.information(self, 'Xong', 'Đã ghi phụ đề vào video.')
            self.log.append('Hoàn thành')
        else:
            QMessageBox.critical(self, 'Lỗi', msg)
            self.log.append(f'Lỗi: {msg}')
        self.worker = None

    def on_preview_done(self, ok, msg):
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        if ok:
            self.preview_scene.clear()
            pixmap = QPixmap(msg)
            self.preview_scene.addItem(QGraphicsPixmapItem(pixmap))
            self.preview_view.fitInView(self.preview_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.log.append('Đã hiển thị xem trước')
        else:
            QMessageBox.critical(self, 'Lỗi', msg)
            self.log.append(f'Lỗi: {msg}')
        self.worker = None