import os
import re
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QFileDialog, QComboBox, QTextEdit, QMessageBox, QProgressBar, QCheckBox
from PySide6.QtCore import Qt, QThread, Signal, Slot
from downloader import download_media
from settings import load_settings, save_settings
import subprocess

class DownloadWorker(QThread):
    progress = Signal(object)
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, url, out_dir, quality, ffmpeg_path, use_bbdown=False, bbdown_path='BBDown.exe', parent=None):
        super().__init__(parent)
        self.url = url
        self.out_dir = out_dir
        self.quality = quality
        self.ffmpeg_path = ffmpeg_path
        self.use_bbdown = use_bbdown
        self.bbdown_path = bbdown_path
        self.cancel_flag = False

    def cancel(self):
        self.cancel_flag = True

    def run(self):
        try:
            p = download_media(
                self.url,
                self.out_dir,
                quality=self.quality,
                ffmpeg_path=self.ffmpeg_path,
                progress_cb=lambda x: self.progress.emit(x),
                cancel_flag=lambda: self.cancel_flag,
                use_bbdown=self.use_bbdown,
                bbdown_path=self.bbdown_path
            )
            if p:
                self.log.emit('Đã tải: ' + p)
                self.progress.emit(100)
                self.done.emit(True, p)
            else:
                self.log.emit('Tải xuống bị hủy')
                self.progress.emit(0)
                self.done.emit(False, 'Tải xuống bị hủy')
        except Exception as e:
            self.log.emit('Lỗi: ' + str(e))
            self.progress.emit(0)
            self.done.emit(False, str(e))

class DownloaderTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        cfg = load_settings()
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'downloads'))  # Sửa thành downloads
        self.outdir = QLineEdit(cfg.get('out_dir', default_dir))
        btn_dir = QPushButton('Chọn folder')
        btn_dir.clicked.connect(self.pick_dir)
        btn_dl = QPushButton('Tải về')
        btn_dl.clicked.connect(self.do_download)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.quality = QComboBox()
        self.quality.addItems(['best', '720p', '480p', '360p'])
        self.bilibili_check = QCheckBox('Bilibili [ 1080 ]')
        self.bilibili_check.setChecked(cfg.get('use_bbdown', False))
        self.bilibili_check.stateChanged.connect(self.update_quality)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        hl = QHBoxLayout()
        hl.addWidget(QLabel('URL'))
        self.url = QLineEdit()
        hl.addWidget(self.url)
        lay.addLayout(hl)
        hl1 = QHBoxLayout()
        hl1.addWidget(QLabel('Output folder'))
        hl1.addWidget(self.outdir)
        hl1.addWidget(btn_dir)
        lay.addLayout(hl1)
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel('Quality'))
        hl2.addWidget(self.quality)
        hl2.addWidget(self.bilibili_check)
        hl2.addWidget(btn_dl)
        hl2.addWidget(self.cancel_btn)
        lay.addLayout(hl2)
        lay.addWidget(self.progress_bar)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)
        self.worker = None

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Chọn folder', self.outdir.text())
        if d:
            self.outdir.setText(d)

    @Slot()
    def update_quality(self):
        if self.bilibili_check.isChecked():
            self.quality.setEnabled(False)
            self.quality.setCurrentText('1080p')
        else:
            self.quality.setEnabled(True)

    @Slot()
    def cancel_download(self):
        if self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.log.append('Đã hủy tải xuống')
            self.progress_bar.setValue(0)

    @Slot(object)
    def update_progress(self, value):
        if isinstance(value, (int, float)):
            self.progress_bar.setValue(int(value))
        else:
            self.log.append(str(value))

    @Slot(str)
    def append_log(self, msg):
        self.log.append(msg)

    @Slot(bool, str)
    def on_download_done(self, success, msg):
        self.cancel_btn.setEnabled(False)
        if success:
            QMessageBox.information(self, 'Xong', f'Đã tải: {msg}')
        else:
            QMessageBox.critical(self, 'Lỗi', msg)
        self.worker = None

    @Slot()
    def do_download(self):
        url = self.url.text().strip()
        out = self.outdir.text().strip()
        qual = self.quality.currentText()
        if not url:
            QMessageBox.warning(self, 'Thiếu URL', 'Nhập URL cần tải')
            return
        if not os.access(out, os.W_OK):
            QMessageBox.warning(self, 'Lỗi thư mục', 'Không có quyền ghi vào thư mục đầu ra')
            return
        cfg = load_settings()
        ffmpeg_path = cfg.get('ffmpeg_path', 'ffmpeg')
        bbdown_path = cfg.get('bbdown_path', 'BBDown.exe')
        use_bbdown = self.bilibili_check.isChecked()
        if use_bbdown:
            if not os.path.exists(bbdown_path):
                QMessageBox.warning(self, 'Lỗi', f'BBDown.exe không tìm thấy tại {bbdown_path}')
                return
            if not re.match(r'.*(bilibili\.com/video/(BV|av)|b23\.tv/(BV|av))', url):
                QMessageBox.warning(self, 'Lỗi', 'URL không phải Bilibili. Vui lòng bỏ chọn Bilibili [ 1080 ] hoặc nhập URL Bilibili.')
                return
        if not os.path.exists(ffmpeg_path):
            QMessageBox.warning(self, 'Lỗi', 'FFmpeg không tìm thấy')
            return
        if not use_bbdown and subprocess.run(['yt-dlp', '--version'], capture_output=True).returncode != 0:
            QMessageBox.warning(self, 'Lỗi', 'yt-dlp không được cài đặt')
            return
        cfg.update({'use_bbdown': self.bilibili_check.isChecked(), 'out_dir': out})
        save_settings(cfg)
        self.log.clear()
        self.progress_bar.setValue(0)
        self.cancel_btn.setEnabled(True)
        self.worker = DownloadWorker(url, out, qual, ffmpeg_path, use_bbdown=use_bbdown, bbdown_path=bbdown_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_download_done)
        self.worker.start()