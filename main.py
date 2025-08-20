import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from tabs.translate_tab import TranslateTab
from tabs.burn_tab import BurnTab
from tabs.downloader_tab import DownloaderTab
from tabs.convert_tab import ConvertTab
from tabs.llm_tab import LLMTab  # Thêm import cho tab LLM
from tabs.voiceover_tab import VoiceOverTab
from tabs.settings_tab import SettingsTab
from settings import load_settings

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Subtitle Studio Pro')
        self.resize(1000, 700)  # Tăng kích thước cửa sổ để phù hợp với tab mới
        
        # Tạo các tab
        tabs = QTabWidget()
        tabs.addTab(TranslateTab(), 'Translate')
        tabs.addTab(BurnTab(), 'Burn/Subtitles')
        tabs.addTab(DownloaderTab(), 'Downloader')
        tabs.addTab(ConvertTab(), 'Convert')
        tabs.addTab(VoiceOverTab(), 'VoiceOver')
        tabs.addTab(LLMTab(), 'LLM')  # Thêm tab LLM trước Settings
        tabs.addTab(SettingsTab(), 'Settings')
        
        self.setCentralWidget(tabs)
        
        # Áp dụng cài đặt font và theme
        self.apply_settings()
    
    def apply_settings(self):
        """Áp dụng các cài đặt từ file settings"""
        cfg = load_settings()
        
        # Cài đặt font
        font = cfg.get('font_family', 'Arial')
        self.setFont(QFont(font, 12))
        
        # Cài đặt theme
        theme = cfg.get('theme', 'Light')
        stylesheet_file = 'style_dark.qss' if theme.lower() == 'dark' else 'style_light.qss'
        try:
            with open(stylesheet_file, 'r', encoding='utf-8') as f:
                QApplication.instance().setStyleSheet(f.read())
        except FileNotFoundError:
            QApplication.instance().setStyleSheet('')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())