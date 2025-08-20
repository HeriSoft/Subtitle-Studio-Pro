from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QMessageBox, QApplication, QFileDialog, QGroupBox, QTextEdit, QFormLayout, QSpinBox
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtCore import Qt
from settings import load_settings, save_settings

class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        cfg = load_settings()

        # Từ điển dịch
        self.translations = {
            'Vietnamese': {
                'OpenAI API Key': 'Khóa API OpenAI',
                'DeepSeek API Key': 'Khóa API DeepSeek',
                'AusyncLab API Key': 'Khóa API AusyncLab',
                'FFmpeg Path': 'Đường dẫn FFmpeg',
                'Browse FFmpeg': 'Chọn FFmpeg',
                'System Font': 'Font hệ thống',
                'Theme': 'Chủ đề',
                'Language': 'Ngôn ngữ',
                'Tự động tải mô hình Hugging Face khi khởi động': 'Tự động tải mô hình Hugging Face khi khởi động',
                'Lưu': 'Lưu',
                'Thành công': 'Thành công',
                'Đã lưu cấu hình.': 'Đã lưu cấu hình.'
            },
            'English': {
                'OpenAI API Key': 'OpenAI API Key',
                'DeepSeek API Key': 'DeepSeek API Key',
                'AusyncLab API Key': 'AusyncLab API Key',
                'FFmpeg Path': 'FFmpeg Path',
                'Browse FFmpeg': 'Browse FFmpeg',
                'System Font': 'System Font',
                'Theme': 'Theme',
                'Language': 'Language',
                'Tự động tải mô hình Hugging Face khi khởi động': 'Auto-download Hugging Face model on startup',
                'Lưu': 'Save',
                'Thành công': 'Success',
                'Đã lưu cấu hình.': 'Settings saved.'
            }
        }

        # OpenAI API Key
        self.openai_key = QLineEdit(cfg.get('openai_api_key', ''))
        self.openai_key.setEchoMode(QLineEdit.Password)
        self.openai_toggle = QPushButton('👁️')
        self.openai_toggle.setObjectName('toggleButton')
        self.openai_toggle.setFixedSize(30, 30)
        self.openai_toggle.clicked.connect(lambda: self.toggle_password(self.openai_key, self.openai_toggle))
        openai_lay = QHBoxLayout()
        openai_lay.addWidget(QLabel(self.tr('OpenAI API Key')))
        openai_lay.addWidget(self.openai_key)
        openai_lay.addWidget(self.openai_toggle)
        lay.addLayout(openai_lay)

        # DeepSeek API Key
        self.deepseek_key = QLineEdit(cfg.get('deepseek_api_key', ''))
        self.deepseek_key.setEchoMode(QLineEdit.Password)
        self.deepseek_toggle = QPushButton('👁️')
        self.deepseek_toggle.setObjectName('toggleButton')
        self.deepseek_toggle.setFixedSize(30, 30)
        self.deepseek_toggle.clicked.connect(lambda: self.toggle_password(self.deepseek_key, self.deepseek_toggle))
        deepseek_lay = QHBoxLayout()
        deepseek_lay.addWidget(QLabel(self.tr('DeepSeek API Key')))
        deepseek_lay.addWidget(self.deepseek_key)
        deepseek_lay.addWidget(self.deepseek_toggle)
        lay.addLayout(deepseek_lay)

        # AusyncLab API Key
        self.ausynclab_key = QLineEdit(cfg.get('ausynclab_api_key', ''))
        self.ausynclab_key.setEchoMode(QLineEdit.Password)
        self.ausynclab_toggle = QPushButton('👁️')
        self.ausynclab_toggle.setObjectName('toggleButton')
        self.ausynclab_toggle.setFixedSize(30, 30)
        self.ausynclab_toggle.clicked.connect(lambda: self.toggle_password(self.ausynclab_key, self.ausynclab_toggle))
        ausynclab_lay = QHBoxLayout()
        ausynclab_lay.addWidget(QLabel(self.tr('AusyncLab API Key')))
        ausynclab_lay.addWidget(self.ausynclab_key)
        ausynclab_lay.addWidget(self.ausynclab_toggle)
        lay.addLayout(ausynclab_lay)

        # FFmpeg path
        self.ffmpeg_path = QLineEdit(cfg.get('ffmpeg_path', 'ffmpeg'))
        btn_ffmpeg = QPushButton(self.tr('Browse FFmpeg'))
        btn_ffmpeg.clicked.connect(self.pick_ffmpeg)
        ffmpeg_lay = QHBoxLayout()
        ffmpeg_lay.addWidget(QLabel(self.tr('FFmpeg Path')))
        ffmpeg_lay.addWidget(self.ffmpeg_path)
        ffmpeg_lay.addWidget(btn_ffmpeg)
        lay.addLayout(ffmpeg_lay)

        # Hugging Face auto-download checkbox
        self.hf_auto_download = QCheckBox(self.tr('Tự động tải mô hình Hugging Face khi khởi động'))
        self.hf_auto_download.setChecked(cfg.get('hf_auto_download', False))
        lay.addWidget(self.hf_auto_download)

        # Font selection
        self.font_combo = QComboBox()
        font_db = QFontDatabase()
        self.font_combo.addItems(font_db.families())
        current_font = cfg.get('font_family', 'Arial')
        self.font_combo.setCurrentText(current_font)
        self.font_combo.currentTextChanged.connect(self.apply_font)
        font_lay = QHBoxLayout()
        font_lay.addWidget(QLabel(self.tr('System Font')))
        font_lay.addWidget(self.font_combo)
        lay.addLayout(font_lay)

        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['Light', 'Dark'])
        self.theme_combo.setCurrentText(cfg.get('theme', 'Light').capitalize())
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        theme_lay = QHBoxLayout()
        theme_lay.addWidget(QLabel(self.tr('Theme')))
        theme_lay.addWidget(self.theme_combo)
        lay.addLayout(theme_lay)

        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(['Vietnamese', 'English'])
        self.language_combo.setCurrentText(cfg.get('language', 'Vietnamese').capitalize())
        self.language_combo.currentTextChanged.connect(self.apply_language)
        lang_lay = QHBoxLayout()
        lang_lay.addWidget(QLabel(self.tr('Language')))
        lang_lay.addWidget(self.language_combo)
        lay.addLayout(lang_lay)

        # Personal Information
        personal_group = QGroupBox("Thông tin cá nhân")
        personal_layout = QVBoxLayout(personal_group)

        # === 3 cột: Cơ bản, Nghề nghiệp, Tính cách ===
        basic_info_group = QGroupBox("Thông tin cơ bản")
        basic_info_layout = QFormLayout()
        
        self.user_name = QLineEdit(cfg.get('user_name', ''))
        self.user_age = QSpinBox()
        self.user_age.setRange(0, 150)
        self.user_age.setValue(cfg.get('user_age', 0))
        self.user_gender = QComboBox()
        self.user_gender.addItems(['Không tiết lộ', 'Nam', 'Nữ', 'Khác'])
        self.user_gender.setCurrentText(cfg.get('user_gender', 'Không tiết lộ'))
        
        basic_info_layout.addRow(QLabel("Tên:"), self.user_name)
        basic_info_layout.addRow(QLabel("Tuổi:"), self.user_age)
        basic_info_layout.addRow(QLabel("Giới tính:"), self.user_gender)
        basic_info_group.setLayout(basic_info_layout)

        career_group = QGroupBox("Thông tin nghề nghiệp")
        career_layout = QFormLayout()
        
        self.user_job = QLineEdit(cfg.get('user_job', ''))
        self.user_education = QComboBox()
        self.user_education.addItems(['Tiểu học', 'Trung học', 'Đại học', 'Sau đại học', 'Khác'])
        self.user_education.setCurrentText(cfg.get('user_education', 'Đại học'))
        self.user_skills = QLineEdit(cfg.get('user_skills', ''))
        
        career_layout.addRow(QLabel("Nghề nghiệp:"), self.user_job)
        career_layout.addRow(QLabel("Học vấn:"), self.user_education)
        career_layout.addRow(QLabel("Kỹ năng:"), self.user_skills)
        career_group.setLayout(career_layout)

        personality_group = QGroupBox("Tính cách")
        personality_layout = QFormLayout()
        
        self.user_interests = QLineEdit(cfg.get('user_interests', ''))
        self.user_dislikes = QLineEdit(cfg.get('user_dislikes', ''))
        self.communication_style = QComboBox()
        self.communication_style.addItems(['Trực tiếp', 'Từ tốn', 'Hài hước', 'Hàn lâm'])
        self.communication_style.setCurrentText(cfg.get('communication_style', 'Trực tiếp'))
        
        personality_layout.addRow(QLabel("Sở thích:"), self.user_interests)
        personality_layout.addRow(QLabel("Không thích:"), self.user_dislikes)
        personality_layout.addRow(QLabel("Phong cách giao tiếp:"), self.communication_style)
        personality_group.setLayout(personality_layout)

        # Tùy chọn chia sẻ
        self.auto_share_check = QCheckBox("Tự động chia sẻ thông tin này khi chat")
        self.auto_share_check.setChecked(cfg.get('auto_share_personal', True))

        # Tooltip cảnh báo
        from PySide6.QtWidgets import QToolTip
        from PySide6.QtCore import QTimer
        
        warning_label = QLabel("⚠️ Thông tin cá nhân")
        warning_label.setToolTip("Thông tin này chỉ lưu cục bộ và được mã hóa")
        
        QTimer.singleShot(1000, lambda: QToolTip.showText(
            warning_label.mapToGlobal(warning_label.rect().bottomLeft()),
            warning_label.toolTip(),
            warning_label
        ))

        # Nút xóa hồ sơ
        self.btn_clear_profile = QPushButton("Xóa toàn bộ hồ sơ")
        self.btn_clear_profile.setStyleSheet("background-color: #ffebee; color: #c62828;")
        self.btn_clear_profile.clicked.connect(self.clear_personal_data)
        
        # Sắp xếp layout
        cols_layout = QHBoxLayout()
        cols_layout.addWidget(basic_info_group)
        cols_layout.addWidget(career_group)
        cols_layout.addWidget(personality_group)
        
        personal_layout.addLayout(cols_layout)
        personal_layout.addWidget(self.auto_share_check)
        personal_layout.addWidget(warning_label)
        personal_layout.addWidget(self.btn_clear_profile, alignment=Qt.AlignRight)
        lay.addWidget(personal_group)

        # Save button
        save_btn = QPushButton(self.tr('Lưu'))
        save_btn.clicked.connect(self.save)
        lay.addWidget(save_btn)
        lay.addStretch()

    def tr(self, text):
        cfg = load_settings()
        language = cfg.get('language', 'Vietnamese')
        return self.translations.get(language, self.translations['Vietnamese']).get(text, text)

    def pick_ffmpeg(self):
        p, _ = QFileDialog.getOpenFileName(self, self.tr('Chọn FFmpeg'), '.', 'Executables (*.exe)')
        if p:
            self.ffmpeg_path.setText(p)

    def toggle_password(self, line_edit, button):
        if line_edit.echoMode() == QLineEdit.Password:
            line_edit.setEchoMode(QLineEdit.Normal)
            button.setText('🙈')
        else:
            line_edit.setEchoMode(QLineEdit.Password)
            button.setText('👁️')

    def save(self):
        cfg = {
            'openai_api_key': self.openai_key.text().strip(),
            'deepseek_api_key': self.deepseek_key.text().strip(),
            'ausynclab_api_key': self.ausynclab_key.text().strip(),
            'ffmpeg_path': self.ffmpeg_path.text().strip(),
            'hf_auto_download': self.hf_auto_download.isChecked(),
            'font_family': self.font_combo.currentText(),
            'theme': self.theme_combo.currentText(),
            'language': self.language_combo.currentText(),
            'user_name': self.user_name.text().strip(),
            'user_age': self.user_age.value(),
            'user_gender': self.user_gender.currentText(),
            'user_job': self.user_job.text().strip(),
            'user_education': self.user_education.currentText(),
            'user_skills': self.user_skills.text().strip(),
            'user_interests': self.user_interests.text().strip(),
            'user_dislikes': self.user_dislikes.text().strip(),
            'communication_style': self.communication_style.currentText(),
            'auto_share_personal': self.auto_share_check.isChecked()
        }
        save_settings(cfg)
        self.apply_font(self.font_combo.currentText())
        self.apply_theme(self.theme_combo.currentText())
        self.apply_language(self.language_combo.currentText())
        QMessageBox.information(self, self.tr('Thành công'), self.tr('Đã lưu cấu hình.'))

    def apply_font(self, font_family):
        app = QApplication.instance()
        app.setFont(QFont(font_family, 12))

    def apply_theme(self, theme):
        app = QApplication.instance()
        stylesheet = f"""
            QWidget {{
                background: {'#333333' if theme.lower() == 'dark' else '#ffffff'};
                color: {'#ffffff' if theme.lower() == 'dark' else '#000000'};
            }}
            QTextEdit {{
                background: {'#333333' if theme.lower() == 'dark' else '#ffffff'};
                color: {'#ffffff' if theme.lower() == 'dark' else '#000000'};
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
            }}
            QComboBox {{
                background: {'#333333' if theme.lower() == 'dark' else '#ffffff'};
                color: {'#ffffff' if theme.lower() == 'dark' else '#000000'};
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QLabel {{
                color: {'#ffffff' if theme.lower() == 'dark' else '#000000'};
            }}
            QPushButton {{
                background: #4CAF50;
                color: white;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
            QPushButton:pressed {{
                background: #3d8b40;
            }}
            QPushButton:disabled {{
                background: #cccccc;
                color: #666666;
            }}
        """
        app.setStyleSheet(stylesheet)
        # Trigger theme update in other tabs
        for widget in app.topLevelWidgets():
            if hasattr(widget, 'apply_theme'):
                widget.apply_theme()

    def apply_language(self, language):
        for widget in self.findChildren(QLabel):
            text = widget.text()
            widget.setText(self.tr(text))
        for widget in self.findChildren(QPushButton):
            text = widget.text()
            if text not in ['👁️', '🙈']:
                widget.setText(self.tr(text))
        for widget in self.findChildren(QCheckBox):
            text = widget.text()
            widget.setText(self.tr(text))
        for widget in self.findChildren(QGroupBox):
            text = widget.title()
            widget.setTitle(self.tr(text))

    def clear_personal_data(self):
        confirm = QMessageBox.question(
            self,
            "Xác nhận",
            "Bạn có chắc muốn xóa toàn bộ hồ sơ cá nhân?\nThao tác này không thể hoàn tác!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            self.user_name.clear()
            self.user_age.setValue(0)
            self.user_gender.setCurrentIndex(0)
            self.user_job.clear()
            self.user_education.setCurrentIndex(0)
            self.user_skills.clear()
            self.user_interests.clear()
            self.user_dislikes.clear()
            self.communication_style.setCurrentIndex(0)
            
            cfg = load_settings()
            for key in [
                'user_name', 'user_age', 'user_gender',
                'user_job', 'user_education', 'user_skills',
                'user_interests', 'user_dislikes', 'communication_style'
            ]:
                if key in cfg:
                    cfg[key] = ''
            
            save_settings(cfg)
            QMessageBox.information(self, "Thành công", "Đã xóa toàn bộ hồ sơ cá nhân")