import os
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, 
    QTextEdit, QFileDialog, QListWidget, QMessageBox, QSplitter, QProgressBar,
    QGroupBox, QCheckBox, QLineEdit, QSizePolicy, QApplication
)
from PySide6.QtGui import QIcon, QTextCursor, QTextCharFormat, QFont
from PySide6.QtCore import Qt, QThread, Signal
from settings import load_settings, save_settings
from llama_cpp import Llama
from docx import Document  # Thêm thư viện python-docx để đọc file .docx

class LLMWorker(QThread):
    response_received = Signal(str)
    progress_updated = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, model_type, model_name, prompt, files=None, deep_think=False, history=None):
        super().__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.prompt = prompt
        self.files = files or []
        self.deep_think = deep_think
        self.history = history or []
        self.cancel_flag = False

    def run(self):
        try:
            if self.model_type == "offline":
                self.run_offline_model()
            elif self.model_type == "online":
                self.run_online_model()
            else:
                raise ValueError("Loại model không hợp lệ")
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    # ===== OFFLINE MODEL =====
    def _estimate_memory_and_warn(self, model_path: str):
        try:
            import psutil
            file_size = Path(model_path).stat().st_size
            avail_ram = psutil.virtual_memory().available
            need_ram = int(file_size * 1.2) + 1_500_000_000

            warn_msgs = []
            if need_ram > avail_ram:
                warn_msgs.append(
                    f"RAM khả dụng ~{avail_ram/1e9:.1f}GB, cần ~{need_ram/1e9:.1f}GB"
                )

            try:
                from settings import load_settings
                cfg = load_settings()
                n_gpu_layers = int(cfg.get('n_gpu_layers', 0) or 0)
                if n_gpu_layers > 0:
                    import pynvml
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    avail_vram = mem.free
                    est_vram_need = min(file_size, int(file_size * max(0.05, min(1.0, n_gpu_layers / 40.0))))
                    if est_vram_need > avail_vram:
                        warn_msgs.append(
                            f"VRAM khả dụng ~{avail_vram/1e9:.1f}GB, cần ~{est_vram_need/1e9:.1f}GB"
                        )
            except Exception:
                pass

            if warn_msgs:
                self.error_occurred.emit("⚠️ Có thể thiếu tài nguyên:\n- " + "\n- ".join(warn_msgs))
        except Exception:
            pass

    def generate_persona_instruction(self):
        cfg = load_settings()
        if not cfg.get('auto_share_personal', True):
            return ""

        basic_info = []
        if cfg.get('user_name'):
            basic_info.append(f"Tên người dùng: {cfg['user_name']}")
        if cfg.get('user_age'):
            basic_info.append(f"Tuổi: {cfg['user_age']}")
        if cfg.get('user_gender') != "Không tiết lộ":
            basic_info.append(f"Giới tính: {cfg['user_gender']}")

        professional_info = []
        if cfg.get('user_job'):
            professional_info.append(f"Nghề nghiệp: {cfg['user_job']}")
        if cfg.get('user_education'):
            professional_info.append(f"Học vấn: {cfg['user_education']}")
        if cfg.get('user_skills'):
            professional_info.append(f"Kỹ năng: {cfg['user_skills']}")

        personality_info = []
        if cfg.get('user_interests'):
            personality_info.append(f"Sở thích: {cfg['user_interests']}")
        if cfg.get('user_dislikes'):
            personality_info.append(f"Tránh đề cập: {cfg['user_dislikes']}")
        if cfg.get('communication_style'):
            personality_info.append(f"Phong cách trả lời: {self._map_communication_style(cfg['communication_style'])}")

        sections = [
            ("THÔNG TIN CƠ BẢN", basic_info),
            ("NGHIỆP VỤ", professional_info),
            ("TÍNH CÁCH", personality_info)
        ]
        
        instruction = ""
        for section_name, items in sections:
            if items:
                instruction += f"\n\n[{section_name}]\n" + "\n".join(items)
        
        return instruction.strip()

    def _map_communication_style(self, style):
        style_map = {
            "Trực tiếp": "trả lời ngắn gọn, đi thẳng vào vấn đề",
            "Từ tốn": "giải thích chi tiết, có ví dụ minh họa",
            "Hài hước": "thêm yếu tố hài hước khi phù hợp, có thể dùng meme",
            "Hàn lâm": "sử dụng ngôn ngữ chuyên môn, dẫn nguồn tham khảo"
        }
        return style_map.get(style, "phù hợp với ngữ cảnh")

    def run_offline_model(self):
        try:
            if not hasattr(self, "_llm_instance"):
                model_path = f"./models/gguf/{self.model_name}"
                self._estimate_memory_and_warn(model_path)
                self._llm_instance = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=8
                )

            messages = [{"role": "system", "content": "Bạn là trợ lý ảo hữu ích."}]
            messages.extend(self.history)
            messages.append({"role": "user", "content": self.prompt})

            response = self._llm_instance.create_chat_completion(messages=messages)
            self.response_received.emit(response["choices"][0]["message"]["content"])
        except Exception as e:
            self.error_occurred.emit(str(e))

    def run_online_model(self):
        try:
            self.progress_updated.emit(10)
            full_prompt = self.prompt
            if self.files:
                file_contents = []
                for file_path in self.files:
                    try:
                        if file_path.lower().endswith('.docx'):
                            # Đọc file .docx bằng python-docx
                            doc = Document(file_path)
                            content = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                            file_contents.append(f"File: {Path(file_path).name}\nContent:\n{content}")
                        else:
                            # Đọc file văn bản với thử encoding khác nhau
                            encodings = ['utf-8', 'latin-1', 'cp1252']
                            content = None
                            for encoding in encodings:
                                try:
                                    with open(file_path, 'r', encoding=encoding) as f:
                                        content = f.read()
                                    break
                                except UnicodeDecodeError:
                                    continue
                            if content is None:
                                raise UnicodeDecodeError(f"Không thể đọc file {file_path} với bất kỳ encoding nào", b"", 0, 0, "invalid encoding")
                            file_contents.append(f"File: {Path(file_path).name}\nContent:\n{content}")
                    except Exception as e:
                        self.error_occurred.emit(f"Không đọc được file {file_path}: {str(e)}")
                        return
                full_prompt += "\n\nTài liệu đính kèm:\n" + "\n\n".join(file_contents)
            
            self.progress_updated.emit(30)
            if "gpt" in self.model_name.lower():
                response = self.call_chatgpt_api(full_prompt, history=self.history)
            elif "deepseek" in self.model_name.lower():
                response = self.call_deepseek_api(full_prompt, history=self.history)
            else:
                raise ValueError(f"Model {self.model_name} không được hỗ trợ")

            self.progress_updated.emit(90)
            self.response_received.emit(response)
            self.progress_updated.emit(100)

        except Exception as e:
            self.error_occurred.emit(f"Lỗi API: {str(e)}")

    def call_chatgpt_api(self, prompt, history=None):
        from openai import OpenAI
        from settings import load_settings

        cfg = load_settings()
        if not cfg.get("openai_api_key"):
            raise ValueError("Thiếu OpenAI API Key")

        client = OpenAI(api_key=cfg["openai_api_key"])

        # Map tên UI sang model thực tế
        model_map = {
            "gpt-4o": "gpt-4o-mini",
            "gpt-4.1": "gpt-4.1",
            "o3-mini": "o3-mini",
        }
        model = model_map.get(self.model_name.lower(), "gpt-4.1")

        # Xây dựng message list
        messages = [{"role": "system", "content": "Bạn là trợ lý AI thông minh nhất."}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7 if self.deep_think else 0.3,
        )
        return response.choices[0].message.content


    def call_deepseek_api(self, prompt, history=None):
        from settings import load_settings
        cfg = load_settings()
        api_key = cfg.get("deepseek_api_key")
        if not api_key:
            raise ValueError("Thiếu DeepSeek API Key")

        base = "https://api.deepseek.com/v1"
        name = self.model_name.lower()
        if "r1" in name or "reason" in name:
            model = "deepseek-reasoner"
        else:
            model = "deepseek-chat"

        messages = [{"role": "system", "content": "Bạn là trợ lý AI thông minh."}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7 if self.deep_think else 0.3
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=60)
        if r.status_code == 401:
            raise ValueError("DeepSeek API key không hợp lệ")
        if r.status_code == 429:
            raise Exception("DeepSeek: Quá giới hạn. Thử lại sau.")
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

class LLMTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.attached_files = []
        self.setup_ui()
        self.load_offline_models()
        self.apply_theme()
        self.chat_history = []   # ★ lưu hội thoại
        self.snippets = []       # ★ lưu code snippet


    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        layout = QVBoxLayout(self)

        # Splitter for input and response panels
        splitter = QSplitter(Qt.Vertical)

        # Input panel
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Loại mô hình:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Offline", "Online"])
        self.model_type_combo.currentTextChanged.connect(self.update_model_list)
        toolbar.addWidget(self.model_type_combo)

        toolbar.addWidget(QLabel("Mô hình:"))
        self.model_combo = QComboBox()
        toolbar.addWidget(self.model_combo)

        self.deep_think_check = QCheckBox("Deep Think")
        toolbar.addWidget(self.deep_think_check)
        toolbar.addStretch()
        input_layout.addLayout(toolbar)

        # File attachments
        file_group = QGroupBox("Tệp đính kèm")
        file_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        file_layout.addWidget(self.file_list)

        file_btn_layout = QHBoxLayout()
        self.add_file_btn = QPushButton("Thêm tệp")
        self.add_file_btn.setIcon(QIcon.fromTheme("document-open"))
        self.add_file_btn.clicked.connect(self.add_file)
        file_btn_layout.addWidget(self.add_file_btn)

        self.clear_files_btn = QPushButton("Xóa tệp")
        self.clear_files_btn.setIcon(QIcon.fromTheme("edit-clear"))
        self.clear_files_btn.clicked.connect(self.clear_files)
        file_btn_layout.addWidget(self.clear_files_btn)
        file_layout.addLayout(file_btn_layout)
        input_layout.addWidget(file_group)

        # Prompt input
        input_group = QGroupBox("Nhập yêu cầu")
        input_group_layout = QVBoxLayout(input_group)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Nhập yêu cầu của bạn...")
        self.prompt_input.setMaximumHeight(100)
        input_group_layout.addWidget(self.prompt_input)
        input_layout.addWidget(input_group)

        # Action buttons
        action_layout = QHBoxLayout()
        self.search_btn = QPushButton()
        self.search_btn.setIcon(QIcon.fromTheme("system-search"))
        self.search_btn.setToolTip("Tìm kiếm ngữ cảnh")
        self.search_btn.clicked.connect(self.search_context)
        action_layout.addWidget(self.search_btn)

        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon.fromTheme("mail-send"))
        self.send_btn.setToolTip("Gửi yêu cầu")
        self.send_btn.clicked.connect(self.send_prompt)
        action_layout.addWidget(self.send_btn)

        self.cancel_btn = QPushButton()
        self.cancel_btn.setIcon(QIcon.fromTheme("process-stop"))
        self.cancel_btn.setToolTip("Hủy thao tác")
        self.cancel_btn.clicked.connect(self.cancel_operation)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)

        action_layout.addStretch()
        input_layout.addLayout(action_layout)

        # Response panel
        response_group = QGroupBox("Phản hồi")
        response_layout = QVBoxLayout(response_group)
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        response_layout.addWidget(self.response_output)

        # Code snippets
        self.snip_group = QGroupBox("Code Snippets")
        self.snip_layout = QVBoxLayout(self.snip_group)
        self.snip_list = QListWidget()
        self.snip_list.setMaximumHeight(120)
        self.snip_layout.addWidget(self.snip_list)
        snip_btns = QHBoxLayout()
        self.copy_snip_btn = QPushButton("Copy Selected")
        self.copy_snip_btn.clicked.connect(self.copy_selected_snippet)
        self.clear_snip_btn = QPushButton("Clear")
        self.clear_snip_btn.clicked.connect(self.clear_snippets)
        snip_btns.addWidget(self.copy_snip_btn)
        snip_btns.addWidget(self.clear_snip_btn)
        snip_btns.addStretch()
        self.snip_layout.addLayout(snip_btns)
        response_layout.addWidget(self.snip_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        response_layout.addWidget(self.progress_bar)

        layout.addWidget(response_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        response_layout.addWidget(self.progress_bar)

        splitter.addWidget(input_panel)
        splitter.addWidget(response_group)
        splitter.setSizes([400, 600])  # Response panel lớn hơn input panel

        main_layout.addWidget(splitter)

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
        for btn in [self.add_file_btn, self.clear_files_btn, 
                   self.search_btn, self.send_btn, self.cancel_btn]:
            btn.setStyleSheet(button_style)

        # Style text areas and toolbar
        self.apply_theme()

    def apply_theme(self):
        cfg = load_settings()
        theme = cfg.get('theme', 'Light').lower()
        text_color = '#ffffff' if theme == 'dark' else '#000000'
        bg_color = '#333333' if theme == 'dark' else '#ffffff'
        code_bg_color = '#444444' if theme == 'dark' else '#f0f0f0'

        text_style = f"""
            QTextEdit {{
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
                background: {bg_color};
                color: {text_color};
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QComboBox {{
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
                background: {bg_color};
                color: {text_color};
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: url(down_arrow.png);  /* Thay bằng icon thực tế */
            }}
            QLabel {{
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
                color: {text_color};
            }}
        """
        self.prompt_input.setStyleSheet(text_style)
        self.response_output.setStyleSheet(text_style)
        self.model_type_combo.setStyleSheet(text_style)
        self.model_combo.setStyleSheet(text_style)
        for label in self.findChildren(QLabel):
            label.setStyleSheet(text_style)

    def load_offline_models(self):
        model_dir = Path("./models/gguf")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.offline_models = []
        for file in model_dir.glob("*.gguf"):
            self.offline_models.append(file.name)
        
        self.update_model_list()

    def update_model_list(self):
        self.model_combo.clear()
        model_type = self.model_type_combo.currentText().lower()
        
        if model_type == "offline":
            if not self.offline_models:
                self.model_combo.addItem("Không tìm thấy mô hình GGUF")
            else:
                self.model_combo.addItems(self.offline_models)
        else:
            self.model_combo.addItems([
                "ChatGPT (gpt-5-mini)",
                "ChatGPT (gpt-4.1)",
                "ChatGPT (o3-mini)",
                "DeepSeek (R1)",
                "DeepSeek (V3)"
            ])

    def add_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Chọn tệp", "", 
            "All Files (*);;Text Files (*.txt);;PDF Files (*.pdf);;Word Documents (*.docx)"
        )
        if files:
            self.attached_files.extend(files)
            self.file_list.addItems([Path(f).name for f in files])
            # Hiển thị thông báo xác nhận trong response_output
            cfg = load_settings()
            theme = cfg.get('theme', 'Light').lower()
            text_color = '#ffffff' if theme == 'dark' else '#000000'
            self.response_output.append(
                f'<p style="color:{text_color}"><b>📎 Tệp:</b> Đã thêm {len(files)} tệp: {[Path(f).name for f in files]}</p>'
            )

    def clear_files(self):
        self.attached_files.clear()
        self.file_list.clear()
        cfg = load_settings()
        theme = cfg.get('theme', 'Light').lower()
        text_color = '#ffffff' if theme == 'dark' else '#000000'
        self.response_output.append(
            f'<p style="color:{text_color}"><b>🗑️ Tệp:</b> Đã xóa tất cả tệp đính kèm</p>'
        )
    
    def append_and_scroll(self, html_text: str):
        self.response_output.append(html_text)
        self.response_output.moveCursor(QTextCursor.End)   # ★ dùng class constant
        self.response_output.ensureCursorVisible()

    def send_prompt(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập yêu cầu")
            return

        model_type = self.model_type_combo.currentText().lower()
        model_name = self.model_combo.currentText()
        deep_think = self.deep_think_check.isChecked()

        # Add user message to chat history
        cfg = load_settings()
        theme = cfg.get('theme', 'Light').lower()
        text_color = '#ffffff' if theme == 'dark' else '#000000'
        self.response_output.append(
            f'<p style="color:{text_color}"><b>🧑 Bạn:</b> {prompt}</p>'
        )
        self.chat_history.append({"role": "user", "content": prompt})

        # Clear prompt input
        self.prompt_input.clear()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.send_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.append_and_scroll(f'<p>...</p>')

        self.worker = LLMWorker(
            model_type=model_type,
            model_name=model_name,
            prompt=prompt,
            files=self.attached_files,
            deep_think=deep_think,
            history=self.chat_history[:]   # ★ truyền history
        )

        self.worker.response_received.connect(self.handle_response)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def search_context(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập truy vấn tìm kiếm")
            return
        cfg = load_settings()
        theme = cfg.get('theme', 'Light').lower()
        text_color = '#ffffff' if theme == 'dark' else '#000000'
        self.response_output.append(
            f'<p style="color:{text_color}"><b>🔍 Tìm kiếm:</b> {prompt}<br>[Chức năng tìm kiếm sẽ được triển khai tại đây]</p>'
        )

    def cancel_operation(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_flag = True
            self.worker.quit()
            cfg = load_settings()
            theme = cfg.get('theme', 'Light').lower()
            text_color = '#ffffff' if theme == 'dark' else '#000000'
            self.response_output.append(
                f'<p style="color:red"><b>🚫 Hủy:</b> Thao tác đã bị hủy bởi người dùng</p>'
            )

    def handle_response(self, response):
        cfg = load_settings()
        theme = cfg.get('theme', 'Light').lower()
        text_color = '#ffffff' if theme == 'dark' else '#000000'
        code_bg_color = '#444444' if theme == 'dark' else '#f0f0f0'

        # Detect code blocks in response
        lines = response.split('\n')
        formatted_response = []
        in_code_block = False
        code_lines = []
        detected_blocks = []  # ★ danh sách code snippet

        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # end block
                    in_code_block = False
                    code = '\n'.join(code_lines)
                    formatted_response.append(
                        f'<pre style="background:{code_bg_color};border-radius:5px;padding:10px;color:{text_color};font-family:Consolas,monospace">'
                        f'{code}'
                        f'</pre>'
                    )
                    detected_blocks.append(code)
                    code_lines = []
                else:
                    in_code_block = True
            elif in_code_block:
                code_lines.append(line)
            else:
                formatted_response.append(line)

        if in_code_block:
            code = '\n'.join(code_lines)
            formatted_response.append(
                f'<pre style="background:{code_bg_color};border-radius:5px;padding:10px;color:{text_color};font-family:Consolas,monospace">'
                f'{code}'
                f'</pre>'
            )
            detected_blocks.append(code)

        # Hiển thị phản hồi
        formatted_text = '<br>'.join(formatted_response)
        self.response_output.append(
            f'<p style="color:{text_color}"><b>🤖 Bot:</b> {formatted_text}</p>'
        )

        # Add JavaScript for copy button
        self.response_output.setHtml(
            self.response_output.toHtml() +
            '<script>'
            f'function copyCode(id) {{'
            f'  var code = document.getElementsByTagName("pre")[id].innerText;'
            f'  navigator.clipboard.writeText(code);'
            f'}}'
            '</script>'
        )

        # ★ cập nhật snippet list
        for code in detected_blocks:
            self.snippets.append(code)
            self.snip_list.addItem(f"Snippet #{len(self.snippets)}")

        # ★ lưu phản hồi vào history
        self.chat_history.append({"role": "assistant", "content": response})
        self.append_and_scroll(f'<p>...</p>')
    
     # ==== snippet helpers ====
    def copy_selected_snippet(self):
        idx = self.snip_list.currentRow()
        if idx < 0 or idx >= len(self.snippets):
            QMessageBox.information(self, "Copy code", "Chọn 1 snippet trước đã.")
            return
        QApplication.clipboard().setText(self.snippets[idx])
        QMessageBox.information(self, "Copy code", f"Đã copy Snippet #{idx+1}")

    def clear_snippets(self):
        self.snippets.clear()
        self.snip_list.clear()

    def handle_error(self, error):
        cfg = load_settings()
        theme = cfg.get('theme', 'Light').lower()
        text_color = '#ffffff' if theme == 'dark' else '#000000'
        QMessageBox.critical(self, "Lỗi", error)
        self.response_output.append(
            f'<p style="color:red"><b>❌ Lỗi:</b> {error}</p>'
        )

    def worker_finished(self):
        self.progress_bar.setVisible(False)
        self.send_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker = None