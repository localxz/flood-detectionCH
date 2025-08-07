import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, QMessageBox,
    QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QTextCursor

    

# Import the main pipeline function from your backend
import backend

# Worker thread to run the analysis without freezing the GUI
class Worker(QThread):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, sar_path, optical_path, weights_path):
        super().__init__()
        self.sar_path = sar_path
        self.optical_path = optical_path
        self.weights_path = weights_path

    def run(self):
        try:
            # The backend function needs a function to call for progress updates.
            # We connect it to our progress signal.
            result_path = backend.run_flood_mapping_pipeline(
                sar_tif=self.sar_path,
                optical_tif=self.optical_path,
                weights_file=self.weights_path,
                progress_callback=self.progress.emit
            )
            self.finished.emit(result_path)
        except Exception as e:
            self.error.emit(f"An error occurred: {e}\n\nCheck console for more details.")
            import traceback
            traceback.print_exc()


# Main Application Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood Mapping Pipeline")
        self.setGeometry(100, 100, 900, 700)

        # --- File Paths ---
        self.sar_path_edit = QLineEdit(self)
        self.sar_path_edit.setPlaceholderText("Path to SAR TIF file...")
        self.sar_path_edit.setReadOnly(True)
        self.sar_browse_btn = QPushButton("Browse...")
        self.sar_browse_btn.clicked.connect(self.browse_sar_file)

        self.optical_path_edit = QLineEdit(self)
        self.optical_path_edit.setPlaceholderText("Path to Optical TIF file...")
        self.optical_path_edit.setReadOnly(True)
        self.optical_browse_btn = QPushButton("Browse...")
        self.optical_browse_btn.clicked.connect(self.browse_optical_file)
        
        # --- Controls ---
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False) # Disabled until files are selected

        # --- Output ---
        self.progress_log = QTextEdit(self)
        self.progress_log.setReadOnly(True)
        self.progress_log.setLineWrapMode(QTextEdit.NoWrap)

        self.image_label = QLabel("Output image will be displayed here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setWidget(self.image_label)
        
        # --- Layout ---
        root_layout = QVBoxLayout()
        
        file_selection_layout = QVBoxLayout()
        sar_layout = QHBoxLayout()
        sar_layout.addWidget(QLabel("SAR TIF:"))
        sar_layout.addWidget(self.sar_path_edit)
        sar_layout.addWidget(self.sar_browse_btn)
        
        optical_layout = QHBoxLayout()
        optical_layout.addWidget(QLabel("Optical TIF:"))
        optical_layout.addWidget(self.optical_path_edit)
        optical_layout.addWidget(self.optical_browse_btn)

        file_selection_layout.addLayout(sar_layout)
        file_selection_layout.addLayout(optical_layout)
        
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(file_selection_layout)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(QLabel("Progress Log:"))
        left_layout.addWidget(self.progress_log)

        main_layout.addLayout(left_layout, 1) # 1 part stretch factor
        main_layout.addWidget(self.image_scroll_area, 1) # 1 part stretch factor
        
        root_layout.addLayout(main_layout)
        
        container = QWidget()
        container.setLayout(root_layout)
        self.setCentralWidget(container)

        self.last_line_is_progress = False
        self.check_inputs()

    def browse_sar_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SAR TIF File", "", "TIF Files (*.tif *.tiff)")
        if path:
            self.sar_path_edit.setText(path)
            self.check_inputs()

    def browse_optical_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Optical TIF File", "", "TIF Files (*.tif *.tiff)")
        if path:
            self.optical_path_edit.setText(path)
            self.check_inputs()

    def check_inputs(self):
        # Enable the 'Run' button only if both paths are filled
        if self.sar_path_edit.text() and self.optical_path_edit.text():
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setEnabled(False)

    def run_analysis(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        self.progress_log.clear()
        self.image_label.setText("Processing, please wait...")
        self.last_line_is_progress = False

        sar_path = self.sar_path_edit.text()
        optical_path = self.optical_path_edit.text()

        # IMPORTANT: This is where you specify the path to your weights file.
        # When packaged, PyInstaller helps us find the correct path.
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            base_path = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            base_path = os.path.dirname(__file__)
        
        weights_path = os.path.join(base_path, 'SegformerJaccardLoss.pth')

        if not os.path.exists(weights_path):
            self.show_error(f"FATAL: Weights file not found!\nExpected at: {weights_path}")
            self.analysis_finished("ERROR")
            return

        # Create and start the worker thread
        self.worker = Worker(sar_path, optical_path, weights_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, message):
        # Check for the special progress message from the backend
        if message.startswith("PROGRESS:"):
            clean_message = message.split(":", 1)[1]
            cursor = self.progress_log.textCursor()

            # If the last line was a progress update, remove it before adding the new one
            if self.last_line_is_progress:
                # Move cursor to the end of the document
                cursor.movePosition(QTextCursor.MoveOperation.End)
                # Select the entire last block (the previous progress line)
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                # Remove the selected text
                cursor.removeSelectedText()

            # Append the new progress message. append() adds a new line.
            self.progress_log.append(clean_message)
            self.last_line_is_progress = True
        else:
            # For all other messages, just append them normally
            self.progress_log.append(message)
            self.last_line_is_progress = False

        # Ensure the log view automatically scrolls to the bottom
        self.progress_log.ensureCursorVisible()

    def analysis_finished(self, result_path):
        if "ERROR" in result_path:
             self.image_label.setText("An error occurred.")
        else:
            self.update_progress(f"Displaying final map: {result_path}")
            pixmap = QPixmap(result_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.check_inputs()

    def show_error(self, error_message):
        self.progress_log.append(error_message)
        QMessageBox.critical(self, "Error", error_message)
        self.analysis_finished("ERROR")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())