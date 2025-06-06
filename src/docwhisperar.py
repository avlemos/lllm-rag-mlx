import sys, os, time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                           QPushButton, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QSplashScreen, QSystemTrayIcon, QMenu)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer, QByteArray
from PyQt6.QtGui import QPixmap, QPainter, QIcon, QAction
from PyQt6.QtSvg import QSvgRenderer
from rag_system import RAGSystem
import threading
if sys.platform == 'darwin':
    from Foundation import NSBundle
    from AppKit import NSApplication, NSApp

# Path to the model cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/blobs")
blobs = "82bfe829fe45ccb46316f2c958c756424381b7a6694f8951fa8cd163a6feea77.incomplete"

class DocWhispererApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

        # Set application name and organization
        self.setApplicationName("DocWhisperer")
        self.setOrganizationName("DocWhisperer")

        # Show the splash screen
        self.splash = DynamicSplashScreen()
        self.splash.show()

        # Process events to ensure the splash is displayed
        self.processEvents()

        # Initialize the RAGSystem in a separate thread
        self.worker_thread = QThread()
        self.worker = RAGSystemWorker()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals to update the splash screen and handle completion
        self.worker.progress.connect(self.update_splash_message)
        self.worker.finished.connect(self.on_rag_system_initialized)

        # Start the worker thread
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def update_splash_message(self, message):
        """Update the splash screen with progress messages."""
        self.splash.update_message(message)
        self.processEvents()

    def on_rag_system_initialized(self, rag_system):
        """Handle completion of RAGSystem initialization."""
        self.rag_system = rag_system
        self.splash.finish(None)  # Close the splash screen
        print("RAG System is ready!")  # Replace with actual app logic


class QueryWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, rag_system, query):
        super().__init__()
        self.rag_system = rag_system
        self.query = query

    def run(self):
        try:
            response = self.rag_system.generate_response(self.query)
            self.finished.emit(response)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.error.emit(error_msg)

class QueryWindow(QMainWindow):
    def __init__(self, rag_system):
        super().__init__()
        self.rag_system = rag_system
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('DocWhisperer')
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create query input
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter your question here...")
        self.query_input.setMaximumHeight(100)
        layout.addWidget(self.query_input)
        
        # Create response display
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        self.response_display.setPlaceholderText("Response will appear here...")
        layout.addWidget(self.response_display)
        
        # Create submit button
        self.submit_button = QPushButton('Ask Question')
        self.submit_button.clicked.connect(self.handle_query)
        layout.addWidget(self.submit_button)
        
        # Create ignore documents checkbox
        # document_count = self.rag_system.get_document_count()
        self.ignore_documents_checkbox = QCheckBox(f"Ignore Documents")
        layout.addWidget(self.ignore_documents_checkbox)
        
        # Set window flags to keep it on top
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

    def handle_query(self):
        query = self.query_input.toPlainText()
        print("query:", query)
        if query:
            if self.ignore_documents_checkbox.isChecked():
                query += " sorry, ignore any provided context, just provide a general answer"
            self.response_display.setPlainText("Generating response...")
            self.submit_button.setEnabled(False)
            
            # Create a QThread object
            self.thread = QThread()
            # Create a worker object
            self.worker = QueryWorker(self.rag_system, query)
            # Move the worker to the thread
            self.worker.moveToThread(self.thread)
            # Connect signals and slots
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.update_response)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.error.connect(self.update_response)
            self.worker.error.connect(self.thread.quit)
            self.worker.error.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            # Start the thread
            self.thread.start()

    def update_response(self, response):
        print("response:", response)
        self.response_display.setPlainText(self.remove_prefix(response))
        self.submit_button.setEnabled(True)
        
    def remove_prefix(self, s):
        prefix = "According to the context, "
        if s.startswith(prefix):
            return s[len(prefix):]  # Remove the prefix
        return s  # Return the original string if it doesn't start with the prefix

class DocWhispererApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        
        # Set application name and organization
        self.setApplicationName("DocWhisperer")
        self.setOrganizationName("DocWhisperer")
        
        if sys.platform == 'darwin':
            # Initialize NSApplication for proper macOS behavior
            NSApplication.sharedApplication()
            info = NSBundle.mainBundle().infoDictionary()
            info["LSBackgroundOnly"] = "1"
            NSApp.setActivationPolicy_(1)
            
            # Create application support directory
            os.makedirs(os.path.expanduser("~/Library/Application Support/DocWhisperer"), exist_ok=True)
        
        # Create and show splash screen
        self.splash = DynamicSplashScreen()
        self.splash.show()
        
        # Process events to ensure splash is shown
        self.processEvents()
        
        # Initialize system tray
        self.init_tray()
        
        # Initialize RAG system in separate thread
        self.thread = QThread()
        self.worker = RAGSystemWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_splash_message)
        self.worker.finished.connect(self.on_rag_system_initialized)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Start the initialization process
        QTimer.singleShot(1000, self.thread.start)
        
        # Prevent application from quitting when last window is closed
        self.setQuitOnLastWindowClosed(False)

    def init_tray(self):
        # Create system tray icon
        self.tray = QSystemTrayIcon(self)
        icon = QIcon('icon.png')
        self.tray.setIcon(icon)
        self.tray.setVisible(True)
        
        # Create tray menu
        self.tray_menu = QMenu()
        
        # Add menu items (they'll be connected later after RAG initialization)
        self.ask_action = QAction("Ask Question")
        self.add_docs_action = QAction("Add Documents")
        self.about_action = QAction("About")
        self.quit_action = QAction("Quit")
        
        self.tray_menu.addAction(self.ask_action)
        self.tray_menu.addAction(self.add_docs_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.about_action)
        self.tray_menu.addAction(self.quit_action)
        
        # Connect quit action
        self.quit_action.triggered.connect(self.quit)
        
        # Set the menu
        self.tray.setContextMenu(self.tray_menu)
        
        # Show the icon
        self.tray.show()

    def update_splash_message(self, message):
        self.splash.update_message(message)
        # self.processEvents()

    def on_rag_system_initialized(self, rag_system):
        self.rag_system = rag_system
        self.query_window = QueryWindow(rag_system)
        
        # Connect menu actions
        self.ask_action.triggered.connect(self.show_query_window)
        self.add_docs_action.triggered.connect(self.add_documents)
        self.about_action.triggered.connect(self.show_about)
        
        # Close splash screen
        self.splash.finish(self.query_window)
        
        # Show notification that app is ready
        self.tray.showMessage(
            "DocWhisperer",
            "Application is ready to use",
            QSystemTrayIcon.MessageIcon.Information
        )
        
    def show_query_window(self):
        if self.query_window:
            self.query_window.show()
            self.query_window.raise_()
            self.query_window.activateWindow()

    def add_documents(self):
        if not self.query_window:
            return
            
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("PDF files (*.pdf)")
        
        if dialog.exec():
            filenames = dialog.selectedFiles()
            self.process_documents(filenames)

    def process_documents(self, file_paths):
        try:
            documents = []
            for path in file_paths:
                text = self.rag_system.process_pdf(path)
                if text:
                    documents.append((path, text))
                    
            if documents:
                self.rag_system.add_documents(documents)
                self.tray.showMessage(
                    "DocWhisperer",
                    f"Successfully processed {len(documents)} documents",
                    QSystemTrayIcon.MessageIcon.Information
                )
        except Exception as e:
            self.tray.showMessage(
                "DocWhisperer",
                f"Error processing documents: {str(e)}",
                QSystemTrayIcon.MessageIcon.Critical
            )

    def show_about(self):
        self.tray.showMessage(
            "About DocWhisperer",
            "DocWhisperer is an intelligent document assistant that helps you interact with your PDF documents using advanced AI technology.",
            QSystemTrayIcon.MessageIcon.Information
        )


class RAGSystemWorker(QObject):
    finished = pyqtSignal(object)  # Emits when the RAGSystem is ready
    progress = pyqtSignal(str)     # Emits progress updates

    def __init__(self):
        super().__init__()
        self._running = True  # Control flag for the thread

    def run(self):
        # cache_dir = os.path.expanduser("~/.cache/mlx-lm/models/")
        # model_name = "Llama-3.2-3B-Instruct-4bit"
        model_path = os.path.join(cache_dir, blobs)


        # Start monitoring in a separate thread
        # if not os.path.exists(cache_dir):
        #     self._running = True
        #     self.monitoring_thread = threading.Thread(target=self.monitor_progress, args=(model_path,))
        #     self.monitoring_thread.start()  # Start the monitor in a separate thread

        # try:
        #     # Emit initial progress
        #     if not os.path.exists(cache_dir):
        #         print("Downloading & Loading the model...")
        #         self.progress.emit("Downloading & Loading the model...")
        #     else:
        #         print("Loading the model...")
        #         self.progress.emit("Loading the model...")

        # Initialize RAG system
        print("Initializing RAG system...")
        rag_system = RAGSystem()

        # Emit progress updates during initialization
        self.progress.emit("Loading document cache...")
        if sys.platform == 'mac':
            rag_system._load_existing_document_cache(
                db_path=os.path.expanduser("~/Library/Application Support/DocWhisperer/rag_cache.db")
            )
        else:
            rag_system._load_existing_document_cache()

        self.progress.emit("Load existing embeddings from storage into FAISS index...")
        rag_system._load_existing_embeddings()

        # Notify completion
        self.progress.emit("Ready!")
        self.finished.emit(rag_system)

        # except Exception as e:
        #     self.progress.emit(f"Error during initialization: {str(e)}")
        # finally:
        #     # Stop the monitoring thread
        #     self._running = False
        #     if hasattr(self, "monitoring_thread") and self.monitoring_thread is not None:
        #         self.monitoring_thread.join()  # Ensure the thread stops gracefully

    def monitor_progress(self, model_path):
        """Monitors model download progress and emits updates."""
        while self._running:
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                print(f"Downloaded: {size / 1e6:.2f} MB")
                self.progress.emit(f"Downloading: {size / 1e6:.2f} MB of 1803.55 MB")
                if size == 1803.55:
                    break
            else:
                print("Waiting for download to start...")
                self.progress.emit("Waiting for the download to start...")
            time.sleep(1)

    def stop(self):
        self._running = False

class DynamicSplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.svg_template = open('splash_template.svg', 'r').read()
        self.update_message("Starting...")

    def update_message(self, message):
        """Update the splash screen with a new message."""
        svg_content = self.svg_template.replace("{loading_text}", message)
        renderer = QSvgRenderer(QByteArray(svg_content.encode()))
        pixmap = QPixmap(600, 400)  # Match the SVG size
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        self.setPixmap(pixmap)
        self.repaint()


class ProgressMonitor(QObject):
    progress = pyqtSignal(str)  # Signal to emit progress updates

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self._running = False

    def monitor(self):
        self._running = True
        while self._running:
            if os.path.exists(self.model_path):
                size = os.path.getsize(self.model_path)
                print(f"Downloaded: {size / 1e6:.2f} MB")
                self.progress.emit(f"Downloading: {size / 1e6:.2f} MB")
            else:
                print("Waiting for download to start...")
                self.progress.emit("Waiting for the download to start...")
            time.sleep(1)

    def stop(self):
        self._running = False

def main():
    # Create and run the application
    app = DocWhispererApp(sys.argv)
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
