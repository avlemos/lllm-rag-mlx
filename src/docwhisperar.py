import rumps
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                           QPushButton, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QSplashScreen, QSystemTrayIcon, QMenu)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer, QByteArray
from PyQt6.QtGui import QPixmap, QPainter, QIcon, QAction
from PyQt6.QtSvg import QSvgRenderer
import os
from rag_system import RAGSystem
import threading
from Foundation import NSBundle
from AppKit import NSApplication, NSApp

# # Initialize NSApplication
# NSApplication.sharedApplication()
# # Hide the Dock icon
# info = NSBundle.mainBundle().infoDictionary()
# info["LSBackgroundOnly"] = "1"
# NSApp.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory


class DynamicSplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.svg_template = open('src/splash_template.svg', 'r').read()
        self.update_message("Starting...")
        
    def update_message(self, message):
        # Update SVG with new message
        svg_content = self.svg_template.replace("{loading_text}", message)
        
        # Convert SVG to QPixmap
        renderer = QSvgRenderer(QByteArray(svg_content.encode()))
        pixmap = QPixmap(600, 400)  # Match SVG viewBox size
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
        # Update splash screen with new pixmap
        self.setPixmap(pixmap)
        self.repaint()

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
        self.response_display.setPlainText(response)
        self.submit_button.setEnabled(True)

class DocWhispererApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        
        # Set application name and organization
        self.setApplicationName("DocWhisperer")
        self.setOrganizationName("DocWhisperer")
        
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
        icon = QIcon('src/icon.png')
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
        self.processEvents()

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

    def quit_app(self, _):
        rumps.quit_application()

class RAGSystemWorker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def run(self):
        try:
            # Emit progress updates during initialization
            self.progress.emit("Loading model...")
            
            # Initialize RAG system
            rag_system = RAGSystem()
            
            # Update progress for different initialization steps
            self.progress.emit("Loading document cache...")
            rag_system._load_existing_document_cache(db_path=os.path.expanduser("~/Library/Application Support/DocWhisperer/rag_cache.db"))
            
            
            self.progress.emit("Load existing embeddings from storage into FAISS index...")
            rag_system._load_existing_embeddings()
            
            self.progress.emit("Ready!")
            self.finished.emit(rag_system)
            
        except Exception as e:
            self.progress.emit(f"Error during initialization: {str(e)}")

def main():
    # Create and run the application
    app = DocWhispererApp(sys.argv)
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
