import rumps
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                           QPushButton, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QSplashScreen)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap
import os
from rag_system import RAGSystem
import threading
from Foundation import NSBundle
from AppKit import NSApplication, NSApp

# Initialize NSApplication
NSApplication.sharedApplication()
# Hide the Dock icon
info = NSBundle.mainBundle().infoDictionary()
info["LSBackgroundOnly"] = "1"
NSApp.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory

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

class DocWhispererApp(rumps.App):
    def __init__(self):
        super().__init__("DocWhisperer", icon="icon.png")
        
        # Create application support directory
        os.makedirs(os.path.expanduser("~/Library/Application Support/DocWhisperer"), exist_ok=True)
        
        # Initialize QApplication
        self.app = QApplication(sys.argv)
        
        # Show splash screen
        splash_pix = QPixmap('splash.png')  # Path to your splash image
        self.splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
        self.splash.show()
        
        # Initialize RAG system in a separate thread
        self.thread = QThread()
        self.worker = RAGSystemWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_rag_system_initialized)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        
    def on_rag_system_initialized(self, rag_system):
        self.rag_system = rag_system
        self.query_window = QueryWindow(self.rag_system)
        
        # Set up menu items
        self.menu.clear()
        self.menu = [
            rumps.MenuItem("Ask Question", callback=self.show_query_window),
            rumps.MenuItem("Add Documents", callback=self.add_documents),
            # rumps.MenuItem("List Documents", callback=self.list_documents),
            None,  # Separator
            rumps.MenuItem("About", callback=self.show_about),
            rumps.MenuItem("Quit", callback=self.quit_app)  # New Quit menu item
        ]
        
        # Close splash screen
        self.splash.close()
        
    @rumps.clicked("Ask Question")
    def show_query_window(self, _):
        self.query_window.show()
        
    @rumps.clicked("Add Documents")
    def add_documents(self, _):
        # Create a file dialog
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
                rumps.notification(
                    title="DocWhisperer",
                    subtitle="Documents Added",
                    message=f"Successfully processed {len(documents)} documents"
                )
        except Exception as e:
            rumps.notification(
                title="DocWhisperer",
                subtitle="Error",
                message=f"Error processing documents: {str(e)}"
            )

    @rumps.clicked("List Documents")
    def list_documents(self, _):
        try:
            num_documents = self.rag_system.get_document_count()  # Assuming this method exists
            rumps.notification(
                title="DocWhisperer",
                subtitle="Documents Available",
                message=f"There are {num_documents} documents available."
            )
        except Exception as e:
            rumps.notification(
                title="DocWhisperer",
                subtitle="Error",
                message=f"Error retrieving document count: {str(e)}"
            )
            
    @rumps.clicked("About")
    def show_about(self, _):
        rumps.alert(
            title="About DocWhisperer",
            icon_path="icon.png",
            message="DocWhisperer is an intelligent document assistant that helps you interact with your PDF documents using advanced AI technology."
        )

    def quit_app(self, _):
        rumps.quit_application()

class RAGSystemWorker(QObject):
    finished = pyqtSignal(object)

    def run(self):
        rag_system = RAGSystem(db_path=os.path.expanduser("~/Library/Application Support/DocWhisperer/rag_cache.db"))
        self.finished.emit(rag_system)

def main():
    DocWhispererApp().run()

if __name__ == "__main__":
    main()
