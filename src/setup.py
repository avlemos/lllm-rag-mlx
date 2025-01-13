from setuptools import setup

APP = ['docwhisperar.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': [],  # Add any required packages here
    'plist': {
        'CFBundleName': 'YourAppName',
        'CFBundleDisplayName': 'Your App Display Name',
        'CFBundleGetInfoString': "Your App Info",
        'CFBundleIdentifier': "com.yourcompany.yourapp",
        'CFBundleVersion': "0.1.0",
        'CFBundleShortVersionString': "0.1.0",
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
