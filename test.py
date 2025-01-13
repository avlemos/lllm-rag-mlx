import os
import time



def monitor_download(cache_path, interval=1):
    while True:
        if os.path.exists(cache_path):
            size = os.path.getsize(cache_path)
            print(f"Downloaded: {size / 1e6:.2f} MB")
        else:
            print("Waiting for download to start...")
        time.sleep(interval)

# Monitor the model directory
monitor_download(os.path.join(cache_dir, blobs))
