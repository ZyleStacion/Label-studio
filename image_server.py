"""
Simple CORS-enabled HTTP server for serving page images to Label Studio.
Serves files from data/images/ on port 9090.

Usage:
    python image_server.py
"""

import http.server
import os
from pathlib import Path

PORT = 9090
SERVE_DIR = Path(__file__).parent / "data" / "images"


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress per-request logs


if __name__ == "__main__":
    os.chdir(SERVE_DIR)
    with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), CORSHandler) as httpd:
        print(f"Image server running at http://localhost:{PORT}/")
        print(f"Serving: {SERVE_DIR}")
        httpd.serve_forever()
