"""
Creates a Label Studio project configured for LayoutLMv3 PDF annotation.
Run once after `label-studio start` is running.
"""

import os
import json
import requests

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")  # set after first login
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs/layoutlmv3_labeling_config.xml")


def create_project(api_key: str, label_config: str) -> dict:
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "title": "LayoutLMv3 PDF Annotation",
        "description": "Document layout annotation for LayoutLMv3 training",
        "label_config": label_config,
        "expert_instruction": (
            "Draw bounding boxes around each document element and assign the correct label. "
            "Use 'Title' for main document titles, 'Section-header' for subsection headings, "
            "'Text' for body paragraphs, 'Table' for tables, 'Figure' for images/charts, "
            "'Caption' for figure/table captions, 'List' for bullet/numbered lists."
        ),
    }
    resp = requests.post(f"{LABEL_STUDIO_URL}/api/projects/", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    if not API_KEY:
        print(
            "Set LABEL_STUDIO_API_KEY before running.\n"
            "Find it at: Account & Settings → Access Token (top-right menu in Label Studio)."
        )
        return

    with open(CONFIG_PATH) as f:
        label_config = f.read()

    project = create_project(API_KEY, label_config)
    print(f"Project created: id={project['id']}  title={project['title']}")
    print(f"Open: {LABEL_STUDIO_URL}/projects/{project['id']}/")


if __name__ == "__main__":
    main()
