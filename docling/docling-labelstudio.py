# Install dependencies
# %pip install docling
# %pip install --upgrade label-studio-sdk

from __future__ import annotations
from label_studio_sdk.client import LabelStudio

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

import time
from PIL import Image

LABEL_STUDIO = "http://localhost:8080"
LS_API_KEY = "a81a470adcac997a1fc177fe9d09aec21a84e48f"
IMAGE_SERVER = "http://localhost:9090"
PDF_DIR = "data/pdfs"

# Helper Functions

def get_page_count(pdf_path) -> int:
    """
    Counts the pages of a given PDF using the PyPDFium backend.

    Args:
        pdf_path (str): The source PDF

    Returns:
        int: The total number of pages in the PDF
    """
    backend = 
    return len(backend)

def convert_bbox_to_ls(bbox, page_width, page_height): 
    """
    BBoxes in SmolDocling are given in LTRB format. We need to convert to the format Label Studio expects:
    the top left coordinate as a percentage of the total image, and the width and height as percents. 

    Args: 
        bbox: the bbox dictionary from SmolDocling's response object
        width: the width of the image in pixels 
        height: the height of the image in pixels 
        label: the label assigned by Docling to the response object

    Returns: a dictionary containing all the information for the value field in Label Studio for Rectangle Labels.
    """

    l = float(bbox["l"])
    t = float(bbox["t"])
    r = float(bbox["r"])
    b = float(bbox["b"])
    origin = str(bbox.get("coord_origin", "TOPLEFT"))

    x1 = min(l, r)
    x2 = max(l, r)

    if origin.endswith("BOTTOMLEFT"):
        y1 = page_height - max(t, b)
        y2 = page_height - min(t, b)
    else:
        y1 = min(t, b)
        y2 = max(t, b)

    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    if page_width <= 0 or page_height <= 0 or w <= 0 or h <= 0:
        return None

    return {
        "x": (x1 / page_width) * 100.0,
        "y": (y1 / page_height) * 100.0,
        "width": (w / page_width) * 100.0,
        "height": (h / page_height) * 100.0,
        "rotation": 0
    }

def map_label(item):
    """
    Docling's processing background may use a different labeling structure than what we expect. Therefore, we standardise each item's given label and convert it to one our system expects.

    Args:
        item: A result object from Docling's predictions

    Returns:
        The matching label for it, if one hasn't been set it defaults to unspecified
    """
    raw = str(item.get("label", "unspecified"))

    label_map = {
        "caption": "caption",
        "checkbox_unselected": "form",
        "checkbox_selected": "form",
        "document_index": "H1",
        "footnote": "footnote",
        "formula": "formula",
        "list": "list",
        "list_item": "list",
        "page_footer": "text",
        "page_header": "text",
        "picture": "picture",
        "section_header": "section_header",
        "table": "table",
        "title": "title",
        "text": "text",
        "unspecified": "unspecified",
    }

    return label_map.get(raw, "unspecified")

def do_ocr(source_file):
    """
    This is where the Docling pipeline exists. We run a timer, and use custom settings to process each file as a PDF, with the PyPDFium2 backend - which runs much faster than procedural image OCR. Next, the annotations are exported into JSON and converted into predictions for Label Studio. 

    Args:
        source_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    start_time = time.time()
    predictions = []

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=False)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    source_path = Path(source_file)
    print(f"processing {source_path}")

    result = converter.convert(source_path)
    doc = result.document.export_to_dict()
    pages = doc.get("pages", {})

    for collection_name in ("texts", "pictures", "tables"):
        for item in doc.get(collection_name, []):
            mapped_label = map_label(item)
            text_value = item.get("text", "")

            for prov in item.get("prov", []):
                page_no = prov.get("page_no")
                bbox = prov.get("bbox")
                if page_no is None or bbox is None:
                    continue

                page_meta = pages.get(str(page_no)) or pages.get(page_no)
                if not page_meta:
                    continue

                size = page_meta.get("size", {})
                page_width = float(size.get("width", 0))
                page_height = float(size.get("height", 0))

                bbox_ls = convert_bbox_to_ls(bbox, page_width, page_height)
                if not bbox_ls:
                    continue

                # Label Studio multi-page indexing is 0-based.
                item_index = max(int(page_no) - 1, 0)
                predictions.append({
                    "item_index": item_index,
                    "page_no": int(page_no),
                    "label": mapped_label,
                    "text_value": text_value,
                    "bbox_value": bbox_ls,
                    "original_width": page_width,
                    "original_height": page_height,
                })

    end_time = time.time() - start_time
    print(f"Done. Processed in {end_time}")
    return predictions

# Connect to label studio
ls = LabelStudio(
    base_url = LABEL_STUDIO,
    api_key = LS_API_KEY,
)

# Labeling Config for OCR using Multi-page document annotation

labeling_config = """
<View style="display:flex;align-items:start;gap:8px;flex-direction:row">
  <Image name="pdf" valueList="$pages" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="layout_label" toName="pdf" showInline="false">
  
    <Label value="H1" background="#2ca02c"/>
    <Label value="H2" background="#98df8a"/>
    <Label value="H3" background="#ff7f0e"/>
    <Label value="H4" background="#FFA39E"/>
    <Label value="H5" background="#fccfcc"/>

    <Label value="caption" background="#FFC069"/>
    <Label value="footnote" background="#1f77b4"/>
    <Label value="form" background="#bcbd22"/>
    <Label value="formula" background="#f9c1be"/>
    <Label value="list" background="#c49c94"/>
    <Label value="picture" background="#ff9896"/>
    <Label value="section_header" background="#393b79"/>
    <Label value="table" background="#D94545"/>
    <Label value="title" background="#940505"/>
    <Label value="text" background="#cccccc"/>
    <Label value="unspecified" background="#000000"/>

    </RectangleLabels>
  </View>

"""

# Create Label Studio Project
project = ls.projects.create(
    title="Docling Testing",
    description="Predictions using the Docling model",
    label_config=labeling_config
)

## Set up a task for each PDF in ../data/pdfs

pdfs = list(Path(PDF_DIR).glob("*.pdf"))
pdf_strings = [p.stem for p in pdfs]

print(pdf_strings)

for pdf in pdfs:
    pdf_title = pdf.stem
    pdf_url = f"{IMAGE_SERVER}/{pdf_title}/"
    pdf_len = get_page_count(pdf_path=pdf)

    task = {
        "pages": [f"{pdf}page_{page_num:04d}.png" for page_num in range(1, 500)]
    }

base_url = "http://localhost:9090/VLRC_Medicinal_Cannabis_Report_web/"

# TODO: Use a dynamic range
sample_task = {
    "pages": [f"{base_url}page_{page_num:04d}.png" for page_num in range(1, 500)]
}

# Upload task to Label Studio
ls.tasks.create(
    project=project.id,
    data=sample_task,
)

# ## Create Docling Predictions and Upload to Label Studio

# Get project information for uploading predictions 
upload_project = ls.projects.get(id=project.id)
li = upload_project.get_label_interface()

# Run predictions on the source file
task_source_file = "inputs/VLRC_Medicinal_Cannabis_Report_web.pdf"
task_predictions = do_ocr(task_source_file)
print("OCR Completed")

# Get the task in the projects, then upload their predictions
for task in ls.tasks.list(project=upload_project.id):
    task_id = int(task.id)
    print(f"processing task {task_id}")

    results = []
    for i, p in enumerate(task_predictions):
        value = dict(p["bbox_value"])
        value["rectanglelabels"] = [p["label"]]
        results.append({
            "id": f"region{i}",
            "from_name": "layout_label",
            "to_name": "pdf",
            "original_width": p["original_width"],
            "original_height": p["original_height"],
            "type": "rectanglelabels",
            "value": value,
            "item_index": p["item_index"],
        })

    ls.predictions.create(task=task_id, result=results, model_version="Docling-PDF-OCR")
    print(f"prediction for task {task_id} uploaded")

# ## Print Sample Predictions

import json

# for every task in the project, get its Docling prediction, and upload them
for task in ls.tasks.list(project=project.id):
    task_id = int(task.id)
    print(f"processing task {task_id}")

    task_source_file = "inputs/VLRC_Medicinal_Cannabis_Report_web.pdf"

    task_predictions = do_ocr(task_source_file)
    print("OCR Completed")

    test_results = []
    for i, p in enumerate(task_predictions):
        value = dict(p["bbox_value"])
        value["rectanglelabels"] = [p["label"]]
        test_results.append({
            "id": f"region{i}",
            "from_name": "layout_label",
            "to_name": "pdf",
            "type": "rectanglelabels",
            "value": value,
            "item_index": p["item_index"],
        })

    print(test_results)



