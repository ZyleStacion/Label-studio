#!/bin/bash
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=/Users/yethuaung/Label-studio/data/images
conda run -n labelstudio label-studio start --data-dir /Users/yethuaung/Label-studio/data
