# Start Label Studio on Windows
# Run from the Label-studio directory in an activated conda or venv environment

$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT = "$PSScriptRoot\data\images"

label-studio start --data-dir "$PSScriptRoot\data"
