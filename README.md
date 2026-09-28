# YOLO Dataset Collector

A generic local workflow for collecting images, reviewing search results, generating assisted bounding-box annotations, building a cumulative multi-class Ultralytics YOLO dataset, training a YOLO model, and exporting the trained model to ONNX.

The workflow consists of three main programs:

1. `app.py` — image collection and search-backend diagnostics
2. `annotate.py` — Qwen3-VL assisted annotation and cumulative YOLO dataset export
3. `train_yolo_models.py` — YOLO training and ONNX export

The source image collections are kept separate from the final training dataset.

---

## 1. Requirements

Recommended environment:

- Windows 10/11
- Python 3.11
- Ollama
- a local Qwen3-VL vision model
- Streamlit
- `streamlit-drawable-canvas`
- Ultralytics YOLO
- NVIDIA GPU recommended for training, but not required for dataset collection/annotation

Check available Ollama models:

```powershell
ollama list
```

Test the configured vision model:

```powershell
ollama run <QWEN_MODEL_NAME>
```

The default local Ollama API address is typically:

```text
http://localhost:11434
```

---

## 2. Project structure

A typical project can look like:

```text
project_root/
├── app.py
├── annotate.py
├── train_yolo_models.py
├── requirements.txt
├── README.md
├── .venv/
└── datasets/
    ├── <COLLECTED_SOURCE_A>/
    │   ├── accepted/
    │   ├── review/
    │   ├── rejected/
    │   └── annotations_qwen/
    ├── <COLLECTED_SOURCE_B>/
    │   └── ...
    └── <FINAL_DATASET_NAME>/
        ├── dataset.yaml
        ├── classes.json
        ├── README.md
        ├── manifests/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

Collected source images and the final YOLO dataset are separate. Exporting annotations does not modify the original source images.

---

## 3. Create the Python environment

From the project root:

```powershell
py -3.11 -m venv .venv
```

Install project requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If required separately:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit-drawable-canvas
.\.venv\Scripts\python.exe -m pip install ultralytics
```

Using `.\.venv\Scripts\python.exe` directly avoids the need to activate the virtual environment and avoids PowerShell script-execution-policy issues with `Activate.ps1`.

Bypass execution policy:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 4. Start the image collector

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit normally opens a local browser address such as:

```text
http://localhost:8501
```

Use the GUI to define the target object/category and the available search sources.

---

## 5. Search-backend diagnostics

Before a large collection, use:

```text
Test search backends
```

For each enabled backend, diagnostics can include:

- HTTP status
- final request URL
- content type
- API result/page count
- parsed image count
- response excerpt
- rate-limit headers when available
- elapsed time
- sample result titles
- actual exception/error

This is intended to expose authentication errors, API limits, HTTP 403/429 responses, malformed responses, and other backend failures instead of silently returning zero results.

Recommended workflow:

1. Enter the target object/model.
2. Enter a useful physical/category description.
3. Enable the desired backends.
4. Run **Test search backends**.
5. Inspect the diagnostics.
6. Run the search preview.
7. Verify that the results represent the intended target.
8. Start the full collection.

---

## 6. Collector output

A source collection can contain:

```text
<COLLECTED_SOURCE>/
├── accepted/
├── review/
└── rejected/
```

`accepted` contains candidates intended for annotation.

`review` can contain uncertain candidates.

`rejected` contains images excluded during collection.

This source collection is not yet the final YOLO training dataset.

---

## 7. Start the annotation application

```powershell
.\.venv\Scripts\python.exe -m streamlit run annotate.py
```

If another Streamlit application is already using port 8501, Streamlit may choose another available local port.

---

## 8. Configure annotation

In the GUI configure values such as:

```text
Ollama URL:
http://localhost:11434

Vision model:
<QWEN_MODEL_NAME>

Collector dataset folder:
<PATH_TO_SOURCE_COLLECTION>

Source images:
accepted

YOLO class name:
<CLASS_NAME>

Project root:
<PROJECT_ROOT>

Dataset name:
<FINAL_DATASET_NAME>
```

The final dataset is created under:

```text
<PROJECT_ROOT>\datasets\<FINAL_DATASET_NAME>\
```

The same final dataset name should be reused when multiple classes are intended to be trained by one multi-class model.

---

## 9. Qwen-assisted annotation

Run:

```text
Auto-annotate with Qwen
```

The vision model attempts to generate bounding boxes for visible instances of the requested class.

Annotation JSON is retained under the source collection so annotation work can survive application restarts.

Existing Qwen annotation JSON can therefore act as persistent annotation state.

To intentionally regenerate cached annotations, use the GUI option for forced re-annotation or re-run Qwen for an individual image.

---

## 10. Review and edit bounding boxes

Qwen-generated boxes are proposals and should be reviewed.

The annotation canvas supports:

- reviewing automatically generated boxes
- moving existing boxes
- resizing existing boxes
- manually drawing a missing box
- saving edits

Manual drawing is primarily a fallback when the automatic annotation misses an object or produces an unusable box.

---

## 11. Image decisions

### Positive

Use the positive decision when the target class is present and its bounding boxes are correct.

A positive image requires at least one bounding box.

### Negative

Use the negative decision when an image should be retained for training but contains no object that should be annotated for the current/final dataset.

Negative images are exported with an empty YOLO `.txt` label file.

They can help reduce false-positive detections.

When building a multi-class dataset, an image containing another class that belongs to the final dataset should not be treated as a completely negative image merely because the currently reviewed class is absent. All relevant target classes visible in a final training image should ultimately be annotated correctly.

### Reject

Use reject when the image should not be included in the final dataset.

Rejected images are not exported.

---

## 12. Navigation

The annotation GUI provides navigation such as:

```text
Previous
Next
Positive & Next
Negative & Next
Reject & Next
```

Annotation state is saved before the corresponding decision/navigation action.

---

## 13. Final YOLO dataset

The exporter builds one cumulative multi-class dataset:

```text
<FINAL_DATASET_NAME>/
├── dataset.yaml
├── classes.json
├── README.md
├── manifests/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Every newly exported class is appended to the same class registry.

---

## 14. Persistent class IDs

Class IDs are stored in `classes.json`.

Example:

```yaml
names:
  0: "<CLASS_A>"
  1: "<CLASS_B>"
  2: "<CLASS_C>"
```

A label for class ID 2 uses:

```text
2 <x_center> <y_center> <width> <height>
```

Do not reorder existing class IDs after labels have been exported. Doing so would change the meaning of existing annotation files.

---

## 15. YOLO label format

Each bounding box is stored as:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to the range 0–1.

A valid negative image has an image file and a corresponding empty `.txt` label file.

---

## 16. dataset.yaml

`dataset.yaml` is generated and updated automatically.

Its structure is similar to:

```yaml
path: <ABSOLUTE_DATASET_PATH>
train: train/images
val: val/images
test: test/images

names:
  0: "<CLASS_A>"
  1: "<CLASS_B>"
  2: "<CLASS_C>"
```

This YAML file is passed directly to Ultralytics during training.

---

## 17. Automatically generated dataset README

Each export also regenerates:

```text
datasets\<FINAL_DATASET_NAME>\README.md
```

This is separate from this project README.

The dataset README is generated from the actual exported files and contains:

- dataset structure
- current class IDs and names
- number of images containing each class
- number of object instances for each class
- train/validation/test image counts
- annotated-object counts
- negative-image counts
- annotation-method description
- YOLO label format
- basic training commands
- ONNX export command

Because it is generated from the exported labels, the statistics remain synchronized with the current dataset.

---

## 18. Export another class

To add another class:

1. Collect its source images with `app.py`.
2. Open the source collection in `annotate.py`.
3. Set the desired YOLO class name.
4. Keep the same project root.
5. Keep the same final dataset name.
6. Run automatic annotation.
7. Review/correct the boxes.
8. Make positive/negative/reject decisions.
9. Export/update the dataset.

The new class is appended to `classes.json` and `dataset.yaml`.

Existing class IDs remain unchanged.

---

## 19. Re-export a class

Re-exporting a previously registered class updates that class's exported files without intentionally deleting the other classes.

Exported files use a class-based prefix to avoid filename collisions between source collections.

---

## 20. Manifests

Per-class export metadata is stored under:

```text
manifests/
```

The manifests record information such as:

- class name
- class ID
- positive-image count
- negative-image count
- split counts
- dataset path

---

## 21. Train / validation / test split

Configure the split in the annotation GUI.

A common starting point is:

```text
Train:       80 %
Validation:  10 %
Test:        10 %
```

The exported layout is:

```text
train/images
train/labels
val/images
val/labels
test/images
test/labels
```

Avoid placing identical or near-identical images in different splits. Data leakage can produce misleading validation/test results.

The test set should remain independent and should not be used for routine model tuning.

---

## 22. Train YOLO

Training uses:

```text
train_yolo_models.py
```

From the project root:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml
```

The default base model is:

```text
yolo26n.pt
```

Default settings are:

```text
epochs: 100
imgsz: 640
batch: 16
```

---

## 23. Select another base model

Specify another compatible base model with:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml --model <BASE_MODEL>.pt
```

Using the same dataset and splits allows different model architectures/sizes to be compared under similar conditions.

---

## 24. NVIDIA GPU training

To explicitly use the first CUDA GPU:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml --device 0
```

With explicit settings:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml --model <BASE_MODEL>.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

If `--device` is omitted, Ultralytics is allowed to select an available device.

---

## 25. Apple Silicon training

On a compatible Apple Silicon Mac:

```bash
python train_yolo_models.py train datasets/<FINAL_DATASET_NAME>/dataset.yaml --device mps
```

---

## 26. Training output

By default, training runs are written under:

```text
models/
└── <RUN_NAME>/
    └── weights/
        ├── best.pt
        ├── last.pt
        └── best.onnx
```

`best.pt` is the primary trained PyTorch checkpoint selected from validation performance.

`last.pt` is the last training checkpoint.

`best.onnx` is the deployment export generated from `best.pt`.

---

## 27. Resume an interrupted run

Resume the same interrupted training run with:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py resume models\<RUN_NAME>\weights\last.pt
```

Resume is intended for continuing the same training run. It is not the default workflow for introducing a substantially changed class set.

---

## 28. Automatic ONNX export

After successful training, `train_yolo_models.py` automatically exports `best.pt` to ONNX unless ONNX export is disabled.

Typical output:

```text
weights/
├── best.pt
├── last.pt
└── best.onnx
```

The default deployment-oriented export uses batch size 1.

---

## 29. Export an existing model to ONNX

A trained `.pt` checkpoint can be exported without retraining:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py export models\<RUN_NAME>\weights\best.pt
```

The resulting ONNX file is stored alongside the checkpoint:

```text
best.pt
best.onnx
```

---

## 30. Disable automatic ONNX export

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml --no-onnx
```

---

## 31. ONNX deployment

The generated `best.onnx` can be used by an ONNX-compatible inference application, including mobile applications.

The inference implementation must match the exported model's:

- input image dimensions
- resizing/letterboxing
- color ordering
- normalization
- tensor layout
- output tensor interpretation
- confidence filtering
- bounding-box coordinate conversion
- NMS/post-processing requirements, when applicable

Validate deployment inference against known test images before relying on the mobile/runtime implementation.

---

## 32. Complete workflow

```text
Image search sources
        │
        ▼
      app.py
        │
        ▼
source collection
accepted / review / rejected
        │
        ▼
    annotate.py
        │
        ├── Qwen3-VL bbox proposals
        ├── manual review/correction
        ├── positive decisions
        ├── negative decisions
        └── rejected images
        │
        ▼
datasets/<FINAL_DATASET_NAME>/
        │
        ├── dataset.yaml
        ├── classes.json
        ├── README.md
        ├── manifests/
        ├── train/
        ├── val/
        └── test/
        │
        ▼
train_yolo_models.py
        │
        ▼
base YOLO model + dataset
        │
        ▼
      best.pt
        │
        ▼
     best.onnx
        │
        ▼
deployment application
```

---

## 33. Quick command reference

Create the environment:

```powershell
py -3.11 -m venv .venv
```

Install requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the collector:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Start the annotator:

```powershell
.\.venv\Scripts\python.exe -m streamlit run annotate.py
```

Train using the default base model:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml --device 0
```

Train using another base model:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py train datasets\<FINAL_DATASET_NAME>\dataset.yaml --model <BASE_MODEL>.pt --device 0
```

Resume an interrupted run:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py resume models\<RUN_NAME>\weights\last.pt
```

Export an existing model to ONNX:

```powershell
.\.venv\Scripts\python.exe train_yolo_models.py export models\<RUN_NAME>\weights\best.pt
```

---

## 34. Important rules

- Review automatically generated annotations before final training.
- Keep existing class IDs stable.
- Do not mix identical or near-identical images across train/validation/test splits.
- Use negative images intentionally to reduce false positives.
- Do not label an image as globally negative if it contains an object class that should be annotated in the final multi-class dataset.
- Keep source image collections so the dataset can be rebuilt if necessary.
- Back up `dataset.yaml`, `classes.json`, annotation JSON, manifests, exported labels, and trained model weights.