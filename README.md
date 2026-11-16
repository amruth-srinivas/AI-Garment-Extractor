# GarmentAI

GarmentAI turns a fashion photograph of a person into an isolated clothing image. It separates the garment from the body. It does not treat the task as “remove the background and keep the person.”

What the pipeline keeps is the original cloth: colour, print, stitching, neckline, sleeves, and length. Where a limb actually covers the garment, an inpainting model can estimate the hidden cloth. That estimate is not the original covered fabric, and the result says so.

## Verified on this machine

- Windows 11, Python 3.11.8 virtualenv, Node 22.15.1
- NVIDIA T1000, 4 GB VRAM, driver CUDA 12.4, PyTorch 2.6.0+cu124
- SegFormer clothes parsing, GrabCut edge refinement, LaMa hole filling
- A real outfit photo produced an isolated yellow sweatpant image (face, hoodie, midriff, boots, and court removed) and a separate cropped-hoodie image
- 18 automated tests passing

SAM 2.1 hiera-tiny is installed and wired in, and it stays off unless you enable it. On a 4 GB GPU it runs only after the parser is unloaded. Diffusion inpainting is not bundled: a useful Stable Diffusion inpaint does not fit beside the parser in 4 GB.

## 1. Create the backend environment

Use Python 3.11. The default `python` on this PC is 3.13, and the vision wheels used here are installed into a 3.11 virtualenv.

```bat
cd /d "D:\Amruth\AI REMOVER\backend"
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install the CUDA build of PyTorch before `requirements.txt`. Installing Torch from PyPI afterwards can replace it with a CPU build.

If this machine has no NVIDIA GPU, skip the CUDA index and install the CPU build instead:

```bat
.venv\Scripts\python.exe -m pip install torch torchvision
```

The API detects the GPU and falls back to CPU when CUDA is unavailable or runs out of memory.

## 2. Download the model weights

```bat
cd /d "D:\Amruth\AI REMOVER\backend"
.venv\Scripts\python.exe scripts\download_models.py
.venv\Scripts\python.exe scripts\check_environment.py
```

Weights land in `backend\weights` and `backend\.cache\huggingface`. The first two files are about 200 MB (LaMa) and 16 MB (MediaPipe). SegFormer and SAM 2 are Hugging Face snapshots.

| Model | Role | Licence |
| --- | --- | --- |
| `mattmdjaga/segformer_b2_clothes` | Labels upper clothes, dress, skirt, pants, hair, face, arms, and legs | Non-commercial research and evaluation only. It is a fine-tune of NVIDIA SegFormer `mit-b2`. Do not ship it in a commercial product. |
| MediaPipe selfie multiclass 256 | Labels background, hair, body skin, face skin, clothes, accessories | Apache-2.0 |
| `facebook/sam2.1-hiera-tiny` | Optional edge refinement. It does not decide what is clothing. | Apache-2.0 |
| LaMa `big-lama.pt` | Estimates cloth hidden by the body, only inside the occlusion mask | Apache-2.0 code and published weights. The fill is an estimate, not recovered cloth. Places training data may have its own terms. |
| rembg / U2-Net | Optional leftover-background trim after the person is already removed | rembg and U2-Net are MIT. U2-Net downloads from GitHub on first use (`%USERPROFILE%\.rembg`). If GitHub cannot be reached, this step is skipped and the clothing mask is kept. |

Parser choice in the app:

- **Auto** tries SegFormer, then MediaPipe.
- **SegFormer** is the more detailed parser and carries the NVIDIA non-commercial licence.
- **MediaPipe** is the Apache-2.0 parser. It has one clothes class. A top and bottom are split only when a skin gap makes that reliable. Dress, saree, kurti, lehenga, shirt, jacket, and jeans are not separate trained classes. The product tag is a label you can change.

## 3. Run the tests

```bat
cd /d "D:\Amruth\AI REMOVER\backend"
.venv\Scripts\python.exe -m pytest tests -q
```

The tests cover upload validation, garment detection on a synthetic parse, body removal, occlusion holes, reconstruction that does not rewrite original cloth, transparent PNG export, and API errors when the model files are absent.

## 4. Start the application

Backend:

```bat
cd /d "D:\Amruth\AI REMOVER\backend"
set HF_HOME=D:\Amruth\AI REMOVER\backend\.cache\huggingface
set HF_HUB_DISABLE_TELEMETRY=1
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend, in a second command prompt:

```bat
cd /d "D:\Amruth\AI REMOVER\frontend"
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## 5. Extract one photograph from the command line

```bat
cd /d "D:\Amruth\AI REMOVER\backend"
.venv\Scripts\python.exe scripts\extract_image.py "D:\path\to\your-photo.jpg"
```

The script keeps the largest detected garment. Outputs:

- `backend\samples\output\<name>\garment.png` — transparent PNG
- `backend\samples\output\<name>\garment_white.jpg` — white background

Use your own photograph. A public-domain sample could not be fetched here because `commons.wikimedia.org` did not resolve. Do not commit client photos.

## How an extraction works

1. The upload is stored as a lossless PNG after EXIF rotation. Fabric is not resized for export.
2. SegFormer or MediaPipe labels cloth separately from hair, face, skin, arms, and legs.
3. If several garments are found, the job pauses and the UI shows each mask. You choose one or more, and a product tag (dress, shirt, t-shirt, top, trousers, jeans, skirt, jacket, coat, saree, kurti, lehenga, scarf, other).
4. GrabCut moves only a narrow band around the clothing mask. SAM 2, when enabled, is clipped to that same band and cannot delete the confident core.
5. Face, hair, skin, and limbs are removed. Small skin-coloured pieces that touch the silhouette, such as a fingertip the parser folded into a sleeve, are removed. A large skin-toned panel is left alone.
6. Reconstruction targets cloth a body part actually covers. A missing sleeve that hair covered is copied from the sleeve still visible. A limb lying on the cloth is filled from the surrounding fabric. An arm or hand that only rests beside the garment is left out, so the cloth does not grow a ribbon. Original garment pixels are copied back unchanged. LaMa remains the fallback when that copy cannot be made, and OpenCV Telea is used if LaMa cannot load. The fill is named as an estimate. Low-confidence masks skip automatic reconstruction.
7. Export writes a full-resolution transparent PNG and a white JPEG. The UI also has a before/after slider, a transparent or white preview, erase and restore brushes, and a control that regenerates the hidden-cloth estimate.

The progress screen shows the stage the worker is actually in. A skipped stage, such as reconstruction turned off, is marked skipped.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Python, GPU, and which weights are present |
| `POST` | `/api/uploads` | JPG, JPEG, PNG, or WEBP. Max 30 MB |
| `GET` | `/api/uploads/{id}/file?kind=preview\|original` | Preview or lossless original |
| `POST` | `/api/extractions` | Start detection. Returns a job |
| `GET` | `/api/extractions/{id}` | Poll real stage, message, and progress |
| `POST` | `/api/extractions/{id}/continue` | Extract the selected garment ids |
| `GET` | `/api/extractions/{id}/garments/{garment_id}/overlay` | Mask overlay |
| `GET` | `/api/results` | History and processed count |
| `GET` | `/api/results/{id}` | Result metadata, confidence, and notes |
| `PATCH` | `/api/results/{id}` | Set the product tag |
| `DELETE` | `/api/results/{id}` | Remove one extraction from the studio |
| `GET` | `/api/results/{id}/file?kind=transparent\|white\|mask\|thumb\|original` | Previews |
| `GET` | `/api/results/{id}/download?format=png` | Transparent PNG |
| `GET` | `/api/results/{id}/download?format=jpeg` | White JPEG |
| `POST` | `/api/results/{id}/mask?regenerate=false` | Upload a hand-edited mask |
| `POST` | `/api/results/{id}/reconstruct` | Regenerate hidden cloth |

A detection job stays in `awaiting_selection` until you continue it, including when only one garment was found, so the mask can be checked first. `selection=largest` is what the command-line script uses.

Low confidence (under 0.45) is returned on the result, reconstruction is skipped, and the mask editor stays available. A mask that covers less than 0.8% of the photo is rejected instead of exported.

## Project layout

```
backend/app/api/routes        upload, extraction, results, health
backend/app/services          parser, detector, segmentation, refinement,
                              human removal, reconstruction, export
backend/app/models            lazy loading, GPU/CPU fallback, downloads
frontend/src                  React, Vite, TypeScript, Tailwind
```

Each AI step is its own service class: `HumanParser`, `GarmentDetector`, `SegmentationService`, `MaskRefiner`, `HumanRemoval`, `GarmentReconstruction`, `BackgroundRemoval`, `ImageExporter`. Replacing a model means replacing that class, not the UI.

## Limits

- Saree, kurti, lehenga, jacket, coat, shirt, and jeans are product tags. The trained classes are upper clothes, dress, skirt, pants, and a single MediaPipe clothes class.
- Hidden plaid, stripes, or prints are continued from visible neighbours when the hole is filled. They are not recovered.
- Fingertips wrapped over a hood or cuff can leave a few pixels. Erase them in the mask editor.
- Masks for very large photos are estimated with the long side capped at 1280 pixels, then applied to the original pixels so the texture is not downscaled.
- Jobs are processed one at a time so the 4 GB GPU is not shared by two networks.
- Uploads, jobs, and results older than 48 hours are deleted on startup.
