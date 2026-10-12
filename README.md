---
title: Road Damage Reporter
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: agpl-3.0
---

# Road Damage Reporter

Mount your phone in the car like a dashcam. A YOLOv8n model runs **on the phone** on the live
camera feed, finds potholes and cracks, and reports each one with its GPS location.
An authority dashboard shows every damaged spot on a map, ranked by priority, plus the dashcams
that are driving right now.

**This is a demo, not a safety tool.**

## How it works
1. **Live dashcam (`/dashcam`).** The model (ONNX) runs in the phone browser with
   onnxruntime-web (WebGPU when available, otherwise WebAssembly). Only small reports
   (location, damage type, confidence, time) are sent, never video, so it uses very little data.
   - A damage is reported only if it is seen in 2 of the last 3 frames (cuts one-frame false alarms).
   - GPS must be accurate to within 60 m.
   - One drive reports the same damage type once per 20 m, so a pothole that stays in view for
     30 frames is reported once. The server checks this too.
   - Reports wait in a queue on the phone when there is no signal and are sent later.
   - The phone sends its position every 10 s, so the dashboard shows live dashcams.
2. **Report with photos.** Upload photos instead. The location comes from the photo's GPS,
   or the phone's location. The same photo is never counted twice.
3. **Road type.** The nearest road and its class come from OpenStreetMap (Nominatim):
   main (motorway, trunk, primary), medium (secondary, tertiary) or small (everything else).
4. **Authority dashboard.** Reports within 25 m are grouped into one spot.
   *Times seen* = number of separate drives or photos that saw the spot.
   Priority score = road importance (3/2/1) x worst damage (pothole 3, alligator 2, other cracks 1) x times seen.
   High is 9 or more, Medium is 4 to 8, Low is under 4. The list can be downloaded as CSV.
5. **Storage.** Reports are saved to a private Hugging Face dataset, so they survive restarts.

## Data
RDD2022, India part only (CC BY 4.0). 7,706 labelled images, split 85/15 at random
(seed 42): 6,550 train and 1,156 validation. The official test labels are not public,
so all numbers below are on my own validation split.

Classes: longitudinal crack (D00), transverse crack (D10), alligator crack (D20), pothole (D40).

## Results (validation, 1,156 images)

| Model | mAP50 | mAP50-95 |
|---|---|---|
| YOLOv8n @640 (used here) | 0.363 | 0.157 |
| YOLOv8s @640 | 0.374 | 0.155 |
| YOLOv8n, India + 5 other countries @736 | 0.355 | 0.164 |

Per class (used model, AP50): alligator 0.60, pothole 0.42, longitudinal 0.30, transverse 0.13.
Transverse has only 9 validation examples, so that number is noisy.

## What I tried
A bigger model (YOLOv8s) and extra training data from Japan, Norway, Czech Republic,
USA and China (motorbike) did not give a clear gain on Indian roads, so I kept the small, fast model.

## Limitations
- Misses fine cracks. False alarms on dust, road patches, shadows and markings.
  Repeated reports of the same spot help filter these out.
- Phone GPS is accurate to about 5 to 15 m, so a spot can be off by that much.
- Speed depends on the phone: roughly 2 to 10 frames per second.
- Phone location in photo mode is where the phone is when you press the button.
- Anyone with the link can report, so a real deployment would need login and moderation.

## Setup (for the Space owner)
Upload `best.onnx` with the other files (running `python app.py` once on a laptop creates it
next to `best.pt`). If it is missing, the Space tries to create it at start-up.
Add a secret named `HF_TOKEN` (a Hugging Face token with write access) in the Space settings.
Reports are then stored in the private dataset `<your-username>/<space-name>-reports`.
Without the secret the app still works, but reports are lost when the Space restarts.

## Credits and licence
- Dataset: RDD2022, Arya et al., [arXiv 2209.08538](https://arxiv.org/abs/2209.08538),
  DOI 10.6084/m9.figshare.21431547.v1, CC BY 4.0.
- Road data (c) OpenStreetMap contributors, via Nominatim.
- Ultralytics YOLOv8 is AGPL-3.0, so this project is released under AGPL-3.0 too.
