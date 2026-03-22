# auv-perception-pipeline

A multi-rate sensor fusion pipeline that simulates the core perception challenge on an Autonomous Underwater Vehicle (AUV).

---

## The Problem

An AUV operating at depth cannot use GPS or LiDAR. Its primary sensors are a camera and a forward-scanning sonar — and these run at different rates. The camera pushes frames at 30 Hz. The sonar pings at 10 Hz. The inference model runs at 8–10 Hz. These three rates are never the same.

A naive implementation — one thread, `cv2.read()`, sequential processing — blocks on every frame request. On a vehicle operating in real-time, one slow frame stalls the entire perception stack. This project demonstrates the correct architecture: decouple producers from consumers, run inference at its own rate, and fuse sensor data at the point of consumption.

---

## Architecture

```
[GStreamer Camera Source] --30Hz--> [camera_buffer] ---> [Inference Node]
                                                               |
[Sonar Simulator] --------10Hz--> [sonar_buffer]  ----------> [Fusion]
                                                               |
                                                         Fused Output
                                              Class | Confidence | Distance (m)
```

**Components:**

- `camera_source.py` — GStreamer pipeline with `appsink`. Runs in native C-threads. Pushes raw `(640, 640, 3)` uint8 frames into `camera_buffer` at 30 Hz via a push-based callback. Chosen over `cv2.VideoCapture` because GStreamer's push model never blocks the main thread waiting for a frame.

- `frame_buffer.py` — Thread-safe decoupling layer built on `collections.deque(maxlen=1)`. Uses `threading.Event` for event-driven notification rather than polling. The `maxlen=1` constraint ensures inference always operates on the latest frame, never a stale one. `deque` was chosen over `queue.Queue` for its C-level implementation and non-blocking overwrite semantics.

- `sonar_simulator.py` — Simulates a 100-beam forward-scanning sonar (modelled after the Tritech Gemini class) operating at 10 Hz. Returns a 1D `float32` array of 100 range values spanning a 90-degree horizontal arc. Uses a cooperative thread with explicit `.join()` on shutdown for clean OS-level teardown.

- `inference_node.py` — YOLOv8n running via ONNX Runtime. Handles the full preprocessing chain: interleaved `(H, W, C)` → planar `(1, C, H, W)` transposition, `uint8` → `float32` normalisation, NMS post-processing via `cv2.dnn.NMSBoxes`. ONNX session is allocated in `__init__` (fail-fast boot: if weights are missing, the pipeline crashes before any hardware is locked).

- `main.py` — Orchestrator. Wires all components, runs the fusion loop, handles graceful shutdown on `KeyboardInterrupt`.

---

## Sensor Fusion

Fusion runs at camera rate (30 Hz) using a **Latest Available (Soft Sync)** policy.

The inference node returns normalised bounding box centroids (`x_center_norm`: 0.0 to 1.0). Each centroid is projected onto the sonar's 100-beam array using a linear mapping:

```python
sonar_index = min(int(x_center_norm * 100), 99)
distance_m = sonar_array[sonar_index]
```

This assumes a linear correspondence between image horizontal position and sonar bearing — valid for a forward-facing camera and sonar with matched fields of view.

**Why Soft Sync over Hard Sync:** Blocking the 30 Hz vision pipeline to wait for a 10 Hz sonar ping introduces artificial latency. The AUV would be running blind for up to 100 ms per cycle. With Soft Sync, inference never waits — it grabs the most recent sonar ping and fuses immediately.

---

## Sample Output

```
Fused Object -> Class: 5 | Conf: 0.90 | Distance: 0.35m
Fused Object -> Class: 2 | Conf: 0.85 | Distance: 0.16m
Fused Object -> Class: 5 | Conf: 0.92 | Distance: 0.88m
Fused Object -> Class: 2 | Conf: 0.82 | Distance: 0.51m
```

---

## Running

**Dependencies:**
```bash
sudo apt install -y python3-gst-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gir1.2-gst-plugins-base-1.0

pip install onnxruntime opencv-python numpy ultralytics
```

**Model:**
Download `yolov8n.onnx` from Ultralytics and place it in `model/yolov8n.onnx`

**Video:**
Download any test video that works for the COCO dataset and place in it `data/test_video.mp4`


**Run:**
```bash
python3 main.py
```

Shutdown cleanly with `Ctrl+C`. All threads join before exit.

---

## Key Design Decisions

| Decision | Rejected Alternative | Reason |
|---|---|---|
| GStreamer push model | `cv2.VideoCapture` pull loop | Pull blocks the thread on every frame request |
| `deque(maxlen=1)` | `queue.Queue(maxsize=1)` | deque overwrites without blocking; Queue raises `Full` exception |
| `threading.Event` | Polling loop | Event-driven — no wasted CPU cycles checking a flag |
| Separate `camera_buffer` and `sonar_buffer` | Single shared buffer | A shared buffer cannot safely hold a `(640,640,3)` image one tick and a `(100,)` float array the next without tensor shape crashes downstream |
| Soft Sync fusion | Hard Sync (wait for sonar) | Hard Sync introduces up to 100ms blind windows at 10 Hz sonar rate |
| Fail-fast ONNX init | Lazy model loading | AUV should crash on boot if weights are missing, not mid-mission |

---

## Limitations & Future Work

This pipeline is a simulation and demonstration of architecture, not a production AUV stack. Known limitations and intended extensions:

- **Sonar data is synthetic.** Real integration would use a driver publishing actual acoustic returns from hardware such as a Tritech Gemini or BlueView.
- **Fusion is geometrically naive.** A production system would apply proper camera-to-sonar extrinsic calibration, coordinate frame transforms, and timestamp-aligned interpolation rather than a linear index mapping.
- **No Kalman filtering.** Detection tracks are not maintained across frames. A production stack would run a multi-object tracker (e.g. SORT or ByteTrack) and fuse sonar ranges into track state.
- **No QoS / latency monitoring.** A production pipeline would instrument buffer depth, fusion Δt, and dropped frame counts continuously, with alerts when thresholds are breached.

---

## Why This Architecture Maps to AUV Perception

The rate mismatch between camera, sonar, and inference is not a simulation artifact — it is the fundamental constraint of any real AUV perception stack. Sonar pings are slow by physics (speed of sound in water ≈ 1500 m/s vs 3×10⁸ m/s for light). Inference is slow by compute budget. Camera is fast by design.

The correct response to this constraint is not to slow everything down to the slowest component. It is to decouple the components, let each run at its natural rate, and fuse at the point where the data meets. That is what this pipeline demonstrates.