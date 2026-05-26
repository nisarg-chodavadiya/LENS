<img width="1024" height="498" alt="image" src="https://github.com/user-attachments/assets/37e8019e-06d6-4916-afb9-adfd3efea09e" />


<!-- ────────  BADGE WALL  ──────── -->

[![Phase 1 · MVP](https://img.shields.io/badge/Phase_1-MVP_shipped-FF6B2C?style=for-the-badge&logo=rocket&logoColor=white)](#-roadmap)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%C2%B7%20v11%20%C2%B7%20v26-00C2B8?style=for-the-badge&logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/)

[![License: AGPL](https://img.shields.io/badge/License-AGPL-FFD700?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-FF69B4?style=flat-square&logo=github)](#-contributing)
[![Issues](https://img.shields.io/badge/Issues-open-22C55E?style=flat-square&logo=github)](../../issues)
[![Stars](https://img.shields.io/badge/⭐-leave_a_star-FFA500?style=flat-square)](../../stargazers)
[![Made with ❤️](https://img.shields.io/badge/made_with-❤️_%26_curiosity-E04A1F?style=flat-square)](#)

<br/>

### LENSv0.1.0 -- The Explainable AI (xAI) for Neural Networks.

### 🔍 _Feed YOLO weights + a single forward pass → get a paragraph back._
### 🧠 _No saliency maps. No GradCAM. Just **language**._

</div>

<br/>

---

## 🎯 What is LENS?
> ### Using billion-parameter LLMs to explain what million-parameter neural networks are actually doing.

**LENS** is a framework that turns the inside of a neural network into **natural language you can read**.

Computer vision models are powerful, but their decisions live in N-D activation tensors that humans can't parse. Existing tools — GradCAM, saliency maps, attention rollout — tell you _where_ the model looked, not _what it was thinking_. LENS asks a different question:

> **"What if the model could just _tell_ you why it predicted what it did?"**

We capture every layer's activation during a single forward pass, compress it into a structured JSON log, and hand both the log **and** the model weights to an LLM. The LLM grounds its explanation in the actual numbers — not vibes.

<div align="center">

| 🔬 **GradCAM says** | 💬 **LENS says** |
|:---|:---|
| _"heatmap over pixels (47, 92)"_ | _"The P4 head fired strongly on filters that learned edge-and-fur textures during training. The runner-up class (cat) lost by 2.3 logits because the body proportions matched the dog cluster more closely. Confidence: high."_ |

</div>

<br/>

---

## ✨ Why LENS?

<table>
<tr>
<td width="50%" valign="top">

### 🪄 **Grounded in real activations**
Every claim the LLM makes is backed by a number from the log. No hallucinated explanations.

### 🧩 **Architecture-tolerant**
Works across YOLOv8, v11, and v26 nano out of the box. Hooks adapt to whatever modules it finds.

### 📦 **Single-file logs**
The whole inference is one JSON. Reproducible, diffable, version-controllable.

</td>
<td width="50%" valign="top">

### ⚡ **Fast & cheap**
One forward pass + one LLM call. No retraining, no probing dataset, no GPU farm.

### 🔌 **Drop-in for any YOLO weights**
Bring your own `.pt` file. LENS handles the rest.

### 🛠 **Debug mode**
`--log-only` dumps the JSON without calling the LLM — perfect for inspecting the schema.

</td>
</tr>
</table>

<br/>

---

## 🚀 Quickstart

### 1️⃣ Install

```bash
pip install -r requirements.txt
```

### 2️⃣ Add your OpenAI key

Create a `.env` in the project root (it's gitignored):

```env
OPENAI_API_KEY=sk-...
```

<details>
<summary>🪟 <b>Windows PowerShell users — click here</b></summary>

```powershell
$env:OPENAI_API_KEY = "sk-..."          # current shell only
setx OPENAI_API_KEY "sk-..."            # persistent, requires new shell
```
</details>

### 3️⃣ Drop in weights + a test image

```
LENS/
├── 🏋  yolo26n.pt
├── 📸  samples/person.jpg
└── ▶  run.py
```

### 4️⃣ Run

```bash
python run.py --weights yolo26n.pt --image samples/person.jpg
```

**Output:** detection result + a 6-paragraph natural-language explanation, grounded in the actual inference log. ✨

<br/>

---

## 🎛 Modes

<table>
<tr>
<th>Mode</th>
<th>Command</th>
<th>When to use it</th>
</tr>
<tr>
<td>🟢 <b>Full</b></td>
<td><code>python run.py --weights w.pt --image i.jpg</code></td>
<td>Default — detection + LLM explanation</td>
</tr>
<tr>
<td>🔵 <b>Log only</b></td>
<td><code>... --log-only</code></td>
<td>Free; dumps JSON for debugging the logger</td>
</tr>
</table>
<br/>

---

## 🔬 What gets captured

Every forward pass, the `ActivationLogger` records:

- 📊 **Per-layer stats** — mean/max activation magnitude, peak channel, peak spatial location
- 🎯 **Raw Detect-head outputs** at all three FPN scales (P3 / P4 / P5)
- 🧮 **Pre-NMS candidate counts** at `conf > 0.25` and `conf > 0.5`
- 🏆 **Winning detection's class logits** vs. runner-up logits

`build_inference_log()` in [`lens/schema.py`](lens/schema.py) compresses this into the **7-block JSON structure**.

<br/>

---

## 🗺 Pipeline at a glance

```
   ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐    ┌─────────────┐
   │  YOLO weights│───▶│  ActivationLogger │───▶│ inference_   │───▶│   Claude/   │
   │   + image    │    │  (forward hooks)  │    │  log.json    │    │   GPT-4o    │
   └──────────────┘    └───────────────────┘    └──────────────┘    └──────┬──────┘
                                                                            │
                                                                            ▼
                                                            ┌─────────────────────────────┐
                                                            │ 📝 6-paragraph explanation  │
                                                            │     grounded in numbers     │
                                                            └─────────────────────────────┘
```

<br/>

---

## 📁 Project structure

| File | Role |
|:---|:---|
| 🪝 [`lens/logger.py`](lens/logger.py) | PyTorch forward-hooks; architecture-tolerant (YOLOv8 / v11 / v26 nano) |
| 📋 [`lens/schema.py`](lens/schema.py) | Builds the `inference_log.json` structure from raw hook data |
| 🧬 [`lens/filter_patterns.py`](lens/filter_patterns.py) | MVP heuristic vocabulary |
| 💬 [`lens/explainer.py`](lens/explainer.py) | LLM call (default `gpt-4o`) with stable system prompt + architecture summary + inference log |
| ▶ [`run.py`](run.py) | CLI entry point |

<br/>

---

## 🖼 Output Example

Here's a real end-to-end run on `bike.jpg` using **YOLOv26n** weights.

### 📸 Annotated Detection

> The model drew a bounding box around the motorcycle with **0.92 confidence**.

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/224ca356-eba3-42ac-ac2d-06fd9b400b8a" />

<br/>

### 📄 Inference Log (`inference_log.json`)

```json
{
  "input": {
    "image_path": "images2\\bike.jpg",
    "image_size": [1200, 800],
    "image_region_of_detection": "middle-center"
  },
  "prediction": {
    "class": "motorcycle",
    "confidence": 0.924,
    "bbox": [177.7, 74.8, 1025.2, 730.2]
  },
  "top_filters": [
    { "layer": "Conv_1",  "pattern": "horizontal_edge",       "strength": 117.767, "peak_xy": [28, 106] },
    { "layer": "C3k2_2",  "pattern": "elongated_upright_shape","strength":  25.717, "peak_xy": [82,  59] },
    { "layer": "SPPF_9",  "pattern": "scene_layout_summary",  "strength":  11.548, "peak_xy": [ 2,  13] }
  ],
  "layer_flow": [
    { "stage": "layers_0-20",  "role": "edge_color_extraction",    "confidence_proxy": 0.006 },
    { "stage": "layers_2-22",  "role": "shape_composition",        "confidence_proxy": 0.002 },
    { "stage": "layer_9",      "role": "global_context",           "confidence_proxy": 0.003 },
    { "stage": "layer_10",     "role": "semantic_feature_encoding","confidence_proxy": 0.001 },
    { "stage": "layers_11-21", "role": "multi_scale_fusion",       "confidence_proxy": 0.002 },
    { "stage": "layer_23",     "role": "final_classification",     "confidence_proxy": 1.0   }
  ],
  "pre_nms": { "candidates": 0, "high_confidence": 0, "final_detections": 1 }
}
```

<br/>

### 💬 LLM Explanation (`explanation.txt`)
```
The model detected a "motorcycle" with high confidence in the middle-center of the image "bike.jpg."

In the initial layers, the model identified a strong horizontal edge pattern at coordinates (28, 106), suggesting the presence of significant linear edges, which are common in vehicles. This was followed by recognizing an elongated upright shape at (82, 59) in the middle layers, and a broad scene layout summary at (2, 13) by the SPPF layer, indicating awareness of the overall arrangement of objects in the scene.

As confidence built up through the layer flow, the early stages (layers 0-20) focused on extracting crucial edges and color details, albeit with low confidence. As it moved into layers 2-22, shapes began to compose, and by layer 9, the model gathered global context. The semantic features were encoded by layer 10, refining recognition, and multi-scale information was fused by layers 11-21. The final, decisive classification occurred in layer 23, which yielded the ultimate confidence.

The model's decision to classify the object as a "motorcycle" rather than another object is grounded in the strong class logit that was apparent in the final classification layer compared to insignificant or non-existent competition from other class logits in this case.

The internal consensus showed that even though there was only one final detection, the absence of pre-nms candidates suggests that the model was quite definite about the singular presence of a motorcycle.

In layman terms, the model spotted defining features typical of motorcycles like horizontal lines and elongated shapes. It subsequently aggregated these observations with the entire scene context to confidently determine that the object in view was a motorcycle, dismissing other possibilities due to strong evidence supporting this class.

Bottom line: The motorcycle classification is unambiguous.
```
<br/>

---

## 🤝 Contributing

### LENS is early. 

- 🐛 **Found a bug?** [Open an issue](../../issues/new)
- 💡 **Have an idea?** Start a [discussion](../../discussions)
- ⭐ **Like the project?** Drop a star — it genuinely helps

### 🎯 Coming soon

- 🔍 Evaluation Strategy.
- 🔍 Domain Adoption Strategy.
- 🤖 Add Support for other Most Frequently Used Pre-Trained Neural Networks Worldwide.
- 🤖 Add Support for LLM models from Claude & Qwen.


<br/>

<div align="center">

### 🔍 Magnifying glass over a glowing neural network, rising from an open box — _"Let's unbox the blackbox."_

**🤖 Built & Engineered by Nisarg Chodvadiya**

📬 **Get in touch:** [nisargc88@gmail.com](mailto:nisargc88@gmail.com)


</div>
