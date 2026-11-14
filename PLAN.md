# Virtual Bronchoscopy Navigation Planner — Project Plan

> Status: v0.4 (2026-11-10). Phase 0 scaffold and the Phase 1 core pipeline are built. Airway segmentation has a real-CT baseline on AeroPath (26 of 27 cases; mean Dice 0.86, about 55% of the true centerline reached); routing and rendering have run end to end on 3 real AeroPath cases (not yet reviewed in depth) and the phantom. See "Status" under §5 and `docs/aeropath_eval_2026-11-10.md`.
> **Intended use (v1): pre-procedure planning and education. Not for intra-procedure guidance or diagnosis.**

---

## 1. Product summary

Load a chest CT → build a 3D airway model → user picks an **origin** (default: trachea/carina) and a **destination** (usually a nodule) → app computes the best airway route, draws it as a line through the airway tree, and produces **virtual bronchoscopy (VB)** fly-through images and video along that route.

### Core user flow
1. Import CT (DICOM folder / zip / PACS export).
2. App auto-segments lungs + airway tree and shows progress. Clinician can review and correct.
3. Clinician scrolls MPR (axial/coronal/sagittal), clicks the nodule → marks destination (optionally adjusts a sphere to the nodule size).
4. Origin defaults to the trachea; can be moved to any airway point.
5. App proposes a route (and up to 2–3 alternates) with metrics: path length, smallest airway diameter along the way, generations reached, residual distance from last airway point to the nodule edge, bend angles.
6. Clinician previews the route in the VB view (play/scrub/step by bifurcation), sees path line + nodule through the wall + distance-to-target.
7. Export: MP4 fly-through, still images at every bifurcation and at the end point, PDF summary, path data (JSON).

### Non-goals for v1
- Real-time tracking (electromagnetic / shape-sensing / video-to-CT registration). Planning only. Design the data model so the path can be consumed by a tracking layer later.
- Diagnosis, nodule malignancy scoring, automatic nodule detection (nice-to-have in a later phase).
- Cloud multi-user, PACS integration, EHR integration.

---

## 2. Technical pipeline

```
DICOM ──► Volume (HU, isotropic) ──► Lung mask + Airway mask ──► Airway surface mesh
                                          │                          │
                                          ▼                          ▼
                              Skeleton → Airway tree graph      VB renderer
                                          │                          ▲
        Nodule point/size ─► Target airway selection ─► Path ─► Smoothed camera trajectory ─► Images / Video
```

### 2.1 Ingest and preprocessing
- Read DICOM series (SimpleITK/ITK or pydicom), validate: single series, consistent spacing, axial, ≥ ~200 slices, **slice thickness ≤ 1.25 mm preferred (≤ 1 mm ideal)**; warn on thick-slice (> 2.5 mm), motion, or truncated lung fields.
- Convert to HU (rescale slope/intercept), orient to a canonical patient frame (LPS), resample to isotropic ~0.5–0.75 mm.
- De-identify on import (strip PHI from anything stored/exported; keep a hashed study ID).

### 2.2 Segmentation
Two tracks, built in this order:
1. **Classical baseline (Phase 1):** trachea seed detection → region growing on air (≈ −1024 to −900 HU start, adaptive threshold) with **leak detection** (explosion control: stop/lower threshold when grown volume jumps) → morphological cleanup. Lung mask via thresholding + largest components + hole-fill.
2. **Learned model (Phase 2):** 3D U-Net (nnU-Net or MONAI) for airway segmentation to reach more distal generations without leaking into parenchyma. Candidate public data: ATM'22 (airway modeling), EXACT'09, AeroPath; LUNA16 / LIDC-IDRI for lung masks and nodules; TotalSegmentator can provide trachea/lobe labels for sanity checks. **Check each dataset's license before training or shipping weights.**
- Ensemble/merge: use ML mask as primary, classical mask as sanity check; flag disagreement for review.
- Editing tools: seed-add (grow more airway from a clicked point), erase-region, threshold slider.

### 2.3 Airway tree (centerline graph)
- 3D thinning/skeletonization (or distance-transform + minimal-path extraction) on the airway mask.
- Build graph: nodes = bifurcations/endpoints, edges = branches, each sample carries **radius** (from distance transform) and local tangent.
- Prune spurious spurs (length/radius thresholds), repair small gaps.
- Label branches: generation number from trachea; lobar/segmental names (RB1…LB10) via atlas/rule-based labeling — **Phase 3**, valuable for clinicians but not required for routing.

### 2.4 Target and route planning
- **Target:** user click + radius → nodule center and approximate extent (optionally refine with local region-growing segmentation of the nodule on lung window).
- **Candidate exit points:** all airway-tree samples within a search radius of the nodule. Score each candidate on:
  - Euclidean distance to nodule surface (lower is better)
  - Airway lumen diameter at the exit point (must admit the chosen scope: standard ≈ 5–6 mm, thin ≈ 4 mm, ultrathin ≈ 3 mm — configurable)
  - Alignment: angle between the airway tangent and the vector to the nodule (smaller is better)
  - Extra-luminal segment crossing: penalize passing through large vessels, fissures, or pleura (needs vessel/fissure segmentation → Phase 3; v1 uses distance only and shows a warning)
  - Path cost from origin: shortest path in the graph, with edge cost increased for narrow lumens and sharp bifurcation angles
- **Route:** Dijkstra/A* on the centerline graph from origin to best exit point, then an explicit **"extraluminal gap"** segment (dashed line) from the last airway point to the nodule. If no airway passes within a reasonable distance, report "no airway route found within X mm" instead of faking one.
- **Smoothing:** 1D Gaussian filtering with iterative distance-transform lumen clamping (avoids the Runge-type wall overshoots and oscillations associated with high-order B-splines); enforces minimum bend radius limits (configurable to scope capabilities); keeps the camera safely inside the lumen.
- Camera frames via **parallel transport** to avoid roll flips (Frenet frames twist at near-straight sections).

### 2.5 Virtual bronchoscopy rendering
- **Surface approach (recommended first):** marching cubes on the airway mask (smoothed) at ≈ −500 HU iso-level equivalent → mesh → decimate → render with OpenGL/Vulkan/VTK.
  - Perspective camera, FOV ≈ 100–120°, near plane ≈ 0.3 mm, optional barrel distortion to mimic a real scope lens
  - Headlight with distance falloff, specular wet-mucosa shading, pink mucosa palette
  - Overlays: route line, next-bifurcation marker, branch label, distance-to-target, nodule shown as translucent colored sphere behind the wall once within X mm, small orientation cube / mini 3D tree with camera marker
- **Volume ray-casting alternative:** direct volume rendering at the HU threshold; sharper subtle detail (e.g. carinal spurs) but heavier. Keep behind a renderer interface so both can be tried.
- Outputs:
  - Interactive fly-through (play / pause / speed / scrub / jump to bifurcation)
  - Still frames at each bifurcation and at the end point
  - MP4 (H.264) export via ffmpeg at ~30 fps, constant speed along path (e.g. 5–10 mm/s) with brief pauses at bifurcations

### 2.6 UI layout
- Main window: 3 MPR panes + 3D airway tree pane + VB pane; one pane can be maximized.
- Crosshair link: moving the VB camera moves MPR crosshairs and vice-versa.
- Path overlay on MPR (projected path points), on 3D tree (colored line), and in VB (floating guide line).
- Side panel: case info (anonymized), segmentation controls, origin/destination, route candidates with metrics, export.
- Always visible banner: "Planning aid — not for clinical use" until regulatory work says otherwise.

---

## 3. Platform and stack (decision: web app, 2026-11-10)

The options originally considered are kept below for the record. The web architecture that replaces them is in §3.1.

| | A. Desktop (Windows) | B. Android tablet | C. Web app |
|---|---|---|---|
| Core stack | Python 3.11, SimpleITK, scikit-image, VTK, PySide6, ONNX/PyTorch, ffmpeg | Kotlin + Compose UI, C++ core via NDK (ITK-lite / custom), Filament or OpenGL ES, ONNX Runtime Mobile | Cornerstone3D or vtk.js client, FastAPI + Python worker, Postgres/S3 |
| Compute fit | Best: CPU/GPU for segmentation, big volumes (300–600 MB) | Tight: RAM limits; ML segmentation on-device is hard | Server does the heavy lifting; client renders |
| VB rendering | VTK/OpenGL — easiest | Doable, more work | WebGL2 — doable, browser memory limits |
| Dev speed to validate algorithms | Fastest | Slowest | Medium |
| Clinical-workstation fit | Good | Good for bedside viewing | Good for sharing; PHI/hosting burden |

**Original recommendation (superseded by the web decision; the headless-core-first part still holds):** build the **processing core as a headless Python library + CLI first** (Phase 0–2). It is platform-independent and lets you validate segmentation/routing quickly on public CTs. Then pick the front end: desktop (PySide6 + VTK) is the shortest path to a usable app; if the Android target matters (the project lives under Android Studio), use the Android app as a *viewer* of precomputed results (mesh + path + video) and keep heavy processing on a desktop/server worker.

Prototype faster by checking whether the algorithm work can first be validated inside **3D Slicer** (Python scripting, has airway segmentation and VTK rendering) before writing a custom UI.

---

### 3.1 Web architecture

```
Browser (React + TypeScript)                       Server
┌──────────────────────────────────┐   HTTPS    ┌─────────────────────────────────────┐
│ Upload (DICOM folder/zip)        │ ─────────► │ FastAPI: auth, cases, job status    │
│ MPR viewer: Cornerstone3D        │            │ Job queue (RQ/Celery + Redis)       │
│ 3D tree + VB view: three.js      │ ◄───────── │ GPU worker: bronchnav core (Python) │
│   (WebGL2, mesh + path JSON)     │  results   │   segmentation → tree → route →     │
│ Route panel, export buttons      │            │   mesh (glTF/binary) + MP4 render   │
└──────────────────────────────────┘            │ Storage: encrypted disk or S3       │
                                                └─────────────────────────────────────┘
```

Division of work:
- **Server (heavy):** DICOM parsing, resampling, segmentation (ML needs a GPU), skeletonization, graph, route scoring, mesh generation, and MP4 rendering (headless VTK/EGL + ffmpeg).
- **Browser (interactive):** MPR scrolling and nodule picking, 3D tree, and the live VB fly-through rendered from the airway mesh and route JSON. Camera moves are local, so scrubbing is instant with no round trip.
- Picking a new origin or destination calls `POST /cases/{id}/route`. Routing runs on the cached graph, so it should return in well under a second. Re-segmentation is the only slow job.
- Volume to client: send a downsampled or chunked copy for MPR. Cornerstone3D can stream a NIfTI or DICOMweb series. Do not ship a full 500 MB volume if avoidable.

Key web-specific decisions:
- **PHI and hosting.** De-identify in the browser before upload (strip patient tags) so PHI never leaves the machine. Offer a `docker compose` local mode (the same stack on localhost) for sites that cannot use a hosted server. HIPAA/GDPR work, encryption at rest and in transit, audit logs and a BAA with the host all come before any real patient data.
- **MP4 export:** server-side for consistent look and speed. Client-side WebCodecs is a later option.
- **Mesh transfer:** decimate to ~200–500k triangles, send as binary glTF with Draco or meshopt compression. Browser memory is the limit.
- **Auth:** start with a single-user or token login, then add OIDC when multi-user is needed.
- **Optional DICOM server:** Orthanc plus DICOMweb as the import path in a later phase.

**Deployment decision (2026-11-10): local docker first.** The whole stack (web, api, worker, redis) runs on one machine via `docker compose` and is reached at `http://localhost`. Consequences:
- No patient data leaves the machine, which removes most hosting and HIPAA/GDPR work from the early phases. Keep in-browser de-ID anyway so exports and logs stay clean.
- Needs an NVIDIA GPU with the NVIDIA Container Toolkit for ML segmentation. Provide a CPU-only fallback (classical segmentation) so the stack still runs without one.
- Mount a local data directory as a volume for cases and results. Do not bake data into images.
- Pin image versions and model weights and record them in each result's metadata.
- Hosted deployment (auth, TLS, object storage, audit logging, BAA) is deferred until after Phase 5.

Revised phase notes:
- Phases 0–3 are unchanged: the Python core and CLI come first and are validated before any UI.
- Phase 4 becomes two tracks: (a) FastAPI + job queue + storage; (b) React client with Cornerstone3D MPR and a three.js VB view. Start with a viewer that loads a precomputed result folder (mesh + route JSON + volume) to de-risk rendering before wiring the backend.
- Add a Phase 4.5 security and privacy pass: in-browser de-ID, TLS, encrypted storage, audit logging, dependency scan.

## 4. Repository layout

Current (as built); planned modules are marked.

```
Navigation/
├── PLAN.md
├── README.md
├── core/                      # headless Python package + CLI: bronchnav
│   ├── pyproject.toml         # extras: dev (pytest), render (vtk, imageio, imageio-ffmpeg)
│   ├── bronchnav/
│   │   ├── io/volume.py       # load (DICOM/.mha/.nii.gz, HU, LPS), validate, isotropic resample
│   │   ├── segmentation/airway.py   # trachea seed, threshold-ladder region growing, leak control
│   │   ├── airway_tree/skeleton.py  # skeleton -> MST centerline tree, radii, spur pruning
│   │   ├── planning/route.py        # scope-aware shortest path, smoothing, camera frames
│   │   ├── render/vb.py             # VTK surface VB: stills, MP4, PLY mesh
│   │   ├── geometry.py        # grid frame <-> physical LPS
│   │   ├── pipeline.py        # run_plan(): CT -> route.json + outputs
│   │   ├── phantom.py         # synthetic test CT (testing only)
│   │   └── cli.py             # bronchnav info | phantom | plan
│   └── tests/                 # unit + end-to-end tests (phantom and synthetic volumes)
│   (planned: segmentation/ml.py and lungs.py; airway_tree/labeling.py; planning/target.py and
│    scoring.py; export/pdf_report.py)
├── server/                    # FastAPI API; job queue, workers and storage adapters planned
├── web/                       # React + TS client; Cornerstone3D MPR and three.js VB view planned
├── docker-compose.yml         # local single-machine deployment (api, worker, redis, web)
├── docker-compose.gpu.yml     # NVIDIA GPU override for the worker
├── data/                      # (gitignored) small local inputs and outputs
└── docs/                      # (planned) algorithm notes, validation reports, risk file
```

Large datasets and the Python venv live outside this Google Drive folder: `%USERPROFILE%\datasets\` and `%USERPROFILE%\venvs\`.

Path/result data contract (JSON, versioned): airway graph, selected route (ordered 3D points in patient mm + radius + tangent), target (center, radius), metrics, software + model versions, input-series hash.

---

## 5. Phases and milestones

| Phase | Goal | Deliverable | Done when |
|---|---|---|---|
| **0 — Setup (1 wk)** | Repo, env, sample data | Python env, 5–10 public chest CTs (AeroPath first, then LUNA16 / ATM'22 subsets, license-checked), viewer sanity script | Can load a CT and show axial/coronal slices |
| **1 — Classical pipeline (3–4 wk)** | End-to-end CLI | Region-growing airway + skeleton + graph + route + surface VB + MP4 | On ≥ 5 cases: trachea → a user-given distal point produces a smooth fly-through with no wall clipping |
| **2 — Robust segmentation (4–6 wk)** | Go distal, avoid leaks | ML airway model, leak guard, editing tools | Airway tree reaches ≥ generation 6–7 on most cases; leak rate low on held-out set; measured with EXACT'09-style metrics (tree length detected, branch detected, false-positive rate) |
| **3 — Target planning (3–4 wk)** | Nodule → best route | Nodule marking, candidate scoring, alternates, extraluminal-gap reporting, optional vessel/fissure avoidance | On cases with expert-annotated routes, planned exit airway matches the expert's segment/branch in most cases (set a numeric target with a pulmonologist) |
| **4 — Interactive app (4–6 wk)** | Usable UI | MPR + 3D + VB panes, linked crosshairs, route editing, export (MP4, stills, PDF, JSON) | A pulmonologist can go from DICOM import to exported video in < 10 min |
| **5 — Validation and hardening (ongoing)** | Trust | Retrospective study vs expert planning; failure-mode catalog; performance and QA | Written validation report; known failure modes documented and surfaced in UI |
| **6 — Regulatory track (if clinical use intended)** | Compliance | QMS-lite, IEC 62304 software file, ISO 14971 risk file, IEC 62366 usability, cybersecurity | Regulatory pathway confirmed with a consultant; not started before Phase 4 |

Timeline above assumes one full-time developer; treat as rough.

### Status (2026-11-10)

**Phase 0 — done (scaffold).** Repo layout, Python env (3.10 on the dev machine), FastAPI skeleton, React placeholder, compose files. Docker is not installed on the dev machine, so the stack has never been run. Dataset research is below.

**Phase 1 — pipeline built; segmentation baselined on real CTs; routing and rendering not yet run on real CTs.**

Built in `core/`:
- Ingest: DICOM series or volume file, HU, LPS orientation, suitability warnings, isotropic resampling that stays within the original scan extent.
- Segmentation: trachea seed (prefers the superior end, must be wrapped in soft tissue, deepest interior pixel) and threshold-ladder region growing. A leak needs both a >2x volume jump and an absolute gain over 30 mL (`min_leak_jump_ml`, untuned); a leak at the very first threshold is an error, not a result.
- Tree: skeleton to minimum-spanning-tree centerline graph with radii, spur pruning, root at the proximal tracheal endpoint.
- Route: scope-aware shortest path (only through airways the scope fits; the exit must meet the full minimum diameter), smoothing with lumen clamping, uniform re-sampling, parallel-transport camera frames, bifurcations located by arclength. Lumen diameter and wall clearance are reported separately.
- Rendering: VTK surface virtual bronchoscopy with normals facing the lumen, stills at each bifurcation, MP4, PLY mesh, binary GLB mesh aligned to physical LPS coordinates, translucent ghost nodule, HUD.
- Export and CLI: `route.json`, `bronchnav info | phantom | plan` with flexible coordinate inputs and tunable parameters.
- Tests: 40 pass in core, 14 in the console API (phantom end to end, leak, seed and corner-contact connectivity cases on synthetic volumes, scope-size constraint, mesh orientation, odd render sizes, recursive DICOM series loading, coordinate parsing, parameter exposures, physical mesh alignment and GLB export).

**Synthetic phantom** (`bronchnav.phantom`): airway Dice about 0.90 with precision about 1.0, a deliberately opened branch tip triggers leak detection and stays contained, the route stays inside the lumen, meshes (.ply, .glb) and route JSON are synchronized in physical LPS space, and a 22 s MP4 renders in seconds.

**Real CTs: AeroPath segmentation baseline** (26 of 27 cases, 6-connected labeling, 25 HU ladder; `docs/aeropath_eval_2026-11-10.md`):

| Metric | Mean | Median |
|---|---|---|
| Dice | 0.861 | 0.869 |
| Precision | 0.966 | 0.988 |
| Recall | 0.783 | 0.806 |
| Truth centerline detected | 0.551 | 0.566 |

- It is safe but shallow: it rarely invents airway (false positives 0.3 mL on average) but reaches only about 55% of the truth centerline, because a single global threshold leaks into the lung (jumps of 100-3000 mL against 24-105 mL of true airway) before distal airways cross it. The leak guard fires on 23 of 26 cases and the leaks are real.
- A finer 10 HU ladder gives the same numbers; global Gaussian smoothing before thresholding lowers centerline detection (0.49 to 0.41 on a 10-case subset). Neither is kept.
- Case 9 (emphysema) fails by design with a clear error: at -950 HU the lung tissue is air-like (1,245 mL, 4% true airway). It needs the learned model.
- 26-connected labeling (the former library default) lets a single diagonal noise voxel bridge an airway wall: case 16 becomes a 1,126 mL leak at the first threshold. 6-connected is better on every metric here (Dice 0.806 to 0.834, false positives 1.6 to 0.3 mL). **The library default is now 6-connected (`connectivity=1`)**, verified on cases 8, 10 and 16, with a regression test for the corner-contact case.
- Case 10 (1.25 mm slices, airway near -915 HU) was a rule bug: only 1.3 mL fell below -950 HU, so the next genuine 36 mL of airway looked like a leak. Fixed: a region under 10 mL (`min_trusted_ml`) only counts as leaking above 100 mL (`early_leak_ml`). Dice 0.03 to 0.73.
- **Routing and rendering on real CTs (3 AeroPath cases, stand-in targets just past a distal airway tip):** all three run end to end in 12-19 s and render stills, an MP4 and a GLB. Case 20: 181 mm route, 4 bifurcations. Case 16: 222 mm, 5. Case 1: 240 mm, 12 bifurcations, min lumen diameter 2.6 mm (the scope-size check allows nodes down to 0.8x the limit because skeleton radii are only accurate to about a voxel). Every route stays inside the lumen.

  **Review of the stills (2026-11-10):** carinae, bifurcations, the target ghost and the HUD all read as real endoluminal views, and the route enters the right lumen at each bifurcation. Two findings. (1) The guide line hooked back on itself at bifurcations and every route warned of a 3-5 mm bend radius: measuring along the routes showed the tight bends (40-60 degrees within 6 mm) were at junctions and at the tracheal tip, where the centerline passes through the junction centre, and a fixed 2.5 mm smoothing scale was too small for wide airways. Fixed with adaptive smoothing (scales 2.5, 5, 8 and 12.5 mm; at each point the smoothest one that keeps clearance of at least half the lumen radius). Median bend radius went from about 20-29 mm to 62-87 mm, samples under 8 mm from 3.6-10% of the route to 0.7-1.2%, and the largest turn within +-3 mm from 50-61 to 20-30 degrees; the remaining 4-5.5 mm minima are 3-5 isolated samples. Routes shortened slightly (case 20: 181 to 176 mm). (2) The wall is bumpy from the voxel mask, and the last frame of each route shows a ragged truncated lumen: that is where the segmentation stops, not a routing fault.

  **Six more cases (3, 22, 26, 8, 14; 9 fails by design on emphysema):** all five ran end to end in 12-21 s with video. Case 26 is clean (min bend radius 27.7 mm). Case 22 started with an 8-17 mm sideways hook of the skeleton at the tracheal tip (a 123 degree turn), now trimmed by `_trim_start_hook` (drops leading points that point more than 60 degrees away from the route's main direction, at most 25 mm, only for the automatic origin; case 22 worst turn 123 to 49 degrees, no other case trimmed). The remaining 2-5 mm bend radii in cases 3, 22, 14 and 8 sit in narrow distal airways (2-3 mm lumen radius) at bifurcations, where the route turns 30-45 degrees with no room to cut the corner; they may be real anatomy and are left as warnings. Stills reviewed in depth for four cases (20, 16, 1, 22); the rest by metrics only. Dark flat patches in case 22's trachea are wall faces turned from the headlight (not holes: the pixels are not the background color), probably thin fragments of the segmentation; not investigated.

  **No-DICOM imports and the console:** `io/image_stack.py` rebuilds an approximate HU volume from a JPEG/PNG/BMP/TIFF series given pixel size, slice spacing and display window, with slice-order detection by growing an airway from both ends. On AeroPath case 20 exported as 8-bit lung-window JPEGs (quality 90), segmentation matched the original (Dice 0.935 vs 0.935 against truth, masks agree at 0.996). One case, one window; unvalidated otherwise. `segmentation/lungs.py` gives an approximate lung mask (Dice 0.92-0.96 vs AeroPath lung labels, about 1.5 s) so automatic targets work on any CT. The local console imports volume files, image series or a path on the PC, refuses imports on proxied URLs (Tailscale), and the interactive fly-through supports looking around (drag, zoom, reset). The gaps to the stand-in target (16-27 mm) mostly reflect the incomplete airway tree. Two real-data bugs were found and fixed on the way: the route root landing on a thin distal branch or a thick central stub, and a length-only spanning tree linking the trachea to the rest through thin wall-hugging skeleton nodes (now radius-aware).
- Phase 1's "done when" (at least 5 real cases with a smooth fly-through) **is not yet met**: 3 of 5 have run, none reviewed in depth. A localhost test console (README) now makes reviewing them easy.

Second-opinion reviews with the Antigravity CLI (`agy`):
- Gemini 3.8 Flash (low): about 11 of 18 claims were false positives; it found one real bug (odd video sizes break H.264) and one worthwhile concern (the leak rule).
- Gemini 4 Argon (medium): most claims real or plausible. Fixes made, with tests where practical: first-threshold leak guard; seed search (superior end, tissue ring, interior pixel); tree root; scope-aware routing; no unchecked smoothing pass at the end of the clamp loop; uniform spacing before curvature; interior-only bifurcations located by arclength; mesh normals (they pointed away from the lumen, hidden by two-sided lighting, and the PLY was inside-out); mesh smoothing that keeps 1-voxel distal tubes with a zero-padded cap; clamped frame positions; backface-culled ghost nodule.
- A bug the new tests exposed: `resample_isotropic` padded one extra air row past the scan edge, which connected the trachea to the air outside the body once the trachea was at that end. Fixed. The phantom is now anatomically oriented (trachea at high z).
- Reviewed and not changed: simultaneous sibling-spur pruning (low impact, order-dependent), the parallel-transport projection (angles between 0.5 mm samples are tiny), guide-line popping at the proximal tip (sub-pixel at 0.5 mm spacing).

**Datasets** (verified 2026-11-10; check licenses again before training or redistributing):

| Dataset | Role | Access and license |
|---|---|---|
| AeroPath | Phase 1/2 airway ground truth: 27 CTs with airway and lung masks, includes pathology | Zenodo record 10069289, one 5.04 GB zip, CC-BY 4.0 (README also says MIT for the loader code; confirm `license.md` in the archive) |
| ATM'22 | Phase 2 training, about 500 CTs | Registration needed; no data license found, confirm on the challenge page and Zenodo records |
| LUNA16 (from LIDC-IDRI) | Real nodule targets; no airway masks | CC BY 4.0 (LIDC-IDRI source reported as CC BY 3.0) |
| EXACT'09 | Standard airway benchmark, 40 CTs | Team registration and terms; no redistribution; a mirror says the original link is dead, email the organizers |
| TotalSegmentator | Trachea/bronchia and lung lobe labels, 1,228 CTs | Zenodo; license field not shown, confirm on the record |

**AeroPath:** downloaded 2026-11-10 to `%USERPROFILE%\datasets\aeropath\` (outside the Google Drive folder), MD5 verified (`3fd5106c175c85d60eaece220f5dfd87`), extracted to `extracted\AeroPath\`. `license.md` is CC BY 4.0 (attribution required). The data is 27 contrast-enhanced CTAs, 0.5-1.25 mm slices, with expert-corrected airway masks (region growing + BronchiNet, manual refinement, pulmonologist-verified).

**Next steps (in order):**
1. Run `plan` (route, stills, video) on real AeroPath cases. There are no nodule annotations: choose parenchymal targets near a distal airway, and pull LUNA16 cases for real nodules. Judge route and video by eye; look for wall clipping, roll flips and camera-in-wall on real anatomy.
2. Go deeper than the global threshold: either branch-wise leak control (grow at a higher threshold and roll back only the leaking front) as a cheap classical improvement, or move to Phase 2 and train the learned model on ATM'22 and AeroPath with a held-out split. The AeroPath numbers above are the baseline to beat.
3. Add the remaining metrics from §6 (branch count detected, leak volume per case as a time series) to `core/scripts/eval_aeropath.py`.
4. Close or amend Phase 1's "done when" once routing and rendering have run on at least 5 real cases.

**Known limits:**
- Threshold-ladder growing under-segments distal airways (about 55% of the true centerline on AeroPath) and fails on emphysema (case 9); Phase 2 adds the learned model. The leak-rule constants (2x ratio, 30 mL jump, 10 mL and 100 mL early rule) are hand-set and have been checked on 27 AeroPath cases only.
- Exit selection is "nearest airway point the scope can reach". The full scoring (alignment, vessels, pleura) is Phase 3.
- Rendering is surface-based with a headlight; staircase rings from the voxel mask may need finer smoothing or a volume renderer on real data.
- The nodule is a translucent "ghost" sphere on an overlay layer, not depth-correct through the wall.
- Branch labels (segment names) are not implemented.
- The phantom's junction counts and radii are not anatomical; do not read clinical meaning into them.

---

## 6. Validation plan

- **Segmentation:** Dice vs reference, tree length detected, branch count detected, false positive (leak) volume — on ATM'22 / EXACT'09-style held-out cases.
- **Centerline:** distance from skeleton to reference centerline; correct bifurcation topology.
- **Routing:** agreement with expert-chosen route and exit segment on retrospective cases; time-to-plan; fraction of nodules for which a "bronchus sign" route exists vs the app's finding.
- **Rendering:** no wall clipping (camera always inside lumen), no roll flips, frame rate target ≥ 30 fps interactive, MP4 export in < 1 min per route.
- **Robustness:** thick slices, motion, emphysema, tumor-obstructed airway, post-surgical anatomy, mucus plugs, contrast vs non-contrast.
- **Clinical review:** blinded rating by ≥ 2 pulmonologists of VB realism and route plausibility.

---

## 7. Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Airway segmentation leaks into lung or stops early | Wrong or truncated route | ML + leak guard, editing tools, always show segmentation for review, show "airway not visible beyond here" |
| **CT-to-body divergence** (CT taken at full inspiration; lung moves/atelectasis at procedure) | Planned target position differs from reality | State clearly in UI/report; planning only; later phase may integrate intra-procedure registration |
| Nodule has no airway leading to it | No valid route | Report gap distance and nearest airway; don't invent a path |
| Over-trust of a polished 3D view | Clinical harm | Prominent "planning aid" labeling, expose uncertainty, require user confirmation of the route |
| Public datasets' licenses restrict commercial use | Blocks productization | License audit in Phase 0; plan for a separately licensed or self-annotated training set |
| Performance on large volumes | Slow UX | Isotropic resampling to fixed grid, mesh decimation, GPU rendering, background workers |
| PHI exposure | Legal | De-ID on import, no cloud in v1, encrypted local storage |
| Regulatory | Cannot be used clinically | Keep research-use labeling until Phase 6; involve regulatory consultant early |

---

## 8. Open questions

1. ~~Platform~~ — **decided: web app, local docker deployment first** (see §3.1). A hosted deployment is deferred until after Phase 5 validation and the privacy/regulatory review.
2. **Intended use:** research/education/demo only, or heading toward clinical use (drives regulatory scope and licensing)?
3. **Data access:** starting from public datasets (AeroPath first, see §5 Status). Still open: do you have your own CT cases (with consent/IRB) to add later, and would any be usable for a retrospective study?
4. **Scope of v1 route planning:** airway-only routing (simplest), or also vessel/pleura avoidance for the extraluminal segment?
5. **Clinical advisor:** is there a pulmonologist/interventional pulmonologist to review routes and VB realism?
6. **Output expectations:** is MP4 + stills enough, or is a live interactive fly-through required in v1?

---

## 9. Next steps (written 2026-11-10)

### 9.1 Where things stand

| Area | State |
|---|---|
| Core pipeline (ingest, airway segmentation, tree, routing, rendering) | Built, 49 tests. Real-CT baseline on AeroPath: Dice 0.86, but only about 55% of the true centerline reached. |
| Routing and fly-through on real CTs | 8 AeroPath cases run end to end; stills reviewed in depth for 4. |
| Local console | CT viewer (scroll, click, HU, windowing), imports (volume file, JPG/PNG series, path), interactive fly-through with look-around, edit-path engine (backend + UI). |
| Public demo (Funnel, passcode) | **Interactive tier live on public data (phantom + the 27 AeroPath CTs)**: sign in, place/edit custom nodule in the phantom CT or pick one on an AeroPath CT, run planning, 3D airway tree with alternatives and path editing, save edited runs. Enforced limits: session isolation, 1 active run per session, no hourly cap (released 2026-11-13), 3 concurrent public runs, 7-min timeout, pruning. Tested (server suite: 42 tests passing on 2026-11-10; core suite: 49). Server hardening from the 2026-11-10 review: the login limit and per-visitor quotas key on the proxy-appended address (not a forgeable first entry), the check-and-start of a public run is atomic, and a malformed cookie is a 401 instead of a 500. |
| Operator nodule editing in the phantom | Wired into console UI, CT endpoints, and CLI (`--nodule X Y Z --nodule-radius R --nodule-hu H`). Placements outside lung or on airway rejected with 422. |
| Not started | Learned segmentation, vessel/pleura avoidance, branch labels, accounts, Docker run, clinical validation. |

### 9.2 In-flight steps (completed 2026-11-10)

1. **Wire nodule editing into the console.** [Done] `ct.py` accepts `nodule` for phantom (cache key includes it, invalid placement is 422); `JobRequest.phantom_nodule` with bounds; `plan_job.py --nodule X Y Z --nodule-radius R --nodule-hu H`. Verified: placements outside a lung or on an airway are refused.
2. **Edit-path UI in the 3D tab.** [Done] Airway centerline tree displayed with node markers, ranked alternatives table (gap, length, min lumen, approach angle), instant replan preview next to automatic route, "Save as a new run", and "Reset to automatic".
3. **Public-mode page.** [Done] UI shows controls for phantom source only; hides AeroPath, imports and Log; displays session privacy note ("Your runs are private to this session and deleted after 12 h"). Signed-in visitor can place nodules, plan routes, and edit paths end to end.
4. **Tests for the public limits.** [Done] The server tests pass (test_console.py and test_ct.py): per-session isolation, one active run per session, hourly cap, global cap, bounds on target/radius/scope/nodule, 7-minute job timeout, pruning, and end-to-end edit run on a phantom job.
5. **Restart and verify.** [Done] Console restarted on port 8000; verified locally and over public Tailscale Funnel URL (`https://<your-node>.<your-tailnet>.ts.net:8443`); 5 concurrent visitors tested with 100% success (0.11 s total) and stable RAM.
6. **Docs and notes.** [Done] README and PLAN.md updated with test counts and interactive demo details.

### 9.3 Run the public demo safely (decisions and hardening)

- Per-IP request limits and a request-size cap in front of the app; log rotation; a visible kill switch (a file that closes public access instantly); a disk cap on the jobs folder.
- Auto-start at login: a launcher is written (`server/start_console_at_login.vbs`) but installing it needs your explicit go-ahead.
- Decide the unrelated public Funnel on port 443 (points at port 3000, nothing listening): remove it or keep it.
- Passcode hygiene: 12+ characters, rotate after each demo, never reuse.
- Never put patient data on the public side. Only public data is reachable there: the synthetic phantom and the AeroPath dataset (CC BY 4.0, opened to passcode visitors on 2026-11-13 at the owner's request). Imports and local paths stay local-only.

### 9.4 Close Phase 1 properly (about a week)

- A reproducible script that plans 8 to 27 AeroPath cases and writes a metrics table (inside-lumen fraction, wall clearance, bend radius, bifurcations, gap); review the remaining stills (cases 3, 26, 8, 14).
- Investigate the dark flat patches in case 22's trachea (thin fragments of the mask?) and the wall bumpiness (smoother mesh or volume rendering).
- Decide and record the numeric "done when" for Phase 1.

### 9.5 Phase 2: go deeper than a global threshold (the biggest quality lever, 4 to 6 weeks)

- **2a. Branch-wise leak control** (classical, 2 to 3 days): grow at a higher threshold and roll back only the leaking front. *Target:* centerline detected 0.55 to at least 0.7 on AeroPath with no new false positives. Cheap and may remove the need for 2b in many cases.
- **2b. Learned model:** nnU-Net on AeroPath plus ATM'22 (registration and license check first; the GPU is an NVIDIA card, to be confirmed with the container toolkit). Fixed split (for example AeroPath 20 train, 7 held out); metrics Dice, centerline detected, branches detected, leak volume. *Target:* centerline at least 0.8, false positives under 2 mL, and emphysema cases (like case 9) handled. Keep the classical path as a fallback and as a sanity check on the model.
- **2c. Extend the evaluation harness** (branch count, leak volume) so every change is scored on the same cases.

### 9.6 Phase 3: planning quality (3 to 4 weeks, needs a clinical advisor)

- Tune the candidate scoring (gap, approach angle, lumen, bends) against expert-chosen routes.
- Vessel and pleura avoidance for the extraluminal segment (needs vessel and fissure segmentation; check TotalSegmentator's license), segment naming for branches, optional nodule detection (LUNA16).

### 9.7 Phase 4: turn the console into the app (4 to 6 weeks)

Accounts (OIDC) instead of a shared passcode, upload with in-browser de-identification, a real job queue (RQ/Redis) instead of subprocesses, a database, Docker compose actually run (Docker is not installed yet), Cornerstone3D MPR and a proper React client, PDF report export, mobile and accessibility passes, and a security pass. Hosted deployment stays deferred until validation.

### 9.8 Validation and regulatory (ongoing; before any clinical use)

Name a pulmonologist advisor; blinded review of route plausibility and video realism; a retrospective comparison against expert planning; a failure-mode catalog shown in the UI; then, only if clinical use is intended, the IEC 62304 / ISO 14971 / IEC 62366 file with a regulatory consultant. Everything stays labeled "Not for clinical use" until then.

### 9.9 Housekeeping

`git init` and a first commit (the folder has no version control), a CI run of both test suites, pinned dependency versions, removing scratch scripts, and a backup plan for the data outside the Drive folder.

### 9.10 Decisions needed from you

1. Go-ahead to install the login launcher (auto-start)?
2. Keep or remove the unrelated public Funnel on port 443?
3. Is the target clinical use, or research and education only? (Drives Phase 4 and 6.)
4. Is there a clinician who can review routes and videos?
5. Do you want me to do 2a (classical, fast) before committing to 2b (training a model)?
6. Should the public demo stay behind a passcode, or move to named accounts?
