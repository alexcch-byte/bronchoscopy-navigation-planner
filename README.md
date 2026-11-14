# Bronchoscopy Navigation Planner

Virtual bronchoscopy planning from chest CT: load a CT, build a 3D airway model, pick an origin and a destination (usually a nodule), get a route through the airway tree plus virtual bronchoscopy stills and a fly-through video.

**Planning aid only. Not for clinical use.** See [PLAN.md](PLAN.md) for the full design, phases, risks and open questions; the current next-steps plan is [PLAN.md section 9](PLAN.md#9-next-steps-written-2026-11-10).

## Status (2026-11-10)

| Part | State |
|---|---|
| `core/` pipeline (ingest, airway segmentation, centerline tree, routing, VB rendering, CLI) | Built. 44 core tests, 23 console API tests, 8 CT tests (75 total passing). Routing and rendering have run end to end on 8 real AeroPath cases and the phantom. |
| Real CT validation | Airway segmentation scored on AeroPath (26 of 27 cases): baseline mean Dice 0.86, precision 0.97, 55% of the true centerline reached; with the thin-airway extension (now the default) Dice 0.87, precision 0.92, 70% of the centerline. See [Evaluating on AeroPath](#evaluating-on-aeropath), `docs/aeropath_eval_2026-11-10.md` (baseline) and `docs/aeropath_eval_2026-11-14_small_airways.md`. |
| `server/` (FastAPI + test console) | Local test console with CT slice viewer, custom nodule placement in phantom, 3D airway tree with alternatives and path editing, adding an airway the segmentation missed (click along it on the CT, preview the route into it, save as a new run; it is marked in the run's warnings as operator-drawn), and a live Funnel demo on public data (phantom and AeroPath). |
| `web/` (React + TypeScript) | Placeholder page. |
| Docker stack | Written but never run: Docker is not installed on the dev machine. |
| ML airway segmentation, nodule-aware exit scoring, branch labels | Not started (Phases 2–3). |

## Layout

| Path | What |
|---|---|
| `core/` | `bronchnav` Python package and CLI. Extras: `dev` (pytest), `render` (vtk, imageio, imageio-ffmpeg) |
| `core/bronchnav/io/` | CT load (DICOM series or `.mha`/`.nii.gz`, HU, LPS), suitability warnings, isotropic resampling |
| `core/bronchnav/segmentation/` | Trachea seed finder and threshold-ladder region growing with leak control |
| `core/bronchnav/airway_tree/` | Skeleton to minimum-spanning-tree centerline graph with radii and spur pruning |
| `core/bronchnav/planning/` | Scope-aware shortest path, smoothing, parallel-transport camera frames |
| `core/bronchnav/render/` | VTK surface virtual bronchoscopy: stills, MP4, PLY mesh |
| `core/bronchnav/phantom.py` | Synthetic chest CT with a branching airway and a nodule (testing only) |
| `core/tests/` | Unit and end-to-end tests on the phantom and small synthetic volumes |
| `server/` | FastAPI API and RQ worker (Phase 0 skeleton) |
| `web/` | React + TypeScript client (Phase 0 placeholder) |
| `docker-compose.yml`, `docker-compose.gpu.yml` | Local deployment: redis, api, worker, web; GPU override for the worker |
| `data/` | Small local inputs and outputs (gitignored). Large datasets live outside this folder, see below |

## Where things live (this is a Google Drive folder)

Venvs, `node_modules` and large datasets sync badly and burn Drive quota, so keep them **outside** this folder:

| What | Where |
|---|---|
| Python venv | `%USERPROFILE%\venvs\bronchnav` |
| AeroPath dataset | `%USERPROFILE%\datasets\aeropath\` |
| `node_modules` | Build `web/` from a local copy outside Drive, or accept the sync overhead |

## Core: develop and test

Python 3.10+ (the dev machine has 3.10; the Docker image uses 3.11).

```bash
python -m venv ~/venvs/bronchnav
~/venvs/bronchnav/Scripts/python -m pip install -e "core[dev,render]"
~/venvs/bronchnav/Scripts/python -m pytest core
~/venvs/bronchnav/Scripts/bronchnav info path/to/dicom_folder_or_ct.mha
```

### GPU acceleration (NVIDIA CUDA, optional)

With an NVIDIA GPU the heavy array steps (threshold-ladder labeling, the thin-airway tube filter, morphology) run on the GPU through CuPy; without one everything runs on the CPU and gives the same masks.

```bash
~/venvs/bronchnav/Scripts/python -m pip install -e "core[gpu]"     # installs cupy-cuda12x with the CUDA 12 runtime libraries
```

It is picked up automatically. `BRONCHNAV_GPU=0` forces the CPU, `BRONCHNAV_GPU=1` fails loudly if no usable GPU is found. If the GPU fails mid-run (out of memory, driver error) the step is repeated on the CPU, so a GPU problem never fails a run. Measured on an RTX 4070 with AeroPath case 5 (a 413 x 442 x 442 grid): the thin-airway extension 10.5 s on the CPU vs 0.3 s on the GPU, the threshold ladder 4.5 s vs 1.5 s, and a whole planning job 36 s vs 11 s (the very first run on a machine also pays about 20 s of one-time CUDA kernel compilation; it is cached on disk afterwards). The video render and the centerline tree are unchanged.

### Try it on the synthetic phantom

No real CT is needed. `phantom` writes a synthetic chest CT (anatomically oriented, trachea at the head end) with a branching airway tree and a free-standing nodule, and prints the nodule position:

```bash
~/venvs/bronchnav/Scripts/bronchnav phantom data/phantom.mha
~/venvs/bronchnav/Scripts/bronchnav plan data/phantom.mha --target 60.2 -1.4 -37.0 --target-radius 5 --out data/out
```

`plan` writes to `data/out/`:

| File | Contents |
|---|---|
| `route.json` | Route points, lumen radius, tangents, bifurcations, metrics, warnings, segmentation notes (physical LPS mm) |
| `airway_mask.nii.gz` | Airway segmentation on the 0.75 mm working grid |
| `airway_mesh.ply` | Airway surface in physical LPS mm, normals facing the lumen |
| `airway_mesh.glb` | Airway surface in binary glTF (GLB) format in physical LPS mm |
| `stills/` | One image per bifurcation plus the end of the route |
| `flythrough.mp4` | Fly-through video |

Meshes (`.ply`, `.glb`) and `route.json` are synchronized in physical LPS space (ITK coordinate convention), allowing immediate overlay in 3D viewers.

CLI options:
- `--target COORD [COORD ...]`: nodule center in physical LPS mm (accepts 3 numbers or comma/space-delimited string, e.g. `--target -60.2 -1.4 -37.0` or `--target "-60.2, -1.4, -37.0"`)
- `--target-radius MM`: nodule radius in mm (default: 5.0)
- `--origin COORD [COORD ...]`: route origin in physical LPS mm (default: top of trachea)
- `--min-diameter MM`: smallest airway the scope can enter in mm (default: 3.0)
- `--min-bend-radius MM`: minimum allowed bend radius in mm before warning (default: 12.0)
- `--fov DEG`: virtual bronchoscopy camera field of view in degrees (default: 100.0)
- `--leak-factor FACTOR`: relative volume jump threshold for leak detection (default: 2.0)
- `--min-leak-jump ML`: minimum absolute volume jump in mL for leak detection (default: 30.0)
- `--spacing MM`: isotropic working spacing in mm (default: 0.75)
- `--width / --height`: rendering resolution (default: 640x480)
- `--speed MM_PER_S`: fly-through speed in mm/s (default: 8.0)
- `--no-render`: skip stills, video, and meshes
- `--no-video`: skip fly-through video generation only

If the route cannot reach the nodule it ends at the closest airway the scope can reach and reports the remaining extraluminal gap. It never invents a path.

## Importing a CT

`bronchnav info|plan <path>` takes either:

- a **folder**: read as a DICOM series (searched recursively; the largest series is used), or
- a **single volume file** in any 3D format SimpleITK reads, no DICOM needed: NIfTI (`.nii`, `.nii.gz`), MetaImage (`.mha`, or `.mhd` plus its `.raw`), NRRD (`.nrrd`), VTK, MINC, MRC, GE4/5 and a few others.

A volume file must be in Hounsfield units with real voxel spacing. A zipped DICOM folder must be unzipped first. A single 2D image, a PDF or a video cannot be used.

Converting other sources: `dcm2niix` (DICOM to NIfTI), 3D Slicer (opens most formats and saves NIfTI/NRRD/MHA), ITK-SNAP, or SimpleITK `ReadImage`/`WriteImage`.

### An image series (JPEG, PNG, BMP, TIFF) when there is no DICOM

`bronchnav plan <folder of images> --image-stack --pixel-spacing MM --slice-spacing MM [--window-center -600 --window-width 1500 --first-slice auto|head|feet --invert]`, or the "My own CT" source of the console. The images are sorted in natural filename order (`img2` before `img10`).

JPEGs carry no HU values and no spacing, so the importer rebuilds an **approximate** volume from what you give it: the pixel size and slice spacing (from the scan report or the viewer that made the images), and the display window the images were saved with, to map gray levels back to HU (a lung window, centre -600 and width 1500, is the default). Anything outside the window was clipped when the images were made, so a lung window keeps air and lung but loses bone and contrast. The images must be in radiological orientation (patient's left on the right, anterior at the top) and unrotated. "First image is" can be detected automatically: both slice orders are tried and the one where the trachea grows into a plausible airway tree wins; set it by hand if the route looks upside down.

How well it works (one real case, AeroPath 20, exported as 8-bit lung-window JPEGs at quality 90 in both slice orders): the airway segmentation matched the one from the original data (Dice 0.935 against the ground truth for both; the two masks agree at Dice 0.996, mean HU error 4 in the lung range), and the slice order was detected correctly both ways. This is one case with a known window; other windows (for example a mediastinal window, which turns air into flat black) will do much worse, and the result is unvalidated. Every such run carries a warning in its results. Not for clinical use.

### Privacy

The console's "My own CT" source copies a file or image series to `%USERPROFILE%\bronchnav-uploads\` (override with `UPLOADS_DIR`; never served over HTTP) or just points at a path on this PC, and has a Remove button. Both are **refused when the request comes through a proxy** (Tailscale `serve` or Funnel), because the console does no de-identification and the public view is read-only: open `http://127.0.0.1:8000` on this PC to import. Runs of imported scans are also hidden from the public view and their files are never served to it. Set `ALLOW_REMOTE_UPLOADS=1` to override (not recommended). De-identify real scans before using them anywhere.

## Datasets

All public. Check each license before use; never commit patient data.

| Dataset | Role | Access and license (verified 2026-11-10) |
|---|---|---|
| [AeroPath](https://zenodo.org/records/10069289) | Phase 1/2 airway ground truth (27 CTs, includes pathology) | Zenodo, one 5.04 GB zip, CC-BY 4.0 |
| [ATM'22](https://zenodo.org/record/7949582) | Phase 2 training (about 500 CTs) | Needs registration; data license not stated, confirm first |
| [LUNA16](https://luna16.grand-challenge.org/Data/) | Real nodule targets (no airway masks) | CC BY 4.0 (source LIDC-IDRI reported as CC BY 3.0) |
| [EXACT'09](https://image.diku.dk/exact/) | Standard airway benchmark (40 CTs) | Registration and terms; no redistribution; a mirror says the link is dead |
| [TotalSegmentator](https://zenodo.org/records/10047292) | Trachea and lung lobe labels (1,228 CTs) | Zenodo; license field not shown, confirm on the record |

### AeroPath

Downloaded 2026-11-10 to `%USERPROFILE%\datasets\aeropath\AeroPath.zip` (5.04 GB). MD5 verified (`3fd5106c175c85d60eaece220f5dfd87`), extracted to `%USERPROFILE%\datasets\aeropath\extracted\AeroPath\`, `license.md` is CC BY 4.0 (attribution required; citation at https://github.com/raidionics/AeroPath). 27 contrast-enhanced CTAs (tumors, sarcoidosis, emphysema), slices 0.5-1.25 mm, each with `N_CT_HR.nii.gz`, `N_CT_HR_label_airways.nii.gz` and `N_CT_HR_label_lungs.nii.gz`. No nodule annotations, so for routing tests pick target points in the lung parenchyma near a distal airway.

Treat the archive as untrusted data: run scripts with `python -I` so the dataset folder is never on `sys.path`.

### Evaluating on AeroPath

```bash
~/venvs/bronchnav/Scripts/python -I core/scripts/eval_aeropath.py \
    --root $HOME/datasets/aeropath/extracted/AeroPath \
    --out data/eval_aeropath.json                             # add --cases 1 2 3 for a subset
```

Options: `--connectivity {1,2,3}` (region-growing labeling; the library default is now 1, i.e. 6-connected), `--step HU` (ladder step, default 25), `--smooth-mm MM` (Gaussian before thresholding). About 10 s per case. Per case it reports Dice, recall and precision against the truth, the fraction of the truth centerline detected, false-positive volume, leak flag and tree stats.

Baseline (6-connected, 26 of 27 cases): mean Dice 0.861, precision 0.966, recall 0.783, centerline detected 0.551. Case 9 (emphysema) errors on purpose: at -950 HU the lung itself is air-like. A finer ladder does not help and smoothing hurts. Full tables and findings: [docs/aeropath_eval_2026-11-10.md](docs/aeropath_eval_2026-11-10.md).

## Local test console (browser page)

A page for trying the pipeline without the command line: pick the synthetic phantom or a real AeroPath case, choose how the target is placed, run, and see the metrics and warnings, an interactive 3D airway with the route (with an inside view that flies along it), the fly-through video and the stills. It runs on this machine only (127.0.0.1), one job at a time.

```bash
~/venvs/bronchnav/Scripts/python -m pip install fastapi uvicorn        # once
~/venvs/bronchnav/Scripts/python server/run_local.py                   # then open http://127.0.0.1:8000
```

- **Targets:** *Automatic* on AeroPath places a test point in the lung just past a distal airway tip (slider chooses which tip, and how many mm beyond). It is not a nodule. On the phantom it uses the known nodule. On your own CT it does the same inside an approximate lung mask found from the scan (Dice 0.92-0.96 against AeroPath's lung labels). *Manual* takes physical LPS mm.
- **My own CT:** import a volume file (`.nii`, `.nii.gz`, `.mha`/`.mhd` + `.raw`, `.nrrd`), an image series (choose files or a folder, then enter pixel size, slice spacing and window), or give a path on this PC. See [Importing a CT](#importing-a-ct).
- **Fly-through, look around:** the recorded MP4 cannot be rotated. The Fly-through tab's *Interactive fly-through* button replays the route live in the 3D view: drag to look in any direction, scroll to zoom the lens, double-click or *Reset look* to recentre, *Look back* to turn around, arrow keys and space when the view is focused. The look direction is an offset from the route direction, so it stays put while the camera moves; play and pause, speed 2-30 mm/s.
- **Outputs** go to `%USERPROFILE%\bronchnav-jobs\<run id>\` (override with `JOBS_DIR`), outside the Drive folder. Earlier runs reappear in the list after a restart. AeroPath is read from `AEROPATH_ROOT` (default `%USERPROFILE%\datasets\aeropath\extracted\AeroPath`).
- A phantom run takes about 15 s, an AeroPath run 15-40 s plus video.
- The 3D view loads three.js from a CDN, so the page needs internet once. This is a developer tool, not the Phase 4 app: no login, never expose it beyond localhost or a tailnet (imports are refused on a proxied URL).
- Code: `server/app/main.py` (jobs API), `server/app/static/index.html` (page), `core/scripts/plan_job.py` (one job, run as a subprocess).

### Public demo over Tailscale Funnel (passcode, public data only)

Example setup: **`https://<your-node>.<your-tailnet>.ts.net:8443` put on the public internet through Tailscale Funnel**, behind a passcode.

```bash
tailscale funnel --bg --https=8443 http://127.0.0.1:8000    # publish (public internet)
tailscale serve status                                       # ":8443 (Funnel on)"
tailscale funnel --https=8443 off                            # stop publishing
tailscale serve --bg --https=8443 http://127.0.0.1:8000     # tailnet-only instead (still passcode + public data only)
```

How access works (`server/app/main.py`). The tier is decided only by whether a request came through a proxy (Funnel and `serve` add forwarding headers), never by Tailscale identity headers, because a client could forge those on a public Funnel:

| Tier | Who | Can do |
|---|---|---|
| local | a direct request from this PC (`http://127.0.0.1:8000`) | everything: run jobs on phantom/AeroPath/imports, import scans, view logs |
| public | anything through the proxy, including your own phone on the tailnet | sign in with passcode, then **interactive demo on public data**: pick the synthetic phantom or any of the AeroPath cases, scroll the CT and pick or place a nodule, run planning, view 3D airway tree with ranked alternative branches, preview replanned paths, and save edited runs |

For the public tier: only public data is available, the synthetic phantom and the AeroPath CTs (CC BY 4.0; attribution in the page footer). There are no imports or local paths, no job logs, and the raw volumes and airway masks cannot be downloaded: visitors get windowed CT slices, the route, the 3D mesh, stills and video. Opening an AeroPath CT in the viewer is capped at 20 loads per 10 minutes per client. Runs are private to the session and deleted after 12 hours. Public limits: 1 active run per session, no hourly cap (set `PUBLIC_MAX_PER_HOUR` in `server/app/main.py` to turn one on), max 3 active runs globally, 7-minute execution timeout, and max 30 kept public runs. With no passcode configured the public tier is closed (HTTP 503), never open. Sign-ins are rate limited (8 wrong attempts per 10 minutes per client), sessions last 12 hours and end when the server restarts or the passcode changes.

- **Passcode:** the first line of `%USERPROFILE%\bronchnav-demo-passcode.txt` (outside the Drive folder; override with `DEMO_PASSCODE` or `DEMO_PASSCODE_FILE`; at least 8 characters). Edit the file to change it; it takes effect immediately and signs everyone out.
- The console must be running (`python server/run_local.py`) on this PC, and the PC must be on, for the URL to work. Anyone can read what the visitors see: only public-data runs, with a CC BY 4.0 attribution in the footer.
- Visitors need internet for the 3D view (three.js CDN). Pre-run cases (locally) so the Runs list has content. Videos are 4-6 MB each and are served from this PC's connection.
- Check `tailscale serve status` before and after: a Funnel entry that already exists on another port (for example 443 to `127.0.0.1:3000`) stays public, and anything that starts listening behind it is reachable by anyone.

## Run the stack (needs Docker)

```bash
docker compose up --build
```

Open http://localhost:5173. API health: http://localhost:8000/api/health.
With an NVIDIA GPU (needs the NVIDIA Container Toolkit): `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`.

Not yet exercised: Docker is not installed on the dev machine.

## Web dev without Docker

Run `npm install && npm run dev` in `web/` from a local (non-Drive) copy of the folder.

## Second-opinion reviews with agy

The Antigravity CLI (`agy`) was used for read-only code reviews. Headless mode cannot approve tool permissions, so paste the source into the prompt (chunks under about 20 KB), say it is pasted text and not to use tools, and prefer `gemini-4-argon-medium` (`gemini-3.8-flash-low` was noisy). Verify every claim against the code before acting. The Argon review found a real inverted-normals bug and a resampling bug; see PLAN.md Phase 1 status.
