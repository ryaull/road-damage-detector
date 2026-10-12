"""Road damage reporter.

Live dashcam (/dashcam): the phone runs the model itself on the camera feed and sends
each pothole or crack it sees, with live GPS, to /api/report.
Report tab: upload road photos -> YOLOv8n finds damage -> photos with a location are reported.
Dashboard tab: every damaged spot on a map, ranked by priority
(road importance x damage severity x number of separate drives/photos that saw it),
plus the live position of dashcams that are driving right now.
"""
import hashlib
import html
import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import folium
import gradio as gr
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ---------------------------------------------------------------- settings
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
IMGSZ = int(os.environ.get("IMGSZ", "640"))     # the model was trained at 640
MAX_PHOTOS = 30                                  # keeps the free CPU Space responsive
MAX_SHOW_SIDE = 1280                             # shrink result images for the gallery
MIN_REPORT_CONF = 0.30                           # weaker detections are shown but never reported
CLUSTER_M = 25                                   # reports closer than this count as one spot

HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_ID = os.environ.get("SPACE_ID")            # set automatically on Hugging Face
REPORTS_REPO = os.environ.get("REPORTS_REPO") or (f"{SPACE_ID}-reports" if SPACE_ID else None)
USER_AGENT = f"road-damage-reporter/1.0 ({'https://huggingface.co/spaces/' + SPACE_ID if SPACE_ID else 'local test'})"

ONNX_PATH = os.environ.get("ONNX_PATH", str(Path(MODEL_PATH).with_suffix(".onnx")))
APP_DIR = Path(__file__).resolve().parent
LIVE_CONF = 0.35                                 # default confidence on the dashcam
MAX_GPS_ERROR_M = 60                             # dashcam reports with worse GPS are refused
SAME_DRIVE_M = 20                                # one drive reports the same damage once per 20 m
ACTIVE_S = 120                                   # dashcam shows as "live" for 2 min after its last ping

model = YOLO(MODEL_PATH)
CLASS_NAMES = [model.names[i] for i in sorted(model.names)]


def ensure_onnx():
    """The phone needs the model as ONNX. Export it once if it is missing."""
    if Path(ONNX_PATH).exists():
        return True
    try:
        out = YOLO(MODEL_PATH).export(format="onnx", imgsz=IMGSZ, simplify=False, dynamic=False)
        if Path(out).resolve() != Path(ONNX_PATH).resolve():
            shutil.move(out, ONNX_PATH)
        print("Exported", ONNX_PATH)
        return True
    except Exception as e:
        print("ONNX export failed, the live dashcam will not work:", e)
        return False


ONNX_OK = ensure_onnx()

# ---------------------------------------------------------------- damage + road scoring
SEVERITY = ["pothole", "alligator_crack", "transverse_crack", "longitudinal_crack"]
SEV_WEIGHT = {"pothole": 3, "alligator_crack": 2, "transverse_crack": 1, "longitudinal_crack": 1}
DAMAGE_COLOURS = {"pothole": "red", "alligator_crack": "orange",
                  "transverse_crack": "blue", "longitudinal_crack": "blue", None: "green"}

ROAD_CLASS = {
    "motorway": "Main", "motorway_link": "Main", "trunk": "Main", "trunk_link": "Main",
    "primary": "Main", "primary_link": "Main",
    "secondary": "Medium", "secondary_link": "Medium", "tertiary": "Medium", "tertiary_link": "Medium",
}
ROAD_WEIGHT = {"Main": 3, "Medium": 2, "Small": 1, "Unknown": 2}   # unknown is not pushed down
PRIORITY_COLOURS = {"High": "red", "Medium": "orange", "Low": "blue"}


def worst(classes):
    for c in SEVERITY:
        if c in classes:
            return c
    return None


def priority(score):
    return "High" if score >= 9 else "Medium" if score >= 4 else "Low"


# ---------------------------------------------------------------- location helpers
def _deg(v):
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600


def get_gps(path):
    """Latitude/longitude from the photo's EXIF data, or (None, None)."""
    try:
        gps = Image.open(path).getexif().get_ifd(0x8825)
        if not gps:
            return None, None
        lat = _deg(gps[2]) * (-1 if gps[1] == "S" else 1)
        lon = _deg(gps[4]) * (-1 if gps[3] == "W" else 1)
        if lat == 0 and lon == 0:
            return None, None
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


def distance_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_road_cache = {}
_road_lock = threading.Lock()
_last_lookup = [0.0]


def road_info(lat, lon):
    """Nearest road name and class from OpenStreetMap (Nominatim). Never raises."""
    key = (round(lat, 4), round(lon, 4))
    if key in _road_cache:
        return _road_cache[key]
    info = {"road": "", "road_type": "", "road_class": "Unknown", "area": ""}
    with _road_lock:                                   # Nominatim allows 1 request per second
        wait = 1.1 - (time.time() - _last_lookup[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 16},
                headers={"User-Agent": USER_AGENT},
                timeout=8,
            )
            j = r.json()
            addr = j.get("address", {})
            if j.get("category") == "highway":
                info["road_type"] = j.get("type", "")
                info["road"] = j.get("name") or addr.get("road", "")
                info["road_class"] = ROAD_CLASS.get(info["road_type"], "Small")
            else:
                info["road"] = addr.get("road", "")
            info["area"] = (addr.get("suburb") or addr.get("neighbourhood") or addr.get("city")
                            or addr.get("town") or addr.get("village") or "")
            _road_cache[key] = info
        except Exception as e:
            print("road lookup failed:", e)
        finally:
            _last_lookup[0] = time.time()
    return info


# ---------------------------------------------------------------- report storage
class ReportStore:
    """Reports kept in memory and in JSONL files.

    On Hugging Face (HF_TOKEN secret set) the files are pushed to a private dataset
    repo every 2 minutes, so reports survive restarts. Locally they stay in ./reports_data.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.reports = []
        self.dir = Path("reports_data")
        self.dir.mkdir(exist_ok=True)
        self.file = self.dir / f"reports-{uuid.uuid4().hex[:12]}.jsonl"
        self.scheduler = None
        self.mode = "local"

        if HF_TOKEN and REPORTS_REPO:
            try:
                from huggingface_hub import CommitScheduler, snapshot_download
                self.scheduler = CommitScheduler(
                    repo_id=REPORTS_REPO, repo_type="dataset", folder_path=self.dir,
                    path_in_repo="data", every=2, private=True, token=HF_TOKEN,
                    allow_patterns=["*.jsonl"],
                )
                remote = snapshot_download(REPORTS_REPO, repo_type="dataset", token=HF_TOKEN,
                                           local_dir="reports_remote", allow_patterns=["data/*.jsonl"])
                self._load(Path(remote) / "data")
                self.mode = "hf"
            except Exception as e:
                print("Hugging Face storage not available, using local files:", e)
                self.scheduler = None
        if self.mode == "local":
            self._load(self.dir)
        print(f"Report storage: {self.mode}, {len(self.reports)} report(s) loaded")

    def _load(self, folder):
        for f in sorted(Path(folder).glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    self.reports.append(json.loads(line))
                except Exception:
                    pass

    def has_photo(self, photo_hash):
        with self.lock:
            return any(r.get("photo_hash") == photo_hash for r in self.reports)

    def new_for_drive(self, session, lat, lon, types):
        """Damage types this drive has NOT already reported within SAME_DRIVE_M of here."""
        with self.lock:
            done = set()
            for r in reversed(self.reports[-3000:]):
                if r.get("session") == session and distance_m(r["lat"], r["lon"], lat, lon) <= SAME_DRIVE_M:
                    done.update(d["type"] for d in r["damage"])
        return [t for t in types if t not in done]

    def add(self, report):
        line = json.dumps(report, ensure_ascii=False) + "\n"
        with self.lock:
            self.reports.append(report)
            if self.scheduler:
                with self.scheduler.lock:
                    with self.file.open("a", encoding="utf-8") as f:
                        f.write(line)
            else:
                with self.file.open("a", encoding="utf-8") as f:
                    f.write(line)

    def all(self):
        with self.lock:
            return list(self.reports)


store = ReportStore()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- maps
def iframe(m, height=460):
    page = html.escape(m.get_root().render(), quote=True)
    return f'<iframe srcdoc="{page}" style="width:100%;height:{height}px;border:0;"></iframe>'


def fit(m, lats, lons):
    if max(lats) > min(lats) or max(lons) > min(lons):
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])


def photo_map(points):
    if not points:
        return ("<p><b>No map.</b> No location for these photos. Allow location access when the "
                "browser asks, or turn on location tags in your camera settings.</p>")
    m = folium.Map(location=[points[0]["lat"], points[0]["lon"]], zoom_start=16, control_scale=True)
    for p in points:
        classes = [d[0] for d in p["dets"]]
        lines = [f"<b>{html.escape(p['name'])}</b>"]
        lines += [f"{html.escape(c)} ({conf})" for c, conf in p["dets"]] or ["no damage found"]
        folium.CircleMarker(
            [p["lat"], p["lon"]], radius=9, color=DAMAGE_COLOURS[worst(classes)],
            fill=True, fill_opacity=0.8, popup=folium.Popup("<br>".join(lines), max_width=260),
        ).add_to(m)
    fit(m, [p["lat"] for p in points], [p["lon"] for p in points])
    return iframe(m)


# ---------------------------------------------------------------- report tab
def analyze(files, conf, use_phone_loc, phone_lat, phone_lon):
    if not files:
        raise gr.Error("Upload or take at least one road photo first.")
    files = [f if isinstance(f, str) else f.name for f in files][:MAX_PHOTOS]
    have_phone = bool(use_phone_loc) and phone_lat not in (None, "") and phone_lon not in (None, "")
    report_conf = max(float(conf), MIN_REPORT_CONF)

    rows, gallery, points, skipped = [], [], [], []
    totals, n_reported, n_dupes = Counter(), 0, 0
    for path in files:
        name = Path(path).name
        try:
            r = model.predict(path, imgsz=IMGSZ, conf=float(conf), device="cpu", verbose=False)[0]
        except Exception:
            skipped.append(name)            # e.g. a broken file
            continue

        lat, lon = get_gps(path)
        source = "photo GPS" if lat is not None else ""
        if lat is None and have_phone:
            lat, lon, source = round(float(phone_lat), 6), round(float(phone_lon), 6), "phone location"

        dets = [(r.names[int(b.cls)], round(float(b.conf), 2)) for b in r.boxes]
        totals.update(c for c, _ in dets)

        # auto-report: damage above the report threshold + a known location
        strong = [(c, s) for c, s in dets if s >= report_conf]
        status = "-"
        if strong and lat is not None:
            h = file_hash(path)
            if store.has_photo(h):
                status, n_dupes = "already reported", n_dupes + 1
            else:
                best = {}
                for c, s in strong:
                    best[c] = max(s, best.get(c, 0))
                rid = uuid.uuid4().hex[:10]
                store.add({
                    "id": rid, "session": rid, "source": "photo",
                    "time": now_iso(),
                    "lat": lat, "lon": lon, "location_source": source,
                    "damage": [{"type": c, "conf": s} for c, s in best.items()],
                    "worst": worst(list(best)),
                    "photo_hash": h, **road_info(lat, lon),
                })
                status, n_reported = "reported", n_reported + 1
        elif strong:
            status = "not reported (no location)"

        for c, s in dets:
            rows.append({"photo": name, "damage": c, "confidence": s,
                         "lat": lat, "lon": lon, "location from": source, "report": status})
        if not dets:
            rows.append({"photo": name, "damage": "none found", "confidence": None,
                         "lat": lat, "lon": lon, "location from": source, "report": "-"})
        if lat is not None:
            points.append({"name": name, "lat": lat, "lon": lon, "dets": dets})

        shown = Image.fromarray(r.plot()[:, :, ::-1])      # BGR -> RGB
        shown.thumbnail((MAX_SHOW_SIDE, MAX_SHOW_SIDE))
        caption = f"{name}: {len(dets)} found" + ("" if lat is not None else " (no location)")
        gallery.append((shown, caption))

    if not gallery:
        raise gr.Error("None of the files could be read. Use JPG, PNG or HEIC photos.")

    cols = ["photo", "damage", "confidence", "lat", "lon", "location from", "report"]
    df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(tempfile.mkdtemp(), "road_damage_results.csv")
    df.to_csv(csv_path, index=False)

    n_checked = len(files) - len(skipped)
    breakdown = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(totals.items())) or "nothing"
    summary = f"**{n_checked} photo(s) checked, {sum(totals.values())} detection(s):** {breakdown}."
    summary += f"\n\n**{n_reported} new report(s) saved** to the dashboard."
    if n_dupes:
        summary += f" {n_dupes} photo(s) were already reported before, so they were not counted again."
    no_loc = n_checked - len(points)
    if no_loc:
        summary += f" {no_loc} photo(s) had no location, so they cannot be reported or mapped."
    if skipped:
        summary += f" Could not read: {', '.join(skipped)}."
    return summary, gallery, df, csv_path, photo_map(points)




# ---------------------------------------------------------------- live dashcams
live = {}                      # session -> last ping (memory only)
live_lock = threading.Lock()


def active_dashcams():
    cutoff = time.time() - ACTIVE_S
    with live_lock:
        return [v for v in live.values() if v["t"] >= cutoff]


# ---------------------------------------------------------------- dashboard tab
def group_spots(reports):
    """Group reports within CLUSTER_M metres into one spot.

    "times seen" counts separate drives or photos, so one car passing a pothole
    counts once, however many video frames showed it.
    """
    spots = []
    for r in sorted(reports, key=lambda x: x["time"]):
        for s in spots:
            if distance_m(s["lat"], s["lon"], r["lat"], r["lon"]) <= CLUSTER_M:
                s["reports"].append(r)
                break
        else:
            spots.append({"lat": r["lat"], "lon": r["lon"], "reports": [r]})

    out = []
    for s in spots:
        rs = s["reports"]
        types = Counter(d["type"] for r in rs for d in r["damage"])
        w = worst(list(types))
        roads = Counter(r.get("road_class", "Unknown") for r in rs)
        road_class = max(roads, key=lambda k: (ROAD_WEIGHT[k] if k != "Unknown" else 0, roads[k]))
        name = next((r["road"] for r in rs if r.get("road")), "")
        area = next((r["area"] for r in rs if r.get("area")), "")
        seen = len({r.get("session", r.get("id")) for r in rs})
        sources = Counter(r.get("source", "photo") for r in rs)
        score = ROAD_WEIGHT[road_class] * SEV_WEIGHT.get(w, 1) * seen
        out.append({
            "priority": priority(score), "score": score,
            "road": name or "(unnamed road)", "area": area, "road class": road_class,
            "worst damage": (w or "").replace("_", " "),
            "damage seen": ", ".join(f"{k.replace('_', ' ')} x{v}" for k, v in types.most_common()),
            "times seen": seen,
            "from": ", ".join(f"{v} {k}" for k, v in sources.most_common()),
            "first seen": rs[0]["time"][:16].replace("T", " "),
            "last seen": rs[-1]["time"][:16].replace("T", " "),
            "lat": s["lat"], "lon": s["lon"],
        })
    out.sort(key=lambda x: (-x["score"], -x["times seen"]))
    return out


DASH_COLS = ["priority", "score", "road", "area", "road class", "worst damage",
             "damage seen", "times seen", "from", "first seen", "last seen", "lat", "lon"]


def dashboard():
    reports = store.all()
    spots = group_spots(reports)
    cams = active_dashcams()
    df = pd.DataFrame(spots, columns=DASH_COLS)
    csv_path = os.path.join(tempfile.mkdtemp(), "road_damage_priority_list.csv")
    df.to_csv(csv_path, index=False)

    if not spots and not cams:
        return ("**No reports yet.** Start the live dashcam, or use the Report tab with photos that have a location.",
                "<p>No damaged spots to show yet.</p>", df, csv_path)

    pcount = Counter(s["priority"] for s in spots)
    summary = (f"**{len(spots)} damaged spot(s)** from {len(reports)} report(s). "
               f"High priority: {pcount['High']}, medium: {pcount['Medium']}, low: {pcount['Low']}. "
               f"**Live dashcams now: {len(cams)}.** "
               f"Storage: {'Hugging Face dataset' if store.mode == 'hf' else 'local only (resets on restart)'}.")

    pts = [(s["lat"], s["lon"]) for s in spots] + [(c["lat"], c["lon"]) for c in cams]
    m = folium.Map(location=list(pts[0]), zoom_start=14, control_scale=True)
    for s in spots:
        pop = (f"<b>{html.escape(s['priority'])} priority</b> (score {s['score']})<br>"
               f"{html.escape(s['road'])} - {html.escape(s['road class'])} road<br>"
               f"{html.escape(s['damage seen'])}<br>"
               f"seen {s['times seen']} time(s), last {html.escape(s['last seen'])} UTC")
        folium.CircleMarker(
            [s["lat"], s["lon"]], radius=min(7 + 3 * s["times seen"], 22),
            color=PRIORITY_COLOURS[s["priority"]], fill=True, fill_opacity=0.75,
            popup=folium.Popup(pop, max_width=280),
        ).add_to(m)
    for c in cams:
        ago = int(time.time() - c["t"])
        speed = f"{c['speed'] * 3.6:.0f} km/h, " if c.get("speed") is not None else ""
        folium.Marker(
            [c["lat"], c["lon"]],
            icon=folium.Icon(color="darkblue", icon="car", prefix="fa"),
            popup=folium.Popup(f"<b>Live dashcam</b><br>{speed}{c['sent']} report(s) this drive<br>"
                               f"updated {ago} s ago", max_width=220),
        ).add_to(m)
    fit(m, [p[0] for p in pts], [p[1] for p in pts])
    return summary, iframe(m, 520), df, csv_path


# ---------------------------------------------------------------- API used by the dashcam page
class Damage(BaseModel):
    type: str
    conf: float


class LiveReport(BaseModel):
    session: str = Field(min_length=6, max_length=64)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    damage: List[Damage] = Field(min_length=1, max_length=8)


class Ping(BaseModel):
    session: str = Field(min_length=6, max_length=64)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    sent: int = 0


api = FastAPI()


@api.get("/dashcam", response_class=HTMLResponse)
def dashcam_page():
    return (APP_DIR / "dashcam.html").read_text(encoding="utf-8")


@api.get("/model.onnx")
def model_file():
    if not Path(ONNX_PATH).exists():
        return JSONResponse({"error": "model not exported"}, status_code=404)
    return FileResponse(ONNX_PATH, media_type="application/octet-stream")


@api.get("/api/config")
def config():
    return {"names": CLASS_NAMES, "imgsz": IMGSZ, "conf": LIVE_CONF,
            "min_report_conf": MIN_REPORT_CONF, "max_gps_error_m": MAX_GPS_ERROR_M,
            "same_drive_m": SAME_DRIVE_M, "model_ready": Path(ONNX_PATH).exists()}


@api.post("/api/ping")
def ping(p: Ping):
    with live_lock:
        live[p.session] = {"lat": p.lat, "lon": p.lon, "speed": p.speed, "sent": p.sent, "t": time.time()}
        for k in [k for k, v in live.items() if v["t"] < time.time() - 3600]:
            del live[k]
    return {"ok": True}


@api.post("/api/report")
def live_report(rep: LiveReport):
    if rep.accuracy is not None and rep.accuracy > MAX_GPS_ERROR_M:
        return {"status": "rejected", "reason": "GPS too weak"}
    best = {}
    for d in rep.damage:
        if d.type in SEV_WEIGHT and d.conf >= MIN_REPORT_CONF:
            best[d.type] = round(max(d.conf, best.get(d.type, 0)), 2)
    new = store.new_for_drive(rep.session, rep.lat, rep.lon, list(best))
    if not new:
        return {"status": "duplicate"}
    lat, lon = round(rep.lat, 6), round(rep.lon, 6)
    store.add({
        "id": uuid.uuid4().hex[:10], "session": rep.session, "source": "dashcam",
        "time": now_iso(), "lat": lat, "lon": lon, "location_source": "live GPS",
        "gps_accuracy_m": rep.accuracy, "speed_mps": rep.speed,
        "damage": [{"type": t, "conf": best[t]} for t in new],
        "worst": worst(new), **road_info(lat, lon),
    })
    return {"status": "saved", "types": new}


# ---------------------------------------------------------------- UI
GET_LOCATION_JS = """
async (files, conf, use_loc, lat, lon) => {
  if (!use_loc || !navigator.geolocation) return [files, conf, use_loc, null, null];
  const pos = await new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (p) => resolve(p), () => resolve(null),
      {enableHighAccuracy: true, timeout: 10000, maximumAge: 60000});
  });
  return [files, conf, use_loc,
          pos ? pos.coords.latitude : null, pos ? pos.coords.longitude : null];
}
"""

INTRO = """
# Road damage reporter
Mount your phone in the car like a dashcam: it finds **potholes** and **cracks** live and reports
each one with its GPS location. The **Authority dashboard** ranks every damaged spot by priority.

*Demo only. The model gets about 0.36 mAP50 on held-out Indian road photos, so expect misses
and false alarms (dust, road patches, shadows). A spot seen on several separate drives is more
trustworthy.*
"""

DASHCAM_LINK = """
<a href="/dashcam" target="_blank" style="display:block;text-align:center;padding:14px;border-radius:10px;
background:#ea580c;color:#fff;font-weight:700;font-size:1.1em;text-decoration:none;">
Open live dashcam (on your phone)</a>
"""

HOW_PRIORITY = """
**How priority works.** Reports within 25 m of each other count as one spot.
Score = road importance x damage severity x times seen.
"Times seen" counts separate drives or photos, so one car passing a pothole counts once,
however many video frames showed it.
Road importance (from OpenStreetMap): main road 3, medium road 2, small road 1, unknown 2.
Severity: pothole 3, alligator crack 2, other cracks 1.
High = 9 or more, Medium = 4 to 8, Low = under 4.
So one pothole on a main road is already High, while a pothole on a small lane becomes High
after 3 separate drives see it. Road data (c) OpenStreetMap contributors.
"""

NOTES = """
**Location.** The app uses the GPS inside the photo first. If the photo has none (most phone
browsers remove it when uploading), it uses your phone's current location, so take the photo
**at the damaged spot**. Photos are not stored: only the location, damage type and time are saved.
"""

with gr.Blocks(title="Road damage reporter") as demo:
    gr.Markdown(INTRO)
    gr.HTML(DASHCAM_LINK)
    with gr.Tabs():
        with gr.Tab("Authority dashboard") as dash_tab:
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                auto_in = gr.Checkbox(value=False, label="Auto-refresh every 20 s")
            dash_summary = gr.Markdown()
            dash_map = gr.HTML()
            dash_table = gr.Dataframe(label="Damaged spots, highest priority first", interactive=False)
            dash_csv = gr.File(label="Download priority list (CSV)")
            gr.Markdown(HOW_PRIORITY)
            timer = gr.Timer(20, active=False)

        with gr.Tab("Report with photos"):
            with gr.Row():
                files_in = gr.File(
                    label=f"Road photos (up to {MAX_PHOTOS}). On a phone you can take one with the camera.",
                    file_count="multiple", type="filepath", file_types=["image", ".heic", ".heif"],
                )
                with gr.Column():
                    conf_in = gr.Slider(
                        0.05, 0.9, value=0.25, step=0.05,
                        label="Confidence threshold (lower = more boxes, more false alarms)",
                    )
                    use_loc_in = gr.Checkbox(value=True, label="Use my phone's location if the photo has none")
                    run_btn = gr.Button("Detect and report", variant="primary")
            phone_lat = gr.Number(visible=False)
            phone_lon = gr.Number(visible=False)
            summary_out = gr.Markdown()
            gallery_out = gr.Gallery(label="Results", columns=3, height="auto")
            table_out = gr.Dataframe(label="Detections", interactive=False)
            csv_out = gr.File(label="Download results (CSV)")
            map_out = gr.HTML()
            gr.Markdown(NOTES)

    run_btn.click(
        analyze,
        inputs=[files_in, conf_in, use_loc_in, phone_lat, phone_lon],
        outputs=[summary_out, gallery_out, table_out, csv_out, map_out],
        js=GET_LOCATION_JS,
    )
    dash_outputs = [dash_summary, dash_map, dash_table, dash_csv]
    refresh_btn.click(dashboard, outputs=dash_outputs)
    dash_tab.select(dashboard, outputs=dash_outputs)
    demo.load(dashboard, outputs=dash_outputs)
    timer.tick(dashboard, outputs=dash_outputs)
    auto_in.change(lambda on: gr.Timer(active=on), inputs=auto_in, outputs=timer)

demo.queue(max_size=16)
# ssr_mode=False: on Hugging Face, server-side rendering would start a Node server
# on port 7860 and block our own server from starting.
app = gr.mount_gradio_app(api, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    print(f"Open http://127.0.0.1:{port}  (dashcam page: http://127.0.0.1:{port}/dashcam)")
    uvicorn.run(app, host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), port=port)
