
import os
import time
import cv2
from ultralytics import YOLO
from threading import Thread, Lock
from flask import Flask, render_template, Response, jsonify
from waitress import serve
from collections import deque


MODEL_PATH = "/home/pi/kapalASVtoRaspberry/model/best.pt"
CONF = 0.18
CAM_W, CAM_H = 640, 480
IMG_SZ = 256
PROCESS_EVERY_N_FRAMES = 1


TRACKING_TTL = 12
CENTER_MARGIN = 42
GATE_WIDTH_PASS_THRESHOLD = CAM_W * 0.75
PASSING_DURATION = 2.0
SEARCH_SWEEP_DURATION = 2.0
SMOOTH_HISTORY = 5
PERSISTENCE_REQUIRE = 3
RECOVERY_GRACE_S = 1.0
FAILSAFE_TIMEOUT = 15.0


STEER_SMALL = "belok_kecil"
STEER_MED = "belok_sedang"
STEER_LARGE = "belok_besar"


COMMAND_COOLDOWN = 0.2
SERIAL_PORT, SERIAL_BAUD, SERIAL_ENABLED = "/dev/ttyUSB0", 9600, True


HOST = "0.0.0.0"
PORT = 2045
TEAM_NAME = "Garuda Layang"
TRACK_INFO = "Merah Kiri - Hijau Kanan"
SHOW_OVERLAY = False


LOG_PATH = "/tmp/garudalayang_v3.0.log"


output_frame_lock = Lock()
telemetry_lock = Lock()
output_frame = None
telemetry = {"lat": "-", "lon": "-", "dir": "0", "cmd": "stop", "state": "INIT"}
ser = None


state = "SEARCHING"
lane_orientation = "RED_LEFT"  
last_turn_direction = "kiri"
state_transition_time = 0
next_turn_hint = "kanan"

det = {
    "bolamerah": {"seen": 0, "last_seen": 0.0, "center_x": None, "box": None},
    "bolahijau": {"seen": 0, "last_seen": 0.0, "center_x": None, "box": None},
}
gate_history = deque(maxlen=SMOOTH_HISTORY)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    s = f"[{ts}] {msg}"
    try:
        with open(LOG_PATH, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass
    print(s)

def safe_ser_write(cmd_str):
    global ser
    if not SERIAL_ENABLED or ser is None:
        return
    try:
        ser.write((cmd_str + "\n").encode("utf-8"))
    except Exception as e:
        log(f"[SERIAL ERROR] {e}")

class CameraStream:
    def __init__(self, src=0, width=CAM_W, height=CAM_H):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.stream.isOpened():
            raise IOError(f"Tidak bisa membuka kamera di indeks {src}")
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.stream.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.stream.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.stream.set(cv2.CAP_PROP_CONTRAST, 0.5)
        self.stream.set(cv2.CAP_PROP_SATURATION, 0.5)
       

        self.stopped = False
        _, self.frame = self.stream.read()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frm = self.stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            self.frame = frm

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        try:
            self.stream.release()
        except:
            pass


def adjust_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    target = 128
    gain = target / (mean_val + 1e-5)
    frame_adj = cv2.convertScaleAbs(frame_eq, alpha=gain, beta=0)
    return frame_adj

def serial_reader():
    global ser, telemetry
    if not SERIAL_ENABLED:
        log("[SERIAL] Disabled")
        return
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        log(f"[SERIAL] Terhubung ke {SERIAL_PORT}")
    except Exception as e:
        log(f"[SERIAL] Gagal koneksi: {e}")
        ser = None
        return
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            if "LAT:" in line:
                parts = [p.strip() for p in line.split(',')]
                lat = parts[0].split(':')[1] if ":" in parts[0] else "-"
                lon = parts[1].split(':')[1] if len(parts) > 1 and ":" in parts[1] else "-"
                dirc = parts[2].split(':')[1] if len(parts) > 2 and ":" in parts[2] else telemetry.get("dir", "0")
                with telemetry_lock:
                    telemetry["lat"], telemetry["lon"], telemetry["dir"] = lat, lon, dirc
        except Exception:
            time.sleep(0.05)


def mark_seen(name, box, cx):
    det[name]["seen"] += 1
    det[name]["last_seen"] = time.time()
    det[name]["center_x"] = cx
    det[name]["box"] = box

def mark_missed(name):
    if time.time() - det[name]["last_seen"] > 0.5:
        det[name]["seen"] = max(0, det[name]["seen"] - 1)
        if det[name]["seen"] == 0:
            det[name]["center_x"] = None
            det[name]["box"] = None

def is_valid(name):
    return det[name]["seen"] >= PERSISTENCE_REQUIRE

def smooth_gate_center():
    if not gate_history:
        return None, 0.0
    xs = [g[0] for g in gate_history]
    ts = [g[1] for g in gate_history]
    s = sum(xs) / len(xs)
    vel = (xs[-1] - xs[0]) / (ts[-1] - ts[0] if ts[-1] != ts[0] else 1)
    return s, vel

def steering_from_norm_error(norm_error):
    ae = abs(norm_error)
    if ae < 0.12: return "maju"
    if ae < 0.30: return STEER_SMALL
    if ae < 0.55: return STEER_MED
    return STEER_LARGE


def run_detection():
    global output_frame, telemetry, ser, state, lane_orientation, last_turn_direction, state_transition_time, next_turn_hint

    model = YOLO(MODEL_PATH)
    vs = CameraStream(src=0, width=CAM_W, height=CAM_H).start()
    time.sleep(1.2)
    log("[INFO] YOLO & Kamera aktif")

    tracked = {}
    frame_count = 0
    colors = {"bolamerah": (0, 0, 255), "bolahijau": (0, 255, 0)}
    last_send_time = 0.0
    latest_cmd = "stop"
    search_start_time = time.time()
    last_any_time = time.time()

    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame = adjust_lighting(frame)
        if frame_count % 30 == 0:
            avg_lum = frame.mean()
            log(f"[LIGHT] Avg luminance: {avg_lum:.1f}")

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = model.predict(frame, conf=CONF, verbose=False, imgsz=IMG_SZ)
            tracked.clear()
            seen_this = {"bolamerah": False, "bolahijau": False}
            for r in results:
                for b in r.boxes:
                    cls = int(b.cls[0])
                    name = model.names.get(cls, "unknown").lower()
                    if name not in ("bolamerah", "bolahijau"):
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    if y2 < CAM_H * 0.28:
                        continue
                    cx = x1 + (x2 - x1)//2
                    tracked[name] = {"box": (x1, y1, x2, y2), "center_x": cx}
                    mark_seen(name, (x1, y1, x2, y2), cx)
                    seen_this[name] = True
                    last_any_time = time.time()
            for nm in ("bolamerah", "bolahijau"):
                if not seen_this[nm]:
                    mark_missed(nm)

        red_valid = is_valid("bolamerah")
        green_valid = is_valid("bolahijau")
        both = red_valid and green_valid
        one = (red_valid or green_valid) and not both
        none = not (red_valid or green_valid)

        if time.time() - last_any_time > FAILSAFE_TIMEOUT:
            log("[FAILSAFE] Tidak ada deteksi lama -> STOP")
            with telemetry_lock:
                telemetry["cmd"] = "stop"; telemetry["state"] = "FAILSAFE"
            safe_ser_write("stop")
            time.sleep(1.0)
            last_any_time = time.time()
            continue

        frame_center = CAM_W // 2
        cmd = "stop"

        if state == "SEARCHING":
            if both or one:
                state = "APPROACHING"
                log("SEARCHING -> APPROACHING")
                continue
            elapsed = time.time() - search_start_time
            if elapsed < SEARCH_SWEEP_DURATION:
                cmd = "belok_kiri" if next_turn_hint == "kiri" else "belok_kanan"
            else:
                search_start_time = time.time()
                cmd = "maju_pelan"

        elif state == "APPROACHING":
            if none:
                last_seen = max(det["bolamerah"]["last_seen"], det["bolahijau"]["last_seen"])
                if time.time() - last_seen < RECOVERY_GRACE_S:
                    cmd = "maju_pelan"
                else:
                    state = "SEARCHING"
                    search_start_time = time.time()
                    log("Lost -> SEARCHING")
                    continue
            elif both:
                rx = det["bolamerah"]["center_x"]
                gx = det["bolahijau"]["center_x"]
                if rx is None or gx is None:
                    cmd = "maju_pelan"
                else:
                    gate_center = (rx + gx) / 2.0
                    gate_width = abs(rx - gx)
                    gate_history.append((gate_center, time.time()))
                    smooth, vel = smooth_gate_center()
                    if smooth is None: smooth = gate_center
                    predicted = smooth + vel * 0.12
                    err_px = predicted - frame_center
                    norm_err = err_px / (CAM_W / 2.0)
                    cmd = steering_from_norm_error(norm_err)
                    if gate_width > GATE_WIDTH_PASS_THRESHOLD * 0.82:
                        cmd = "maju_pelan"
                    if gate_width > GATE_WIDTH_PASS_THRESHOLD and abs(norm_err) < 0.28:
                        state = "PASSING"
                        state_transition_time = time.time()
                        log("APPROACHING -> PASSING")
                        continue
            elif one:
                cmd = "belok_kecil_kanan" if red_valid else "belok_kecil_kiri"

        elif state == "PASSING":
            cmd = "maju"
            if time.time() - state_transition_time > PASSING_DURATION:
                gate_history.clear()
                state = "SEARCHING"
                search_start_time = time.time()
                log("PASSING -> SEARCHING")
                continue

        if "kiri" in cmd: last_turn_direction = "kiri"
        elif "kanan" in cmd: last_turn_direction = "kanan"

        now = time.time()
        if now - last_send_time > COMMAND_COOLDOWN and cmd != "stop":
            with telemetry_lock:
                telemetry["cmd"], telemetry["state"] = cmd, state
            safe_ser_write(cmd)
            last_send_time = now
            log(f"[CMD] {state:<10} | {cmd:<16} | ori={lane_orientation}")

        if SHOW_OVERLAY:
            for name in tracked:
                x1, y1, x2, y2 = tracked[name]["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[name], 2)
            cv2.line(frame, (CAM_W//2, 0), (CAM_W//2, CAM_H), (255, 255, 0), 1)
            cv2.putText(frame, f"{state} {cmd}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        with output_frame_lock:
            output_frame = frame.copy()

        frame_count += 1


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", team_name=TEAM_NAME, track_info=TRACK_INFO)

def generate_video_stream():
    while True:
        with output_frame_lock:
            if output_frame is None:
                time.sleep(0.02)
                continue
            flag, encoded = cv2.imencode(".jpg", output_frame)
            if not flag:
                time.sleep(0.02)
                continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/telemetry")
def get_telemetry():
    with telemetry_lock:
        return jsonify(telemetry)

if __name__ == '__main__':
    try:
        open(LOG_PATH, "a").close()
        log("=== START Garuda Layang ===")

        Thread(target=serial_reader, daemon=True).start()
        Thread(target=run_detection, daemon=True).start()
        log(f"Server aktif di http://{HOST}:{PORT}")

        serve(app, host=HOST, port=PORT, threads=8)

    except KeyboardInterrupt:
        log("Shutdown oleh user.")
    except Exception as e:
        log(f"[FATAL] {e}")
    finally:
        safe_ser_write("stop")
        if ser and getattr(ser, "is_open", False):
            try:
                ser.close()
            except:
                pass
        log("=== STOP Garuda Layang ===")

