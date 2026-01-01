import os
os.environ['MAVLINK_DIALECT'] = 'ardupilotmega'
import time
import numpy as np
from pymavlink import mavutil
from ultralytics import YOLO
import cv2
import threading
from datetime import datetime

# --- Embedded RC Controller ---
class DroneRCController:
    def __init__(self, connection_string, baud_rate=57600, max_loop_rate=50):
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.max_loop_rate = max_loop_rate
        self.master = None
        self.last_heartbeat_time = 0
        self.connected = False
        self.last_channels = {} 
        self._stop_event = threading.Event()
        self._heartbeat_thread = None

    def connect(self):
        try:
            print(f"[RC] Connecting to {self.connection_string}...")
            if "udpin" in self.connection_string:
                 self.master = mavutil.mavlink_connection(self.connection_string)
            else:
                 self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate)
            msg = self.master.wait_heartbeat(timeout=15)
            if msg:
                self.connected = True
                self.last_heartbeat_time = time.time()
                print(f"[RC] Connected! IDs: {self.master.target_system}, {self.master.target_component}")
                if not self._heartbeat_thread or not self._heartbeat_thread.is_alive():
                    self._stop_event.clear()
                    self._heartbeat_thread = threading.Thread(target=self._heartbeat_sender, daemon=True)
                    self._heartbeat_thread.start()
                return True
            else:
                print("[RC] Connection timed out.")
                return False
        except Exception as e:
            print(f"[RC] Error: {e}")
            return False

    def _heartbeat_sender(self):
        while not self._stop_event.is_set():
            if self.master:
                try:
                    self.master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                except: pass
            time.sleep(1.0)

    def check_health(self):
        now = time.time()
        msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
        if msg:
            self.last_heartbeat_time = now
            self.connected = True
        if now - self.last_heartbeat_time > 5.0:
            self.connected = False
            if now - self.last_heartbeat_time > 8.0:
                 print("[RC] Reconnecting...")
                 if self.connect(): self.last_heartbeat_time = now
                 else: self.last_heartbeat_time = now - 5.0
        return self.connected

    def get_channel_value(self, channel_id, channels_msg=None):
        if channels_msg:
            attr_name = f"chan{channel_id}_raw"
            if hasattr(channels_msg, attr_name):
                val = getattr(channels_msg, attr_name)
                if 800 < val < 2200:
                    self.last_channels[channel_id] = val
                    return val
        return self.last_channels.get(channel_id, 0)

    def is_channel_high(self, value, threshold=1500):
        return value >= threshold

    def run_loop(self, callback):
        print("[RC] Starting loop...")
        last_loop_time = time.time()
        while True:
            now = time.time()
            dt = now - last_loop_time
            if dt < 1.0 / self.max_loop_rate:
                time.sleep(0.001)
                continue
            last_loop_time = now
            self.check_health()
            rc_msg = self.master.recv_match(type='RC_CHANNELS', blocking=False)
            if not callback(self, rc_msg, dt): break

    def close(self):
        self._stop_event.set()
        if self._heartbeat_thread: self._heartbeat_thread.join(timeout=1.0)
        if self.master: self.master.close()

# --- Configuration (Optimized for Jetson Orin Nano) ---
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 57600
TARGET_ALTITUDE = 6.0      
CRUISE_ALTITUDE = 10.0
DESCEND_SPEED = 0.40        
ASCENT_SPEED = 0.8
HOLD_DURATION = 12         
INSPECTION_COOLDOWN = 1.0    
MAX_INSPECTIONS = 2
PID_KP = 0.6               
PID_KI = 0.04              
PID_KD = 0.18              
DEADZONE = 30              
VEL_SCALE = 120.0          
YAW_SCALE = 250.0          
SMOOTHING = 0.65           
MODEL_FILE = "best.pt"
INFERENCE_SIZE = 640
MAX_LOOP_RATE = 30.0       # Higher rate for Jetson
DETECT_DIR = "detected_frames"
SERVO_CHANNELS = [9, 10, 11, 12, 13]
SERVO_HIGH_PWM = 1900
SERVO_NEUTRAL_PWM = 1100
ROTATE_DURATION = 0.5

os.makedirs(DETECT_DIR, exist_ok=True)

# --- Utilities ---
class PID:
    def __init__(self, kp, ki, kd, limit=1.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit 
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def compute(self, error, current_time):
        if self.last_time is None:
            dt = 0.033
        else:
            dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0: dt = 0.033
        self.integral += error * dt
        self.integral = max(min(self.integral, 5.0), -5.0) 
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        if np.isnan(output) or np.isinf(output): return 0.0
        return max(min(output, self.limit), -self.limit)

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

def gstreamer_pipeline():
    """Hardware Accelerated (NVDEC) for Jetson Orin Nano"""
    return (
        f"rtspsrc location={RTSP_URL} latency=41 udp-reconnect=1 timeout=5000000000 do-retransmission=false ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1 sync=false max-buffers=1"
    )

# --- MAVLink Actions ---
def send_body_velocity(master, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111, 
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate
    )

def set_mode(master, mode_id):
    master.mav.set_mode_send(master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, int(mode_id))

def trigger_payload_drop_async(master, channel):
    def _drop():
        print(f"[SERVO] Activating AUX (channel {channel})")
        master.mav.command_long_send(master.target_system, master.target_component, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, channel, SERVO_HIGH_PWM, 0, 0, 0, 0, 0)
        time.sleep(ROTATE_DURATION)
        master.mav.command_long_send(master.target_system, master.target_component, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, channel, SERVO_NEUTRAL_PWM, 0, 0, 0, 0, 0)
    threading.Thread(target=_drop, daemon=True).start()

# --- Main Application ---
def main():
    print("Connecting to vehicle...")
    rc_controller = DroneRCController(SERIAL_PORT, baud_rate=BAUD_RATE, max_loop_rate=MAX_LOOP_RATE)
    if not rc_controller.connect(): return
    master = rc_controller.master

    modes = {
        "AUTO": master.mode_mapping().get('AUTO', 3),
        "GUIDED": master.mode_mapping().get('GUIDED', 4),
        "RTL": master.mode_mapping().get('RTL', 10)
    }

    print(f"Loading YOLO model: {MODEL_FILE}")
    model = YOLO(MODEL_FILE)
    model.overrides.update({'conf': 0.5, 'classes': [0], 'verbose': False})

    pid_x = PID(PID_KP, PID_KI, PID_KD)
    pid_y = PID(PID_KP, PID_KI, PID_KD)
    smooth_px = smooth_py = None

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return

    state = {
        "active": False, "mode": "MONITORING",
        "inspections": 0, "wp2_reached": False,
        "gps": {"alt": 0.0},
        "timers": {"last_ins": 0, "start_ins": 0, "last_seen": 0},
        "flags": {"dropped": False}
    }

    def mission_callback(controller, rc_msg, dt):
        master = controller.master
        now = time.time()
        
        msg_pos = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg_pos: state["gps"]["alt"] = msg_pos.relative_alt / 1000.0

        msg_wp = master.recv_match(type='MISSION_CURRENT', blocking=False)
        if msg_wp and not state["wp2_reached"] and msg_wp.seq > 2:
            print("[MISSION] Zone Reached (WP2 complete)")
            state["wp2_reached"] = True

        ch10 = controller.get_channel_value(10, rc_msg)
        if controller.is_channel_high(ch10, 1500):
            if not state["active"]:
                print("[SYSTEM] Activated")
                state.update({"active": True, "mode": "MONITORING"})
        else:
            if state["active"]:
                print("[SYSTEM] Stopped")
                state["active"] = False
                set_mode(master, modes["AUTO"])

        if not state["active"]:
            cap.grab()
            return True

        ret, frame = cap.read()
        if not ret: return True
        
        results = model(frame, imgsz=INFERENCE_SIZE, half=True)
        annotated = results[0].plot()
        
        person_detected = False
        px = py = None
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        if results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_px, raw_py = (x1 + x2) // 2, (y1 + y2) // 2
                    if smooth_px is None: smooth_px, smooth_py = raw_px, raw_py
                    else:
                        nonlocal smooth_px, smooth_py
                        smooth_px = SMOOTHING * smooth_px + (1 - SMOOTHING) * raw_px
                        smooth_py = SMOOTHING * smooth_py + (1 - SMOOTHING) * raw_py
                    px, py = smooth_px, smooth_py
                    state["timers"]["last_seen"] = now
                    break
        else:
            if now - state["timers"]["last_seen"] > 1.0:
                smooth_px = smooth_py = None

        vf = vr = vz = yr = 0.0

        if state["mode"] == "MONITORING":
            if person_detected and state["wp2_reached"] and state["inspections"] < MAX_INSPECTIONS and (now - state["timers"]["last_ins"] > INSPECTION_COOLDOWN):
                print("[MISSION] Target spotted -> DESCENDING")
                set_mode(master, modes["GUIDED"])
                cv2.imwrite(f"{DETECT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", annotated)
                state["mode"] = "DESCENDING"
                state["timers"]["start_ins"] = now
                state["flags"]["dropped"] = False

        elif state["mode"] in ["DESCENDING", "HOLDING"]:
            cv2.imwrite(f"{DETECT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", annotated)
            if person_detected and px is not None:
                err_x = px - cx
                err_y = py - cy
                if abs(err_x) > DEADZONE:
                    alt_factor = max(0.5, state["gps"]["alt"] / 10.0) 
                    vr = (pid_x.compute(err_x, now) / VEL_SCALE) * alt_factor
                    yr = (pid_x.compute(err_x, now) / YAW_SCALE) * alt_factor
                if abs(err_y) > DEADZONE:
                    alt_factor = max(0.5, state["gps"]["alt"] / 10.0)
                    vf = (-pid_y.compute(err_y, now) / VEL_SCALE) * alt_factor

            if state["mode"] == "DESCENDING":
                vz = DESCEND_SPEED
                if state["gps"]["alt"] <= TARGET_ALTITUDE + 0.3:
                    print("[MISSION] Reached Hold Altitude")
                    state["mode"] = "HOLDING"
                    state["timers"]["start_ins"] = now
            else:
                vz = 0.0
                remaining = int(HOLD_DURATION - (now - state["timers"]["start_ins"]))
                cv2.putText(annotated, f"HOLD: {max(0, remaining)}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if not state["flags"]["dropped"]:
                    trigger_payload_drop_async(master, SERVO_CHANNELS[state["inspections"]])
                    state["flags"]["dropped"] = True

                if now - state["timers"]["start_ins"] > HOLD_DURATION:
                    print("[MISSION] Hold complete -> Returning to AUTO")
                    state["inspections"] += 1
                    state["timers"]["last_ins"] = now
                    if state["inspections"] >= MAX_INSPECTIONS:
                        print("[MISSION] Max inspections -> RTL")
                        set_mode(master, modes["RTL"])
                        return False
                    else:
                        set_mode(master, modes["AUTO"])
                        state["mode"] = "MONITORING"
            send_body_velocity(master, vf, vr, vz, yr)

        cv2.putText(annotated, f"ALT: {state['gps']['alt']:.1f}m | MODE: {state['mode']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.circle(annotated, (cx, cy), DEADZONE, (255, 255, 255), 2)
        print(f"[MISSION] ALT: {state['gps']['alt']:.1f}m | MODE: {state['mode']}")
         
        # cv2.imshow("Jetson Mission Monitor", annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'): return False
        # return True

    try:
        rc_controller.run_loop(mission_callback)
    except KeyboardInterrupt: pass
    finally:
        rc_controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
