#!/usr/bin/env python3

import time
import numpy as np
from pymavlink import mavutil
from ultralytics import YOLO
from rc_controller import DroneRCController
import cv2
import os
import threading
from datetime import datetime

# --- Configuration (Merged from Reference) ---
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
TARGET_ALTITUDE = 5.0      # Reference: 5.0m
CRUISE_ALTITUDE = 10.0
DESCEND_SPEED = 0.8        # Reference: 0.8m/s
ASCENT_SPEED = 0.8
HOLD_DURATION = 15         # Reference: 15s
INSPECTION_COOLDOWN = 3    # Reference: 3s
MAX_INSPECTIONS = 2
PID_KP = 0.6               # Proportional
PID_KI = 0.04              # Integral (New)
PID_KD = 0.18              # Derivative
DEADZONE = 30              # Reference: 30px
VEL_SCALE = 120.0          # Baseline at 10m
YAW_SCALE = 250.0          
SMOOTHING = 0.65           # Smoothing factor (New)

PERSON_TIMEOUT = 12.0
MODEL_FILE = "best.pt"
INFERENCE_SIZE = 640
CONF_THRESHOLD = 0.50      # Reference: 0.50
MAX_LOOP_RATE = 20.0       # Higher rate for better tracking
DETECT_DIR = "detected_frames"
SERVO_CHANNELS = [9, 10, 11, 12, 13]
SERVO_HIGH_PWM = 1900
SERVO_NEUTRAL_PWM = 1100
ROTATE_DURATION = 0.5

os.makedirs(DETECT_DIR, exist_ok=True)

# --- Utilities (Reference Style) ---

class PID:
    def __init__(self, kp, ki, kd, limit=1.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit # Output limit
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

        # Integral with Anti-Windup (simple clamping)
        self.integral += error * dt
        self.integral = max(min(self.integral, 5.0), -5.0) 

        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        if np.isnan(output) or np.isinf(output):
            return 0.0
        
        # Clamp output
        return max(min(output, self.limit), -self.limit)

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

def gstreamer_pipeline():
    """Laptop (CPU-based) decoding with user's appsink flags"""
    return (
        f"rtspsrc location={RTSP_URL} latency=41 ! "
        "rtph264depay ! h264parse ! decodebin ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1 sync=false max-buffers=1"
    )

# --- MAVLink Actions ---

def send_body_velocity(master, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
    """Sends velocity commands using user's preferred MAV_FRAME_BODY_OFFSET_NED."""
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111, # Velocity + YawRate mask
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate
    )

def set_mode(master, mode_id):
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        int(mode_id)
    )

def trigger_payload_drop_async(master, channel):
    def _drop():
        print(f"[SERVO] Activating AUX (channel {channel})")
        # Rotate to HIGH
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, SERVO_HIGH_PWM, 0, 0, 0, 0, 0
        )
        time.sleep(ROTATE_DURATION)
        # Return to NEUTRAL
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, SERVO_NEUTRAL_PWM, 0, 0, 0, 0, 0
        )
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
    
    # Target smoothing memory
    smooth_px = None
    smooth_py = None


    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return

    state = {
        "active": False,
        "mode": "MONITORING",
        "inspections": 0,
        "wp2_reached": False,
        "gps": {"alt": 0.0},
        "timers": {"last_ins": 0, "start_ins": 0, "last_seen": 0},
        "flags": {"dropped": False}
    }

    def mission_callback(controller, rc_msg, dt):
        master = controller.master
        now = time.time()
        
        # 1. Update Telemetry (Altitude Only as per reference preference)
        msg_pos = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg_pos: state["gps"]["alt"] = msg_pos.relative_alt / 1000.0

        # 1b. Update Mission Progress (WP index trigger)
        msg_wp = master.recv_match(type='MISSION_CURRENT', blocking=False)
        if msg_wp and not state["wp2_reached"] and msg_wp.seq > 2:
            print("[MISSION] Zone Reached (WP2 complete)")
            state["wp2_reached"] = True

        # 2. System Activation (RC Switch)
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

        # 3. Perception
        ret, frame = cap.read()
        if not ret: return True
        
        results = model(frame, imgsz=INFERENCE_SIZE, half=True)
        annotated = results[0].plot()
        
        # Find person centers
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
                    
                    # Apply Low-Pass Smoothing
                    if smooth_px is None:
                        smooth_px, smooth_py = raw_px, raw_py
                    else:
                        smooth_px = SMOOTHING * smooth_px + (1 - SMOOTHING) * raw_px
                        smooth_py = SMOOTHING * smooth_py + (1 - SMOOTHING) * raw_py
                    
                    px, py = smooth_px, smooth_py
                    state["timers"]["last_seen"] = now
                    break
        else:
            # Reset smoothing if person lost for too long
            if now - state["timers"]["last_seen"] > 1.0:
                smooth_px = smooth_py = None


        # 4. Control Logic (Reference Implementation)
        vf = vr = vz = yr = 0.0

        if state["mode"] == "MONITORING":
            if person_detected and state["wp2_reached"] and state["inspections"] < 2 and (now - state["timers"]["last_ins"] > INSPECTION_COOLDOWN):
                print("[MISSION] Target spotted -> DESCENDING")
                set_mode(master, modes["GUIDED"])
                state["mode"] = "DESCENDING"
                state["timers"]["start_ins"] = now
                state["flags"]["dropped"] = False

        elif state["mode"] in ["DESCENDING", "HOLDING"]:
            # Logic shared for both states: Track Target
            if person_detected and px is not None:
                err_x = px - cx
                err_y = py - cy
                
                if abs(err_x) > DEADZONE:
                    # Altitude-aware scaling
                    alt_factor = max(0.5, state["gps"]["alt"] / 10.0) 
                    vr = (pid_x.compute(err_x, now) / VEL_SCALE) * alt_factor
                    yr = (pid_x.compute(err_x, now) / YAW_SCALE) * alt_factor
                if abs(err_y) > DEADZONE:
                    alt_factor = max(0.5, state["gps"]["alt"] / 10.0)
                    vf = (-pid_y.compute(err_y, now) / VEL_SCALE) * alt_factor


            if state["mode"] == "DESCENDING":
                vz = DESCEND_SPEED # Positive is DOWN in NED
                if state["gps"]["alt"] <= TARGET_ALTITUDE + 0.3:
                    print("[MISSION] Reached Hold Altitude (5.0m)")
                    state["mode"] = "HOLDING"
                    state["timers"]["start_ins"] = now
            
            else: # HOLDING
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
                    if state["inspections"] >= 2:
                        print("[MISSION] Max inspections -> RTL")
                        set_mode(master, modes["RTL"])
                        return False
                    else:
                        set_mode(master, modes["AUTO"])
                        state["mode"] = "MONITORING"

            send_body_velocity(master, vf, vr, vz, yr)

        # UI Overlay
        cv2.putText(annotated, f"ALT: {state['gps']['alt']:.1f}m | MODE: {state['mode']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.circle(annotated, (cx, cy), DEADZONE, (255, 255, 255), 2)
        cv2.imshow("Base Station Monitor", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): return False
        return True

    try:
        rc_controller.run_loop(mission_callback)
    except KeyboardInterrupt: pass
    finally:
        rc_controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()