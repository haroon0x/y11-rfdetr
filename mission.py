#!/usr/bin/env python3

import time
import numpy as np
from pymavlink import mavutil
from ultralytics import YOLO
import cv2
import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from gps_aware import TargetSelector, extract_detections_from_results

RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
TARGET_ALTITUDE = 6.0
DESCEND_SPEED = 0.40
HOLD_DURATION = 12
INSPECTION_COOLDOWN = 10
MAX_INSPECTIONS = 5
WP2_TOLERANCE = 5.0
PID_KP = 0.82
PID_KD = 0.42
DEADZONE = 65
VEL_SCALE = 78.0
YAW_SCALE = 1800.0
PERSON_TIMEOUT = 12.0
MODEL_FILE = "best.pt"
INFERENCE_SIZE = 640
CONF_THRESHOLD = 0.62
MAX_LOOP_RATE = 10.0
DETECT_DIR = "detected_frames"
RC_SURVEILLANCE_CHANNEL = 10
RC_HIGH_THRESHOLD = 1500
SERVO_CHANNELS = [9, 10, 11, 12, 13]
SERVO_HIGH_PWM = 1900
SERVO_NEUTRAL_PWM = 1100
ROTATE_DURATION = 0.5

os.makedirs(DETECT_DIR, exist_ok=True)


def gstreamer_pipeline():
    return (
        f"rtspsrc location={RTSP_URL} latency=41 udp-reconnect=1 timeout=0 do-retransmission=false ! "
        "application/x-rtp ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=2 sync=false"
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


class PID:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0
        self.last_time = None

    def compute(self, error, current_time):
        if self.last_time is None:
            dt = 0.12
        else:
            dt = max(current_time - self.last_time, 0.025)
        self.last_time = current_time
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.kd * derivative

    def reset(self):
        self.prev_error = 0.0
        self.last_time = None


def send_body_velocity(master, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate
    )


def set_mode(master, mode_id):
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        int(mode_id)
    )


def trigger_payload_drop(master, channel):
    print(f"[SERVO] Rotating AUX{channel-8} (channel {channel})")
    # Rotate to HIGH
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        SERVO_HIGH_PWM,
        0, 0, 0, 0, 0
    )
    time.sleep(ROTATE_DURATION)
    # Return to NEUTRAL
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        SERVO_NEUTRAL_PWM,
        0, 0, 0, 0, 0
    )


def download_mission(master):
    master.waypoint_request_list_send()
    mission_items = []
    start_time = time.time()

    while time.time() - start_time < 10:
        msg_item = master.recv_match(
            type='MISSION_ITEM', blocking=True, timeout=5)
        if msg_item is None:
            break
        mission_items.append(msg_item)
        if msg_item.seq == master.waypoint_count() - 1:
            break

    return mission_items


def get_waypoint_2(mission_items):
    for item in mission_items:
        if item.seq == 2:
            return item.x, item.y
    return None, None


def save_detection_frame(frame, annotated_frame, altitude, prefix="det"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_str = f"{altitude:.1f}m" if altitude is not None else "alt_unknown"
    cv2.imwrite(f"{DETECT_DIR}/{prefix}_{timestamp}_{alt_str}.png", frame)
    cv2.imwrite(f"{DETECT_DIR}/{prefix}_annotated_{timestamp}_{alt_str}.png", annotated_frame)


def save_inspection_gps_frame(annotated_frame, inspection_count, lat, lon, alt):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_str = f"{alt:.1f}m" if alt is not None else "alt_unknown"
    h = annotated_frame.shape[0]
    gps_text = f"Inspection #{inspection_count} | GPS: {lat:.7f}, {lon:.7f} | Alt: {alt:.1f}m"
    cv2.putText(annotated_frame, gps_text, (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(
        f"{DETECT_DIR}/inspection_gps_{inspection_count}_{timestamp}_{alt_str}.png", annotated_frame)




def compute_tracking_velocity(px, py, cx, cy, pid_x, pid_y, now, deadzone, vel_scale):
    vx = vy = 0.0
    if px is not None and py is not None:
        err_x = px - cx
        err_y = py - cy
        if abs(err_x) > deadzone:
            vy = pid_x.compute(err_x, now) / vel_scale
        if abs(err_y) > deadzone:
            vx = -pid_y.compute(err_y, now) / vel_scale
    return vx, vy


def main():
    print("Connecting to vehicle...")
    master = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
    master.wait_heartbeat(timeout=20)
    print("Connected.")

    AUTO_MODE = master.mode_mapping().get('AUTO', 3)
    GUIDED_MODE = master.mode_mapping().get('GUIDED', 4)
    RTL_MODE = master.mode_mapping().get('RTL', 10)

    print(f"Loading YOLO model: {MODEL_FILE}")
    model = YOLO(MODEL_FILE)
    model.overrides['conf'] = CONF_THRESHOLD
    model.overrides['classes'] = [0]

    pid_x = PID(PID_KP, PID_KD)
    pid_y = PID(PID_KP, PID_KD)

    # Initialize GPS-Aware Target Selector
    target_selector = TargetSelector()
    current_target = None  # To store the selected target for tracking

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return

    print("Stream opened. Waiting for RC Channel 10 HIGH to start system...")

    print("Stream opened. Waiting for RC Channel 10 HIGH to start system...")

    # Centralized state management
    state = {
        "system_active": False,
        "mission_state": "MONITORING",
        "inspection_count": 0,
        "reached_wp2": False,
        "wp2_pos": (None, None),  # (lat, lon)
        "current_pos": {"lat": None, "lon": None, "alt": None, "heading": 0},
        "last_inspection_time": time.time() - 60,
        "inspection_start_time": 0,
        "last_person_seen": time.time(),
        "last_loop_time": time.time(),
        "gps_frame_captured": False,
        "payload_dropped": False,
        "frame": np.zeros((480, 640, 3), np.uint8)
    }

    try:
        while True:
            now = time.time()
            if now - state["last_loop_time"] < 1.0 / MAX_LOOP_RATE:
                time.sleep(0.015)
                continue
            state["last_loop_time"] = now

            rc_msg = master.recv_match(type='RC_CHANNELS', blocking=False, timeout=0.01)
            rc_chan10 = 0
            if rc_msg:
                channels = [
                    rc_msg.chan1_raw, rc_msg.chan2_raw, rc_msg.chan3_raw, rc_msg.chan4_raw,
                    rc_msg.chan5_raw, rc_msg.chan6_raw, rc_msg.chan7_raw, rc_msg.chan8_raw,
                    rc_msg.chan9_raw, rc_msg.chan10_raw, rc_msg.chan11_raw, rc_msg.chan12_raw,
                    rc_msg.chan13_raw, rc_msg.chan14_raw, rc_msg.chan15_raw, rc_msg.chan16_raw,
                    rc_msg.chan17_raw, rc_msg.chan18_raw
                ]
                if len(channels) >= RC_SURVEILLANCE_CHANNEL:
                    rc_chan10 = channels[RC_SURVEILLANCE_CHANNEL - 1]

            if rc_chan10 > RC_HIGH_THRESHOLD:
                if not state["system_active"]:
                    print("RC Channel 10 HIGH → SYSTEM ACTIVATED")
                    state["system_active"] = True
                    state["reached_wp2"] = False
                    state["inspection_count"] = 0
                    state["mission_state"] = "MONITORING"
                    state["last_inspection_time"] = now - 60
                    state["gps_frame_captured"] = False
                    state["payload_dropped"] = False
                    target_selector.reset()
                    current_target = None
                    pid_x.reset()
                    pid_y.reset()

                    print("Downloading mission for Waypoint 2...")
                    mission_items = download_mission(master)
                    wp2_lat, wp2_lon = get_waypoint_2(mission_items)
                    state["wp2_pos"] = (wp2_lat, wp2_lon)

                    if wp2_lat is not None:
                        print(f"Waypoint 2 loaded: Lat={wp2_lat:.7f}, Lon={wp2_lon:.7f}")
                    else:
                        print("ERROR: Waypoint 2 not found!")
            else:
                if state["system_active"]:
                    print("RC Channel 10 LOW → SYSTEM STOPPED")
                    state["system_active"] = False
                    if state["mission_state"] != "MONITORING":
                        set_mode(master, AUTO_MODE)
                        send_body_velocity(master)

            if not state["system_active"]:
                cv2.putText(state["frame"], "SYSTEM OFF - Flip Ch10 HIGH to start", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("YOLO11 Person Inspection", state["frame"])
                cv2.waitKey(1)
                continue

            ret, frame = cap.read()
            if not ret:
                print("Frame read failed - reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
                time.sleep(1.5)
                continue
            state["frame"] = frame

            h, w = state["frame"].shape[:2]
            cx, cy = w // 2, h // 2

            if msg:
                state["current_pos"]["lat"] = msg.lat / 1e7
                state["current_pos"]["lon"] = msg.lon / 1e7
                state["current_pos"]["alt"] = msg.relative_alt * 0.001

            # Get Heading from VFR_HUD
            hud_msg = master.recv_match(type='VFR_HUD', blocking=False)
            if hud_msg:
                state["current_pos"]["heading"] = hud_msg.heading

            if (state["current_pos"]["lat"] and state["current_pos"]["lon"] and
                    not state["reached_wp2"] and state["wp2_pos"][0] is not None):
                distance = haversine_distance(
                    state["current_pos"]["lat"], state["current_pos"]["lon"],
                    state["wp2_pos"][0], state["wp2_pos"][1])
                if distance <= WP2_TOLERANCE:
                    state["reached_wp2"] = True
                    print(f"Reached Waypoint 2 – inspection zone active!")

            results = model(state["frame"], imgsz=INFERENCE_SIZE, half=True, verbose=False)
            annotated_frame = results[0].plot()

            detections = extract_detections_from_results(results, CONF_THRESHOLD)
            
            # Update last person seen time if any detections
            if detections:
                state["last_person_seen"] = now

            if state["mission_state"] == "MONITORING":
                if (detections and state["reached_wp2"] and
                    state["inspection_count"] < MAX_INSPECTIONS and
                        (now - state["last_inspection_time"] > INSPECTION_COOLDOWN)):
                    
                    # Select best target using GPS-aware logic
                    if (state["current_pos"]["lat"] is not None and 
                        state["current_pos"]["lon"] is not None and 
                        state["current_pos"]["alt"] is not None):
                        
                        target = target_selector.select_target(
                            detections,
                            state["current_pos"]["lat"],
                            state["current_pos"]["lon"],
                            state["current_pos"]["alt"],
                            state["current_pos"]["heading"],
                            w, h
                        )
                        
                        if target:
                            print(f"Person {state['inspection_count'] + 1}/{MAX_INSPECTIONS} selected → inspection start")
                            current_target = target
                            save_detection_frame(state["frame"], annotated_frame, state["current_pos"]["alt"], prefix="frame1_init")
                            set_mode(master, GUIDED_MODE)
                            time.sleep(0.6)
                            state["mission_state"] = "DESCENDING"
                            state["inspection_start_time"] = now
                            state["last_inspection_time"] = now
                            state["gps_frame_captured"] = False
                            state["payload_dropped"] = False

            elif state["mission_state"] == "DESCENDING":
                # Track the nearest detection during descent
                if detections:
                    # Find detection closest to screen center
                    min_dist = float('inf')
                    px, py = detections[0].px, detections[0].py # Default to first
                    
                    for det in detections:
                        dist = (det.px - cx)**2 + (det.py - cy)**2
                        if dist < min_dist:
                            min_dist = dist
                            px, py = det.px, det.py
                    
                    vx, vy = compute_tracking_velocity(px, py, cx, cy, pid_x, pid_y,
                                                    now, DEADZONE, VEL_SCALE)
                else:
                    vx, vy = 0.0, 0.0 # No detection, hold horizontal position
                
                send_body_velocity(master, vx, vy, DESCEND_SPEED, 0.0)

                if state["current_pos"]["alt"] and state["current_pos"]["alt"] <= TARGET_ALTITUDE + 0.35:
                    print("Reached Target Altitude → HOLDING")
                    save_detection_frame(state["frame"], annotated_frame, state["current_pos"]["alt"], prefix="frame2_alt")
                    state["mission_state"] = "HOLDING"

            elif state["mission_state"] == "HOLDING":
                if detections:
                    # Find detection closest to center
                    min_dist = float('inf')
                    px, py = None, None
                    for det in detections:
                        dist = (det.px - cx)**2 + (det.py - cy)**2
                        if dist < min_dist:
                            min_dist = dist
                            px, py = det.px, det.py
                            
                    if px is not None:
                        vx, vy = compute_tracking_velocity(px, py, cx, cy, pid_x, pid_y,
                                                        now, DEADZONE, VEL_SCALE)
                    else:
                        vx, vy = 0.0, 0.0
                else:
                    vx, vy = 0.0, 0.0

                send_body_velocity(master, vx, vy, 0.0, 0.0)

                if now - state["last_person_seen"] > PERSON_TIMEOUT:
                    print("Person lost → back to AUTO")
                    set_mode(master, AUTO_MODE)
                    state["last_inspection_time"] = now
                    state["mission_state"] = "MONITORING"

                if now - state["inspection_start_time"] > HOLD_DURATION:
                    if not state["payload_dropped"]:
                        if state["inspection_count"] < len(SERVO_CHANNELS):
                            current_channel = SERVO_CHANNELS[state["inspection_count"]]
                            trigger_payload_drop(master, current_channel)
                            save_detection_frame(state["frame"], annotated_frame, state["current_pos"]["alt"], prefix="frame3_drop")
                            
                            # Mark the target location as served
                            if current_target:
                                if (state["current_pos"]["lat"] is not None and 
                                    state["current_pos"]["lon"] is not None):
                                    target_selector.mark_served(state["current_pos"]["lat"], state["current_pos"]["lon"])
                                else:
                                    target_selector.mark_served(current_target.lat, current_target.lon)
                        state["payload_dropped"] = True

                    state["inspection_count"] += 1
                    print(f"Inspection {state['inspection_count']}/{MAX_INSPECTIONS} complete")

                    if (not state["gps_frame_captured"] and
                            state["current_pos"]["lat"] and state["current_pos"]["lon"]):
                        save_inspection_gps_frame(annotated_frame, state["inspection_count"],
                                                  state["current_pos"]["lat"], state["current_pos"]["lon"],
                                                  state["current_pos"]["alt"])
                        state["gps_frame_captured"] = True

                    if state["inspection_count"] >= MAX_INSPECTIONS:
                        print("Max inspections reached → RTL")
                        set_mode(master, RTL_MODE)
                        state["mission_state"] = "RTL"
                    else:
                        set_mode(master, AUTO_MODE)
                        state["last_inspection_time"] = now
                        state["mission_state"] = "MONITORING"

            status = f"System: {'ON' if state['system_active'] else 'OFF'} | State: {state['mission_state']} | Inspections: {state['inspection_count']}/{MAX_INSPECTIONS}"
            if state["system_active"] and not state["reached_wp2"]:
                status += " | Waiting for WP2"

            cv2.putText(annotated_frame, status, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("YOLO11 Person Inspection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print("\nCleaning up...")
        send_body_velocity(master)
        time.sleep(1.2)
        set_mode(master, AUTO_MODE)
        cap.release()
        cv2.destroyAllWindows()
        print("Script finished.")


if __name__ == "__main__":
    main()