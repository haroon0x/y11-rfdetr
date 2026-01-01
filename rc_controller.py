import time
from pymavlink import mavutil

class DroneRCController:
    def __init__(self, connection_string, baud_rate=115200, max_loop_rate=50):
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.max_loop_rate = max_loop_rate
        self.master = None
        self.last_heartbeat_time = 0
        self.last_sent_heartbeat = 0
        self.connected = False
        self.last_channels = {}  # {channel_id: value} persistence

    def connect(self):
        """Establish MAVLink connection and wait for first heartbeat."""
        try:
            print(f"[RC] Connecting to {self.connection_string}...")
            if "udpin" in self.connection_string:
                 self.master = mavutil.mavlink_connection(self.connection_string)
            else:
                 self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate)
            
            # Wait for first heartbeat to identify system IDs
            msg = self.master.wait_heartbeat(timeout=15)
            if msg:
                self.connected = True
                self.last_heartbeat_time = time.time()
                print(f"[RC] Connected! System ID: {self.master.target_system}, Component ID: {self.master.target_component}")
                return True
            else:
                print("[RC] Connection timed out (no heartbeat).")
                return False
        except Exception as e:
            print(f"[RC] Connection error: {e}")
            return False

    def check_health(self):
        """Validate connection health and handle automatic reconnection."""
        now = time.time()
        
        # Send heartbeat every 1s to signify we are alive
        if now - self.last_sent_heartbeat > 1.0:
            if self.master:
                try:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                    self.last_sent_heartbeat = now
                except Exception as e:
                    print(f"[RC] Failed to send heartbeat: {e}")
                    self.connected = False

        # Monitor incoming heartbeats
        msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
        if msg:
            self.last_heartbeat_time = now
            if not self.connected:
                print("[RC] Connection Restored.")
                self.connected = True

        # Timeout handling (5s without heartbeat)
        if now - self.last_heartbeat_time > 5.0:
            if self.connected:
                print("[RC] Connection LOST (heartbeat timeout).")
                self.connected = False
            
            # Attempt reconnection every 2s
            if now - self.last_heartbeat_time > 7.0: # give it a small gap
                 print("[RC] Attempting automatic reconnection...")
                 if self.connect():
                     self.last_heartbeat_time = now
                     return True
                 else:
                     self.last_heartbeat_time = now - 6.0 # retry soon
        
        return self.connected

    def get_channel_value(self, channel_id, channels_msg=None):
        """
        Safely extract channel value with persistence and validation.
        channel_id is 1-indexed (e.g., 10 for Ch10)
        """
        if channels_msg:
            # Extract raw value from RC_CHANNELS message
            attr_name = f"chan{channel_id}_raw"
            if hasattr(channels_msg, attr_name):
                val = getattr(channels_msg, attr_name)
                # Proper validation: 0 means channel data not available
                if val > 0:
                    self.last_channels[channel_id] = val
                    return val
        
        # Fallback to last known value if current is unavailable
        return self.last_channels.get(channel_id, 0)

    def is_channel_high(self, value, threshold=1700):
        """Standard switch logic: 1000-1300 (LOW), 1700-2000 (HIGH)."""
        return value >= threshold

    def run_loop(self, callback):
        """Main robust loop for the mission logic."""
        print("[RC] Starting main loop...")
        last_loop_time = time.time()
        
        while True:
            now = time.time()
            dt = now - last_loop_time
            
            # Control loop rate
            if dt < 1.0 / self.max_loop_rate:
                time.sleep(0.001)
                continue
            
            last_loop_time = now
            
            # Connection Health Check
            is_healthy = self.check_health()
            
            # Extract RC message if available
            rc_msg = self.master.recv_match(type='RC_CHANNELS', blocking=False)
            
            # Execute user callback
            if not callback(self, rc_msg, dt):
                break
                
        print("[RC] Loop terminated.")

    def close(self):
        if self.master:
            self.master.close()
