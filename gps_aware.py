#!/usr/bin/env python3
"""
GPS-Aware Target Selection Module

This module provides functions for:
1. Converting pixel positions to estimated GPS coordinates
2. Filtering targets based on exclusion zones (already served positions)
3. Selecting targets using the Greedy Nearest-First algorithm (default)

Usage: Import into mission.py and call select_target() when multiple people are detected.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Camera field of view (degrees) - adjust based on your camera
CAMERA_FOV_HORIZONTAL = 60.0  # degrees
CAMERA_FOV_VERTICAL = 40.0    # degrees

# Exclusion radius: any detection within this radius of a served position is ignored
EXCLUSION_RADIUS = 3.5  # meters (half of 7m minimum person spacing)


@dataclass
class Detection:
    """Represents a single person detection from YOLO."""
    px: int           # Pixel X (center of bounding box)
    py: int           # Pixel Y (center of bounding box)
    confidence: float # YOLO confidence score
    box: Tuple[int, int, int, int] = field(default_factory=tuple)  # (x1, y1, x2, y2)


@dataclass
class GeoTarget:
    """A detection with estimated GPS coordinates."""
    detection: Detection
    lat: float
    lon: float
    priority_score: float = 0.0


class TargetSelector:
    """
    Manages target selection with GPS awareness.
    
    Maintains a list of served positions and provides methods for:
    - Estimating GPS from pixel position
    - Filtering out already-served targets
    - Selecting targets using Greedy Nearest-First strategy
    """
    
    def __init__(self, exclusion_radius: float = EXCLUSION_RADIUS):
        self.served_positions: List[Tuple[float, float]] = []
        self.exclusion_radius = exclusion_radius
    
    def reset(self):
        """Clear all served positions (e.g., when starting a new mission)."""
        self.served_positions = []
        print("[GPS_AWARE] Served positions cleared")
    
    def mark_served(self, lat: float, lon: float):
        """Mark a GPS position as served (payload dropped there)."""
        self.served_positions.append((lat, lon))
        print(f"[GPS_AWARE] Position marked as served: ({lat:.7f}, {lon:.7f})")
        print(f"[GPS_AWARE] Total served: {len(self.served_positions)}")
    
    def estimate_gps_from_pixel(
        self,
        px: int, py: int,
        drone_lat: float, drone_lon: float, drone_alt: float,
        drone_heading: float,
        frame_w: int, frame_h: int,
        fov_h: float = CAMERA_FOV_HORIZONTAL,
        fov_v: float = CAMERA_FOV_VERTICAL
    ) -> Tuple[float, float]:
        """
        Convert pixel position to estimated GPS coordinates.
        
        Assumes:
        - Camera is pointing straight down (nadir)
        - Drone heading affects the orientation of the camera
        
        Args:
            px, py: Pixel position (center of detection)
            drone_lat, drone_lon, drone_alt: Current drone position
            drone_heading: Drone heading in degrees (0=North, 90=East)
            frame_w, frame_h: Frame dimensions in pixels
            fov_h, fov_v: Camera field of view in degrees
        
        Returns:
            (estimated_lat, estimated_lon)
        """
        # Calculate ground coverage based on altitude and FOV
        ground_width = 2 * drone_alt * np.tan(np.radians(fov_h / 2))
        ground_height = 2 * drone_alt * np.tan(np.radians(fov_v / 2))
        
        # Offset from frame center in meters (camera frame)
        # Positive X = right, Positive Y = down (in image coordinates)
        offset_x_camera = (px - frame_w / 2) / frame_w * ground_width
        offset_y_camera = (py - frame_h / 2) / frame_h * ground_height
        
        # Rotate by drone heading to get North-East offsets
        # Heading 0 = North, 90 = East
        heading_rad = np.radians(drone_heading)
        
        # In camera frame: X is right, Y is forward
        # Rotate to world frame: X is East, Y is North
        offset_east = offset_x_camera * np.cos(heading_rad) + offset_y_camera * np.sin(heading_rad)
        offset_north = -offset_x_camera * np.sin(heading_rad) + offset_y_camera * np.cos(heading_rad)
        
        # Convert meter offsets to lat/lon
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        lat_offset = offset_north / 111320.0
        lon_offset = offset_east / (111320.0 * np.cos(np.radians(drone_lat)))
        
        estimated_lat = drone_lat + lat_offset
        estimated_lon = drone_lon + lon_offset
        
        return estimated_lat, estimated_lon

    def estimate_gps_from_pixel_wgs84(
        self,
        px: int, py: int,
        drone_lat: float, drone_lon: float, drone_alt: float,
        drone_heading: float,
        frame_w: int, frame_h: int,
        fov_h: float = CAMERA_FOV_HORIZONTAL,
        fov_v: float = CAMERA_FOV_VERTICAL
    ) -> Tuple[float, float]:
        """
        Convert pixel position to estimated GPS coordinates using WGS84 Ellipsoid.
        
        This provides higher accuracy than the flat-earth approximation.
        
        Args:
            px, py: Pixel position (center of detection)
            drone_lat, drone_lon, drone_alt: Current drone position
            drone_heading: Drone heading in degrees (0=North, 90=East)
            frame_w, frame_h: Frame dimensions in pixels
            fov_h, fov_v: Camera field of view in degrees
        
        Returns:
            (estimated_lat, estimated_lon)
        """
        # Calculate ground coverage based on altitude and FOV
        ground_width = 2 * drone_alt * np.tan(np.radians(fov_h / 2))
        ground_height = 2 * drone_alt * np.tan(np.radians(fov_v / 2))
        
        # Offset from frame center in meters (camera frame)
        offset_x_camera = (px - frame_w / 2) / frame_w * ground_width
        offset_y_camera = (py - frame_h / 2) / frame_h * ground_height
        
        # Rotate by drone heading to get North-East offsets
        heading_rad = np.radians(drone_heading)
        offset_east = offset_x_camera * np.cos(heading_rad) + offset_y_camera * np.sin(heading_rad)
        offset_north = -offset_x_camera * np.sin(heading_rad) + offset_y_camera * np.cos(heading_rad)
        
        # WGS84 Ellipsoid Constants
        a = 6378137.0  # Equatorial radius (meters)
        f = 1 / 298.257223563  # Flattening
        e2 = 2*f - f**2  # Square of eccentricity
        
        lat_rad = np.radians(drone_lat)
        sin_lat = np.sin(lat_rad)
        
        # Meridional radius of curvature (North-South)
        M = a * (1 - e2) / np.power(1 - e2 * sin_lat**2, 1.5)
        
        # Prime vertical radius of curvature (East-West)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        
        # Convert meter offsets to lat/lon degrees
        lat_offset_rad = offset_north / M
        lon_offset_rad = offset_east / (N * np.cos(lat_rad))
        
        estimated_lat = drone_lat + np.degrees(lat_offset_rad)
        estimated_lon = drone_lon + np.degrees(lon_offset_rad)
        
        return estimated_lat, estimated_lon
    
    def estimate_gps_from_pixel_dcm(
        self,
        px: int, py: int,
        drone_lat: float, drone_lon: float, drone_alt: float,
        drone_roll: float, drone_pitch: float, drone_yaw: float,
        frame_w: int, frame_h: int,
        fov_h: float = CAMERA_FOV_HORIZONTAL,
        fov_v: float = CAMERA_FOV_VERTICAL
    ) -> Tuple[float, float]:
        """
        Convert pixel position to GPS using full Direction Cosine Matrix (DCM).
        
        This accounts for drone Roll, Pitch, and Yaw (Attitude).
        """
        # 1. Convert Pixel to Camera Frame Vector (Normalized)
        ang_x = (px - frame_w / 2) / frame_w * np.radians(fov_h)
        ang_y = (py - frame_h / 2) / frame_h * np.radians(fov_v)
        
        tan_x = np.tan(ang_x)
        tan_y = np.tan(ang_y)
        
        # Body frame vector (un-rotated)
        body_vector = np.array([-tan_y, tan_x, 1.0])
        
        # 2. Construct Rotation Matrix (DCM) from Body to NED
        r = np.radians(drone_roll)
        p = np.radians(drone_pitch)
        y = np.radians(drone_yaw)
        
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        
        Rx = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])
        
        Ry = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ])
        
        Rz = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        
        # Combined R = Rz * Ry * Rx
        R_ned = Rz @ Ry @ Rx
        
        # 3. Rotate Vector to NED Frame
        ned_vector = R_ned @ body_vector
        
        if ned_vector[2] == 0:
            ned_vector[2] = 0.0001 # Prevent div by zero
            
        scale = drone_alt / ned_vector[2]
        
        offset_north = ned_vector[0] * scale
        offset_east = ned_vector[1] * scale
        
        # 4. Convert meters to Lat/Lon (WGS84 approx)
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2*f - f**2
        lat_rad = np.radians(drone_lat)
        sin_lat = np.sin(lat_rad)
        M = a * (1 - e2) / np.power(1 - e2 * sin_lat**2, 1.5)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        
        lat_offset_rad = offset_north / M
        lon_offset_rad = offset_east / (N * np.cos(lat_rad))
        
        estimated_lat = drone_lat + np.degrees(lat_offset_rad)
        estimated_lon = drone_lon + np.degrees(lon_offset_rad)
        
        return estimated_lat, estimated_lon

    def is_within_exclusion_zone(self, lat: float, lon: float) -> bool:
        """Check if a GPS position is within any exclusion zone."""
        for served_lat, served_lon in self.served_positions:
            # Use Vincenty for high-precision exclusion check
            distance = vincenty_distance(lat, lon, served_lat, served_lon)
            if distance < self.exclusion_radius:
                return True
        return False

    def select_target(
        self,
        detections: List[Detection],
        drone_lat: float, drone_lon: float, drone_alt: float,
        drone_heading: float,
        frame_w: int, frame_h: int
    ) -> Optional[GeoTarget]:
        """
        Select the NEAREST target from multiple detections (Greedy Nearest-First).
        """
        valid_targets: List[GeoTarget] = []
        
        for det in detections:
            # Estimate GPS position using WGS84 for better accuracy
            est_lat, est_lon = self.estimate_gps_from_pixel_wgs84(
                det.px, det.py,
                drone_lat, drone_lon, drone_alt,
                drone_heading,
                frame_w, frame_h
            )
            
            # Check exclusion zone
            if self.is_within_exclusion_zone(est_lat, est_lon):
                print(f"[GPS_AWARE] Detection at ({det.px}, {det.py}) is in exclusion zone - SKIPPED")
                continue
            
            # Calculate distance from frame center (lower = closer = better)
            norm_x = (det.px - frame_w / 2) / (frame_w / 2)
            norm_y = (det.py - frame_h / 2) / (frame_h / 2)
            distance_from_center = np.sqrt(norm_x**2 + norm_y**2)
            
            # Use negative distance as score (so closest has highest score)
            proximity_score = 1.0 - distance_from_center
            
            geo_target = GeoTarget(
                detection=det,
                lat=est_lat,
                lon=est_lon,
                priority_score=proximity_score
            )
            valid_targets.append(geo_target)
            print(f"[GPS_AWARE] Valid target: pixel=({det.px}, {det.py}), "
                  f"GPS=({est_lat:.7f}, {est_lon:.7f}), proximity={proximity_score:.3f}")
        
        if not valid_targets:
            print("[GPS_AWARE] No valid targets after filtering")
            return None
        
        # Sort by proximity score (highest = closest to center)
        valid_targets.sort(key=lambda t: t.priority_score, reverse=True)
        
        nearest_target = valid_targets[0]
        print(f"[GPS_AWARE] Selected target (nearest-first): GPS=({nearest_target.lat:.7f}, {nearest_target.lon:.7f}), "
              f"proximity={nearest_target.priority_score:.3f}")
        
        return nearest_target

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two GPS points in meters.
    """
    R = 6371000  # Earth's radius in meters
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def vincenty_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two GPS points using Vincenty's formulae (Ellipsoidal Earth).
    
    More accurate than Haversine for small distances and exclusion zones.
    Returns distance in meters.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0        # semi-major axis
    f = 1 / 298.257223563 # flattening
    b = 6356752.314245   # semi-minor axis
    
    phi1, L1 = np.radians(lat1), np.radians(lon1)
    phi2, L2 = np.radians(lat2), np.radians(lon2)
    
    U1 = np.arctan((1 - f) * np.tan(phi1))
    U2 = np.arctan((1 - f) * np.tan(phi2))
    L = L2 - L1
    
    sinU1, cosU1 = np.sin(U1), np.cos(U1)
    sinU2, cosU2 = np.sin(U2), np.cos(U2)
    
    lambda_val = L
    iter_limit = 100
    
    for _ in range(iter_limit):
        sin_lambda, cos_lambda = np.sin(lambda_val), np.cos(lambda_val)
        
        sin_sigma = np.sqrt(
            (cosU2 * sin_lambda) ** 2 +
            (cosU1 * sinU2 - sinU1 * cosU2 * cos_lambda) ** 2
        )
        
        if sin_sigma == 0:
            return 0.0  # Co-incident points
            
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = np.arctan2(sin_sigma, cos_sigma)
        
        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        
        # Check for division by zero (equatorial line)
        if cos_sq_alpha == 0:
            cos2_sigma_m = 0 
        else:
            cos2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha
            
        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        
        lambda_prev = lambda_val
        lambda_val = L + (1 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (
                cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)
            )
        )
        
        if abs(lambda_val - lambda_prev) < 1e-12:
            break
            
    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = B * sin_sigma * (
        cos2_sigma_m + B / 4 * (
            cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) -
            B / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos2_sigma_m ** 2)
        )
    )
    
    s = b * A * (sigma - delta_sigma)
    return s


def extract_detections_from_results(results, conf_threshold: float = 0.5) -> List[Detection]:
    """
    Extract Detection objects from YOLO results.
    """
    detections = []
    
    for r in results:
        if not r.boxes:
            continue
        for box in r.boxes:
            if int(box.cls) == 0 and float(box.conf) > conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                px = (x1 + x2) // 2
                py = (y1 + y2) // 2
                
                det = Detection(
                    px=px,
                    py=py,
                    confidence=float(box.conf),
                    box=(x1, y1, x2, y2)
                )
                detections.append(det)
    
    return detections


if __name__ == "__main__":
    # Test the GPS estimation
    selector = TargetSelector()
    
    # Simulate drone at 50m altitude, heading North
    drone_lat, drone_lon, drone_alt = 12.9716, 77.5946, 50.0
    drone_heading = 0  # North
    frame_w, frame_h = 1920, 1080
    
    # Simulate 3 detections
    detections = [
        Detection(px=960, py=540, confidence=0.85),   # Center
        Detection(px=400, py=300, confidence=0.75),   # Top-left
        Detection(px=1500, py=800, confidence=0.80),  # Bottom-right
    ]
    
    print("=" * 60)
    print("GPS-AWARE TARGET SELECTION TEST (Nearest-First)")
    print("=" * 60)
    print(f"Drone Position: ({drone_lat}, {drone_lon}) at {drone_alt}m")
    print(f"Drone Heading: {drone_heading}° (North)")
    print(f"Frame Size: {frame_w}x{frame_h}")
    print("-" * 60)
    
    # Test GPS estimation
    for i, det in enumerate(detections):
        est_lat, est_lon = selector.estimate_gps_from_pixel(
            det.px, det.py,
            drone_lat, drone_lon, drone_alt,
            drone_heading,
            frame_w, frame_h
        )
        print(f"Detection {i+1}: pixel=({det.px}, {det.py}) → GPS=({est_lat:.7f}, {est_lon:.7f})")
    
    print("-" * 60)
    
    # Test target selection
    nearest = selector.select_target(
        detections,
        drone_lat, drone_lon, drone_alt,
        drone_heading,
        frame_w, frame_h
    )
    
    if nearest:
        print(f"\nNearest Target: GPS=({nearest.lat:.7f}, {nearest.lon:.7f})")
        selector.mark_served(nearest.lat, nearest.lon)
        
        # Try again - should skip the served one
        print("\n--- After marking as served ---")
        nearest2 = selector.select_target(
            detections,
            drone_lat, drone_lon, drone_alt,
            drone_heading,
            frame_w, frame_h
        )
        if nearest2:
            print(f"Next Nearest Target: GPS=({nearest2.lat:.7f}, {nearest2.lon:.7f})")
            
    print("-" * 60)
    print("DCM TEST (Drone Tilted 10 deg Pitch)")
    dcm_lat, dcm_lon = selector.estimate_gps_from_pixel_dcm(
        px=960, py=540,
        drone_lat=drone_lat, drone_lon=drone_lon, drone_alt=drone_alt,
        drone_roll=0, drone_pitch=10, drone_yaw=0,
        frame_w=1920, frame_h=1080
    )
    print(f"Center Pixel (0 pitch) -> ({drone_lat:.7f}, {drone_lon:.7f})")
    print(f"Center Pixel (10 pitch) -> ({dcm_lat:.7f}, {dcm_lon:.7f})")
    print("DCM Logic Verification Complete.")
    print("-" * 60)
    print("DISTANCE CALCULATION TEST (Haversine vs Vincenty)")
    # Test distance between two close points (approx 5 meters apart)
    p1 = (12.9716000, 77.5946000)
    p2 = (12.9716500, 77.5946000) # Approx 5.5m North
    
    dist_hav = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    dist_vic = vincenty_distance(p1[0], p1[1], p2[0], p2[1])
    
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Haversine Distance: {dist_hav:.5f} m")
    print(f"Vincenty Distance:  {dist_vic:.5f} m")
    print(f"Difference:         {abs(dist_hav - dist_vic):.5f} m")
