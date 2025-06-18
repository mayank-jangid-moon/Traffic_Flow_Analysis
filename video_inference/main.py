import argparse
from typing import List, Dict
import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
from tqdm import tqdm
import supervision as sv
from collections import deque

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class TrafficAnalyzer:
    def __init__(self, num_zones: int, alpha: float = 0.15, fps: float = 30.0):
        self.num_zones = num_zones
        self.alpha = alpha
        self.fps = fps
        
        self.entry_counts = [0] * num_zones
        self.tracker_entered_zones = {}
        self.tracker_last_seen = {}
        self.frame_count = 0
        
        self.ema_dwell_times = [0.0] * num_zones
        self.current_dwell_times = [[] for _ in range(num_zones)]
        self.tracker_enter_time = {}
        
        self.max_dwell_times = [0.0] * num_zones
        self.min_dwell_times = [float('inf')] * num_zones
        self.total_measurements = [0] * num_zones
        
        self.current_zone_vehicles = [set() for _ in range(num_zones)]
        
    def update_analysis(self, detections_per_zone: List[sv.Detections], 
                       time_in_zone_per_zone: List[np.ndarray]) -> tuple:
        self.frame_count += 1
        current_counts = [0] * self.num_zones
        current_zone_status = {}
        
        for zone_idx, detections in enumerate(detections_per_zone):
            current_counts[zone_idx] = len(detections)
            current_active_vehicles = set()
            
            if detections.tracker_id is not None:
                for i, tracker_id in enumerate(detections.tracker_id):
                    if tracker_id not in current_zone_status:
                        current_zone_status[tracker_id] = set()
                    current_zone_status[tracker_id].add(zone_idx)
                    current_active_vehicles.add(tracker_id)
                    self.tracker_last_seen[tracker_id] = self.frame_count
                    
                    if tracker_id not in self.tracker_enter_time:
                        self.tracker_enter_time[tracker_id] = {}
                    if zone_idx not in self.tracker_enter_time[tracker_id]:
                        self.tracker_enter_time[tracker_id][zone_idx] = self.frame_count
            
            self._update_zone_ema_realtime(zone_idx, current_active_vehicles)
            
            self.current_zone_vehicles[zone_idx] = current_active_vehicles
        
        for tracker_id, current_zones in current_zone_status.items():
            if tracker_id not in self.tracker_entered_zones:
                self.tracker_entered_zones[tracker_id] = set()
            
            new_zones = current_zones - self.tracker_entered_zones[tracker_id]
            for zone_idx in new_zones:
                self.entry_counts[zone_idx] += 1
                print(f"DEBUG: Tracker {tracker_id} entered Zone {zone_idx + 1}")
            
            self.tracker_entered_zones[tracker_id].update(current_zones)
        
        old_trackers = [
            tid for tid, last_frame in self.tracker_last_seen.items()
            if self.frame_count - last_frame > 60
        ]
        
        for tid in old_trackers:
            if tid in self.tracker_enter_time:
                for zone_idx, enter_frame in self.tracker_enter_time[tid].items():
                    dwell_time = (self.frame_count - enter_frame) / self.fps
                    if dwell_time > 0.5:
                        self._update_ema_dwell_time(zone_idx, dwell_time)
            
            self.tracker_entered_zones.pop(tid, None)
            self.tracker_last_seen.pop(tid, None)
            self.tracker_enter_time.pop(tid, None)
        
        for zone_idx in range(self.num_zones):
            self.current_dwell_times[zone_idx] = []
            if len(time_in_zone_per_zone) > zone_idx:
                self.current_dwell_times[zone_idx] = time_in_zone_per_zone[zone_idx].tolist()
        
        return current_counts, self.get_traffic_stats()
    
    def _update_zone_ema_realtime(self, zone_idx: int, current_vehicles: set):
        if not current_vehicles:
            return
            
        current_dwell_times = []
        for tracker_id in current_vehicles:
            if (tracker_id in self.tracker_enter_time and 
                zone_idx in self.tracker_enter_time[tracker_id]):
                
                enter_frame = self.tracker_enter_time[tracker_id][zone_idx]
                current_dwell_time = (self.frame_count - enter_frame) / self.fps
                
                if current_dwell_time > 0.5:
                    current_dwell_times.append(current_dwell_time)
        
        if current_dwell_times:
            avg_current_dwell = np.mean(current_dwell_times)
            
            if self.frame_count % 15 == 0:
                self._update_ema_dwell_time(zone_idx, avg_current_dwell)
    
    def _update_ema_dwell_time(self, zone_idx: int, dwell_time: float):
        if self.total_measurements[zone_idx] == 0:
            self.ema_dwell_times[zone_idx] = dwell_time
        else:
            self.ema_dwell_times[zone_idx] = (
                self.alpha * dwell_time + (1 - self.alpha) * self.ema_dwell_times[zone_idx]
            )
        
        self.total_measurements[zone_idx] += 1
        
        self.max_dwell_times[zone_idx] = max(self.max_dwell_times[zone_idx], dwell_time)
        if self.min_dwell_times[zone_idx] == float('inf'):
            self.min_dwell_times[zone_idx] = dwell_time
        else:
            self.min_dwell_times[zone_idx] = min(self.min_dwell_times[zone_idx], dwell_time)
    
    def get_traffic_condition(self, zone_idx: int) -> tuple:
        ema_time = self.ema_dwell_times[zone_idx]
        
        if ema_time > 10:
            return "HIGH TRAFFIC", (0, 0, 255)
        elif ema_time > 5:
            return "MODERATE TRAFFIC", (0, 165, 255)
        else:
            return "LIGHT TRAFFIC", (0, 255, 0)
    
    def get_current_avg_dwell_time(self, zone_idx: int) -> float:
        if not self.current_zone_vehicles[zone_idx]:
            return 0.0
            
        current_dwell_times = []
        for tracker_id in self.current_zone_vehicles[zone_idx]:
            if (tracker_id in self.tracker_enter_time and 
                zone_idx in self.tracker_enter_time[tracker_id]):
                
                enter_frame = self.tracker_enter_time[tracker_id][zone_idx]
                current_dwell_time = (self.frame_count - enter_frame) / self.fps
                current_dwell_times.append(current_dwell_time)
        
        return np.mean(current_dwell_times) if current_dwell_times else 0.0
    
    def get_traffic_stats(self) -> Dict:
        return {
            'entry_counts': self.entry_counts,
            'ema_dwell_times': self.ema_dwell_times,
            'max_dwell_times': self.max_dwell_times,
            'min_dwell_times': [t if t != float('inf') else 0.0 for t in self.min_dwell_times],
            'current_vehicles': [len(times) for times in self.current_dwell_times],
            'total_measurements': self.total_measurements,
            'current_avg_dwell_times': [self.get_current_avg_dwell_time(i) for i in range(self.num_zones)]
        }


def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def draw_enhanced_statistics(frame, traffic_analyzer, zones):
    height, width = frame.shape[:2]
    stats = traffic_analyzer.get_traffic_stats()
    
    overlay = frame.copy()
    
    panel_width = 380
    panel_height = len(zones) * 130
    panel_x = width - panel_width - 20
    panel_y = 20
    
    draw_rounded_rectangle(overlay, 
                          (panel_x, panel_y), 
                          (panel_x + panel_width, panel_y + panel_height),
                          (0, 0, 0), radius=15)
    
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    header_text = "REAL-TIME TRAFFIC ANALYSIS"
    text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    header_x = panel_x + (panel_width - text_size[0]) // 2
    
    cv2.putText(frame, header_text, (header_x, panel_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.line(frame, (panel_x + 20, panel_y + 45), 
             (panel_x + panel_width - 20, panel_y + 45), (100, 100, 100), 2)
    
    y_offset = panel_y + 70
    
    for idx in range(len(zones)):
        zone_color = COLORS.by_idx(idx).as_bgr()
        traffic_condition, condition_color = traffic_analyzer.get_traffic_condition(idx)
        
        cv2.circle(frame, (panel_x + 25, y_offset), 10, zone_color, -1)
        cv2.circle(frame, (panel_x + 25, y_offset), 10, (255, 255, 255), 2)
        
        cv2.putText(frame, f"ZONE {idx + 1}", (panel_x + 45, y_offset + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        condition_text = traffic_condition
        cond_text_size = cv2.getTextSize(condition_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        badge_x = panel_x + panel_width - cond_text_size[0] - 30
        
        draw_rounded_rectangle(frame,
                              (badge_x - 8, y_offset - 10),
                              (badge_x + cond_text_size[0] + 8, y_offset + 5),
                              condition_color, radius=8)
        
        cv2.putText(frame, condition_text, (badge_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 25
        
        current_avg = stats['current_avg_dwell_times'][idx]
        ema_avg = stats['ema_dwell_times'][idx]
        
        stats_text = [
            f"Active Vehicles: {stats['current_vehicles'][idx]}",
            f"Total Entries: {stats['entry_counts'][idx]}",
            f"Current Avg Dwell: {current_avg:.1f}s",
            f"EMA Dwell Time: {ema_avg:.1f}s"
        ]
        
        for i, stat in enumerate(stats_text):
            color = (200, 200, 200)
            
            if i == 2 and current_avg > 0:
                if current_avg > 10:
                    color = (0, 100, 255)
                elif current_avg > 5:
                    color = (0, 200, 255)
                else:
                    color = (100, 255, 100)
            
            elif i == 3 and ema_avg > 0:
                if ema_avg > 10:
                    color = (0, 100, 255)
                elif ema_avg > 5:
                    color = (0, 200, 255)
                else:
                    color = (100, 255, 100)
            
            cv2.putText(frame, stat, (panel_x + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18
        
        y_offset += 15


def main(
    source_video_path: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
    target_video_path: str = None,
) -> None:
    model = YOLO(weights)
    tracker = sv.ByteTrack()
    
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]
    traffic_analyzer = TrafficAnalyzer(len(zones), alpha=0.15, fps=video_info.fps)

    if target_video_path:
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            process_frames(frames_generator, model, tracker, zones, timers, traffic_analyzer, 
                         device, confidence, iou, classes, sink, video_info)
    else:
        process_frames(frames_generator, model, tracker, zones, timers, traffic_analyzer, 
                     device, confidence, iou, classes, None, video_info)

    stats = traffic_analyzer.get_traffic_stats()
    print("\n" + "="*60)
    print("COMPREHENSIVE TRAFFIC ANALYSIS REPORT")
    print("="*60)
    
    for idx in range(len(zones)):
        traffic_condition, _ = traffic_analyzer.get_traffic_condition(idx)
        print(f"\nZONE {idx + 1} ANALYSIS:")
        print(f"  Total Entries: {stats['entry_counts'][idx]}")
        print(f"  EMA Dwell Time: {stats['ema_dwell_times'][idx]:.2f} seconds")
        print(f"  Maximum Dwell Time: {stats['max_dwell_times'][idx]:.2f} seconds")
        print(f"  Minimum Dwell Time: {stats['min_dwell_times'][idx]:.2f} seconds")
        print(f"  Current Vehicles: {stats['current_vehicles'][idx]}")
        print(f"  Traffic Condition: {traffic_condition}")
        print(f"  Total Measurements: {stats['total_measurements'][idx]}")


def process_frames(frames_generator, model, tracker, zones, timers, traffic_analyzer, 
                  device, confidence, iou, classes, video_sink=None, video_info=None):
    try:
        frame_iterator = tqdm(frames_generator, total=video_info.total_frames if video_info else None)
        
        for frame in frame_iterator:
            results = model(frame, verbose=False, device=device, conf=confidence, iou=iou)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            if classes:
                detections = detections[find_in_list(detections.class_id, classes)]
            
            detections = detections.with_nms(threshold=iou)
            detections = tracker.update_with_detections(detections)

            annotated_frame = frame.copy()
            detections_per_zone = []
            time_in_zone_per_zone = []

            for idx, zone in enumerate(zones):
                zone_color = COLORS.by_idx(idx)
                
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [zone.polygon.astype(np.int32)], zone_color.as_bgr())
                cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
                
                cv2.polylines(annotated_frame, [zone.polygon.astype(np.int32)], 
                             True, zone_color.as_bgr(), 3)

                detections_in_zone = detections[zone.trigger(detections)]
                detections_per_zone.append(detections_in_zone)
                
                time_in_zone = timers[idx].tick(detections_in_zone)
                time_in_zone_per_zone.append(time_in_zone)
                
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )
                
                labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]
                
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )

            current_counts, stats = traffic_analyzer.update_analysis(detections_per_zone, time_in_zone_per_zone)
            
            draw_enhanced_statistics(annotated_frame, traffic_analyzer, zones)

            if video_sink:
                video_sink.write_frame(annotated_frame)

            cv2.imshow("Real-time Traffic Analysis", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time traffic analysis with EMA-based dwell time analytics and enhanced visualization."
    )
    parser.add_argument("--zone_configuration_path", type=str, required=True,
                       help="Path to the zone configuration JSON file.")
    parser.add_argument("--source_video_path", type=str, required=True,
                       help="Path to the source video file.")
    parser.add_argument("--target_video_path", type=str, default=None,
                       help="Path to save the output video file.")
    parser.add_argument("--weights", type=str, default="yolov8s.pt",
                       help="Path to the model weights file.")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Computation device ('cpu', 'mps' or 'cuda').")
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                       help="Confidence level for detections (0 to 1).")
    parser.add_argument("--iou_threshold", default=0.7, type=float,
                       help="IOU threshold for non-max suppression.")
    parser.add_argument("--classes", nargs="*", type=int, default=[],
                       help="List of class IDs to track.")
    
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
        target_video_path=args.target_video_path,
    )