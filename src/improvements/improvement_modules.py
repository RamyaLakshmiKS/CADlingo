"""
CADlingo Improvement Modules

This file contains standalone Python classes and utilities for:
1. Geometric Validation
2. Enhanced Metrics
3. Production-ready components

Usage:
    from improvement_modules import GeometricValidator, GeometricMetrics
    
    validator = GeometricValidator()
    metrics = GeometricMetrics()
"""

import numpy as np
from typing import List, Dict, Tuple
from shapely.geometry import box
import warnings


class GeometricValidator:
    """
    Rule-based geometric validation for generated floor plans.
    
    Validates:
    - No overlapping rooms
    - Valid room dimensions
    - Realistic room adjacency
    - Overall floor plan coherence
    
    Example:
        >>> validator = GeometricValidator()
        >>> rooms = [{'type': 'living_room', 'center': (0,0), 'width': 4, 'height': 5}]
        >>> report = validator.validate_floor_plan(rooms)
        >>> print(f"Valid: {report['is_valid']}, Score: {report['overall_score']}")
    """
    
    def __init__(self, min_room_area=2.0, max_room_area=100.0, 
                 min_overlap_threshold=0.01, adjacency_threshold=0.5):
        """
        Initialize validator with constraints.
        
        Args:
            min_room_area: Minimum room area in m²
            max_room_area: Maximum room area in m²
            min_overlap_threshold: Minimum overlap percentage to flag error
            adjacency_threshold: Minimum adjacency to check for walls
        """
        self.min_room_area = min_room_area
        self.max_room_area = max_room_area
        self.min_overlap_threshold = min_overlap_threshold
        self.adjacency_threshold = adjacency_threshold
        self.issues = []
    
    def validate_room(self, room: Dict) -> bool:
        """Validate individual room constraints."""
        try:
            area = room['width'] * room['height']
            
            if area < self.min_room_area:
                self.issues.append(f"Room too small: {area:.2f}m²")
                return False
            
            if area > self.max_room_area:
                self.issues.append(f"Room too large: {area:.2f}m²")
                return False
            
            if room['width'] <= 0 or room['height'] <= 0:
                self.issues.append(f"Invalid dimensions")
                return False
            
            return True
        except Exception as e:
            self.issues.append(f"Error validating room: {e}")
            return False
    
    def check_overlap(self, rooms: List[Dict], tolerance: float = 0.1) -> List[Tuple]:
        """Detect overlapping rooms."""
        overlaps = []
        
        for i, room1 in enumerate(rooms):
            try:
                box1 = box(
                    room1['center'][0] - room1['width']/2,
                    room1['center'][1] - room1['height']/2,
                    room1['center'][0] + room1['width']/2,
                    room1['center'][1] + room1['height']/2
                )
            except:
                continue
            
            for j, room2 in enumerate(rooms[i+1:], start=i+1):
                try:
                    box2 = box(
                        room2['center'][0] - room2['width']/2,
                        room2['center'][1] - room2['height']/2,
                        room2['center'][0] + room2['width']/2,
                        room2['center'][1] + room2['height']/2
                    )
                    
                    intersection = box1.intersection(box2)
                    if intersection.area > tolerance:
                        overlap_pct = (intersection.area / min(box1.area, box2.area)) * 100
                        overlaps.append((i, j, intersection.area, overlap_pct))
                        self.issues.append(f"Overlap: {overlap_pct:.1f}%")
                except:
                    continue
        
        return overlaps
    
    def check_adjacency(self, rooms: List[Dict]) -> List[Tuple]:
        """Identify adjacent rooms (for topology analysis)."""
        adjacent = []
        
        for i, room1 in enumerate(rooms):
            box1 = box(
                room1['center'][0] - room1['width']/2,
                room1['center'][1] - room1['height']/2,
                room1['center'][0] + room1['width']/2,
                room1['center'][1] + room1['height']/2
            )
            
            for j, room2 in enumerate(rooms[i+1:], start=i+1):
                box2 = box(
                    room2['center'][0] - room2['width']/2,
                    room2['center'][1] - room2['height']/2,
                    room2['center'][0] + room2['width']/2,
                    room2['center'][1] + room2['height']/2
                )
                
                distance = box1.distance(box2)
                if distance < self.adjacency_threshold:
                    adjacent.append((i, j, distance))
        
        return adjacent
    
    def validate_floor_plan(self, rooms: List[Dict], return_report: bool = True) -> Dict:
        """Comprehensive floor plan validation."""
        self.issues = []
        
        valid_rooms = sum(1 for room in rooms if self.validate_room(room))
        room_validity_score = (valid_rooms / len(rooms) * 100) if rooms else 0
        
        overlaps = self.check_overlap(rooms)
        overlap_score = 100 if not overlaps else max(0, 100 - len(overlaps) * 20)
        
        adjacent_rooms = self.check_adjacency(rooms)
        
        overall_score = (room_validity_score + overlap_score) / 2
        is_valid = overall_score >= 80
        
        if return_report:
            return {
                'is_valid': is_valid,
                'overall_score': overall_score,
                'room_validity_score': room_validity_score,
                'overlap_score': overlap_score,
                'valid_rooms': valid_rooms,
                'total_rooms': len(rooms),
                'overlap_count': len(overlaps),
                'adjacent_pairs': len(adjacent_rooms),
                'issues': self.issues[:10],
                'issue_count': len(self.issues)
            }
        
        return {'is_valid': is_valid, 'overall_score': overall_score}


class GeometricMetrics:
    """
    Compute geometric consistency metrics beyond BLEU.
    
    Metrics:
    - IoU (Intersection over Union): Room overlap accuracy
    - Room Count Accuracy: Predicted vs actual room count
    - Adjacency Accuracy: Room relationship preservation
    - Layout Plausibility: Does layout make architectural sense?
    
    Example:
        >>> metrics = GeometricMetrics()
        >>> iou = metrics.compute_iou(pred_rooms, ref_rooms)
        >>> accuracy = metrics.room_count_accuracy(len(pred_rooms), len(ref_rooms))
    """
    
    @staticmethod
    def compute_iou(predicted_rooms: List[Dict], reference_rooms: List[Dict]) -> float:
        """Compute IoU between predicted and reference floor plans."""
        if not predicted_rooms or not reference_rooms:
            return 0.0
        
        try:
            pred_box = box(
                min(r['center'][0] - r['width']/2 for r in predicted_rooms),
                min(r['center'][1] - r['height']/2 for r in predicted_rooms),
                max(r['center'][0] + r['width']/2 for r in predicted_rooms),
                max(r['center'][1] + r['height']/2 for r in predicted_rooms)
            )
            
            ref_box = box(
                min(r['center'][0] - r['width']/2 for r in reference_rooms),
                min(r['center'][1] - r['height']/2 for r in reference_rooms),
                max(r['center'][0] + r['width']/2 for r in reference_rooms),
                max(r['center'][1] + r['height']/2 for r in reference_rooms)
            )
            
            intersection = pred_box.intersection(ref_box).area
            union = pred_box.union(ref_box).area
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def room_count_accuracy(predicted_count: int, reference_count: int) -> float:
        """Measure accuracy of predicted room count."""
        if reference_count == 0:
            return 1.0 if predicted_count == 0 else 0.0
        
        diff = abs(predicted_count - reference_count)
        return max(0.0, 1.0 - (diff / reference_count))
    
    @staticmethod
    def room_type_accuracy(predicted_types: List[str], reference_types: List[str]) -> float:
        """Measure accuracy of predicted room types."""
        if not reference_types:
            return 1.0 if not predicted_types else 0.0
        
        pred_freq = {}
        for t in predicted_types:
            pred_freq[t] = pred_freq.get(t, 0) + 1
        
        ref_freq = {}
        for t in reference_types:
            ref_freq[t] = ref_freq.get(t, 0) + 1
        
        matches = 0
        for room_type, count in ref_freq.items():
            matches += min(count, pred_freq.get(room_type, 0))
        
        return matches / len(reference_types) if reference_types else 0.0
    
    @staticmethod
    def adjacency_accuracy(predicted_rooms: List[Dict], reference_rooms: List[Dict]) -> float:
        """Measure how well room adjacencies are preserved."""
        def get_adjacencies(rooms: List[Dict], threshold: float = 1.0) -> set:
            adjacencies = set()
            for i, r1 in enumerate(rooms):
                for j, r2 in enumerate(rooms[i+1:]):
                    dist = np.sqrt(
                        (r1['center'][0] - r2['center'][0])**2 +
                        (r1['center'][1] - r2['center'][1])**2
                    )
                    
                    if dist < threshold:
                        pair = tuple(sorted([r1.get('type', 'unknown'), 
                                           r2.get('type', 'unknown')]))
                        adjacencies.add(pair)
            
            return adjacencies
        
        try:
            pred_adj = get_adjacencies(predicted_rooms)
            ref_adj = get_adjacencies(reference_rooms)
            
            if not ref_adj:
                return 1.0
            
            common = len(pred_adj.intersection(ref_adj))
            return common / len(ref_adj) if ref_adj else 0.0
        except:
            return 0.0
    
    @staticmethod
    def layout_plausibility(rooms: List[Dict]) -> Tuple[float, List[str]]:
        """Assess architectural plausibility of layout."""
        issues = []
        score = 100.0
        
        room_types = [r.get('type', 'unknown') for r in rooms]
        room_dict = {r.get('type', 'unknown'): r for r in rooms}
        
        if 'bathroom' not in room_types:
            issues.append("Missing bathroom")
            score -= 15
        
        if 'living_room' in room_dict:
            living_area = room_dict['living_room'].get('area', 0)
            if living_area < 15:
                issues.append(f"Small living room ({living_area:.1f}m²)")
                score -= 10
        else:
            issues.append("Missing living room")
            score -= 5
        
        bedrooms = [t for t in room_types if 'bedroom' in t]
        if not bedrooms:
            issues.append("Missing bedrooms")
            score -= 20
        
        unique_types = len(set(room_types))
        if unique_types < 3:
            issues.append(f"Low room diversity ({unique_types} types)")
            score -= 5
        
        return max(0, score), issues


if __name__ == "__main__":
    # Example usage
    validator = GeometricValidator()
    metrics = GeometricMetrics()
    
    # Sample floor plan
    sample_rooms = [
        {'type': 'living_room', 'center': (0, 0), 'width': 4, 'height': 5, 'area': 20},
        {'type': 'kitchen', 'center': (5, 0), 'width': 3, 'height': 2.5, 'area': 7.5},
        {'type': 'bathroom', 'center': (5, 3), 'width': 2, 'height': 2, 'area': 4},
        {'type': 'master_bedroom', 'center': (0, 6), 'width': 3.5, 'height': 4, 'area': 14},
    ]
    
    # Validate
    report = validator.validate_floor_plan(sample_rooms)
    print(f"Validation Report: {report}")
    
    # Check plausibility
    plausibility, issues = metrics.layout_plausibility(sample_rooms)
    print(f"\nPlausibility Score: {plausibility}%")
    print(f"Issues: {issues}")
