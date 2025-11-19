"""
Geometric Validation Module for CADlingo

Rule-based validation and correction of generated floor plans.
Ensures no overlapping rooms, valid dimensions, and architectural plausibility.

Author: Ramya Lakshmi KS
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple
from shapely.geometry import box
import logging

logger = logging.getLogger(__name__)


class GeometricValidator:
    """
    Rule-based geometric validation for generated floor plans.
    
    Validates:
    - No overlapping rooms
    - Valid room dimensions  
    - Realistic room adjacency
    - Overall floor plan coherence
    
    Example:
    --------
    >>> validator = GeometricValidator()
    >>> rooms = [
    ...     {'type': 'bedroom', 'center': (5, 5), 'width': 4, 'height': 3},
    ...     {'type': 'kitchen', 'center': (10, 5), 'width': 3, 'height': 2}
    ... ]
    >>> report = validator.validate_floor_plan(rooms)
    >>> print(f"Valid: {report['is_valid']}, Score: {report['overall_score']}")
    """
    
    def __init__(
        self,
        min_room_area: float = 2.0,
        max_room_area: float = 100.0,
        min_overlap_threshold: float = 0.01,
        adjacency_threshold: float = 0.5
    ):
        """
        Initialize validator with constraints.
        
        Args:
            min_room_area: Minimum room area in m² (default: 2.0)
            max_room_area: Maximum room area in m² (default: 100.0)
            min_overlap_threshold: Minimum overlap in m² to flag error (default: 0.01)
            adjacency_threshold: Max distance in m for rooms to be adjacent (default: 0.5)
        """
        self.min_room_area = min_room_area
        self.max_room_area = max_room_area
        self.min_overlap_threshold = min_overlap_threshold
        self.adjacency_threshold = adjacency_threshold
        self.issues = []
        self.warnings = []
    
    def validate_room(self, room: Dict) -> Tuple[bool, List[str]]:
        """
        Validate individual room constraints.
        
        Args:
            room: Room dictionary with 'width', 'height', 'type', 'center'
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        try:
            area = room.get('width', 0) * room.get('height', 0)
            room_type = room.get('type', 'unknown')
            
            # Check area constraints
            if area < self.min_room_area:
                issues.append(f"Room {room_type} too small: {area:.2f}m² (min: {self.min_room_area}m²)")
            
            if area > self.max_room_area:
                issues.append(f"Room {room_type} too large: {area:.2f}m² (max: {self.max_room_area}m²)")
            
            # Check for non-positive dimensions
            if room.get('width', 0) <= 0 or room.get('height', 0) <= 0:
                issues.append(f"Room {room_type} has invalid dimensions: {room.get('width')}×{room.get('height')}")
            
            # Check for missing center
            if not room.get('center') or len(room.get('center', [])) != 2:
                issues.append(f"Room {room_type} missing valid center coordinate")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error validating room: {str(e)}"]
    
    def check_overlaps(self, rooms: List[Dict], tolerance: float = 0.1) -> List[Tuple]:
        """
        Detect overlapping rooms.
        
        Args:
            rooms: List of room dictionaries
            tolerance: Overlap tolerance in m² (default: 0.1)
            
        Returns:
            List of (room1_idx, room2_idx, overlap_area, overlap_pct) tuples
        """
        overlaps = []
        
        for i, room1 in enumerate(rooms):
            try:
                center1 = room1.get('center', (0, 0))
                w1 = room1.get('width', 0)
                h1 = room1.get('height', 0)
                
                box1 = box(
                    center1[0] - w1 / 2,
                    center1[1] - h1 / 2,
                    center1[0] + w1 / 2,
                    center1[1] + h1 / 2
                )
                
                if box1.area == 0:
                    continue
                    
            except Exception as e:
                logger.warning(f"Error creating box for room1: {e}")
                continue
            
            for j, room2 in enumerate(rooms[i + 1 :], start=i + 1):
                try:
                    center2 = room2.get('center', (0, 0))
                    w2 = room2.get('width', 0)
                    h2 = room2.get('height', 0)
                    
                    box2 = box(
                        center2[0] - w2 / 2,
                        center2[1] - h2 / 2,
                        center2[0] + w2 / 2,
                        center2[1] + h2 / 2
                    )
                    
                    if box2.area == 0:
                        continue
                    
                    intersection = box1.intersection(box2)
                    if intersection.area > tolerance:
                        overlap_pct = (intersection.area / min(box1.area, box2.area)) * 100
                        overlaps.append((i, j, intersection.area, overlap_pct))
                        
                        issue = (
                            f"Overlap: {room1.get('type', 'Room')} and "
                            f"{room2.get('type', 'Room')} ({overlap_pct:.1f}%)"
                        )
                        self.issues.append(issue)
                        
                except Exception as e:
                    logger.warning(f"Error checking overlap between rooms: {e}")
                    continue
        
        return overlaps
    
    def check_adjacencies(self, rooms: List[Dict]) -> List[Tuple]:
        """
        Identify adjacent rooms (for topology analysis).
        
        Args:
            rooms: List of room dictionaries
            
        Returns:
            List of (room1_idx, room2_idx, distance) tuples for adjacent rooms
        """
        adjacent = []
        
        for i, room1 in enumerate(rooms):
            try:
                center1 = room1.get('center', (0, 0))
                w1 = room1.get('width', 0)
                h1 = room1.get('height', 0)
                
                box1 = box(
                    center1[0] - w1 / 2,
                    center1[1] - h1 / 2,
                    center1[0] + w1 / 2,
                    center1[1] + h1 / 2
                )
                
            except Exception as e:
                logger.warning(f"Error creating adjacency box for room1: {e}")
                continue
            
            for j, room2 in enumerate(rooms[i + 1 :], start=i + 1):
                try:
                    center2 = room2.get('center', (0, 0))
                    w2 = room2.get('width', 0)
                    h2 = room2.get('height', 0)
                    
                    box2 = box(
                        center2[0] - w2 / 2,
                        center2[1] - h2 / 2,
                        center2[0] + w2 / 2,
                        center2[1] + h2 / 2
                    )
                    
                    distance = box1.distance(box2)
                    if distance < self.adjacency_threshold:
                        adjacent.append((i, j, distance))
                        
                except Exception as e:
                    logger.warning(f"Error checking adjacency: {e}")
                    continue
        
        return adjacent
    
    def validate_floor_plan(
        self,
        rooms: List[Dict],
        return_report: bool = True
    ) -> Dict:
        """
        Comprehensive floor plan validation.
        
        Args:
            rooms: List of room dictionaries
            return_report: Whether to return detailed report (default: True)
            
        Returns:
            Validation report with scores and issues
        """
        self.issues = []
        self.warnings = []
        
        # Validate individual rooms
        valid_count = 0
        for room in rooms:
            is_valid, room_issues = self.validate_room(room)
            if is_valid:
                valid_count += 1
            self.issues.extend(room_issues)
        
        room_validity_score = (valid_count / len(rooms) * 100) if rooms else 0
        
        # Check overlaps
        overlaps = self.check_overlaps(rooms)
        overlap_score = 100 if not overlaps else max(0, 100 - len(overlaps) * 20)
        
        # Check adjacencies
        adjacent_rooms = self.check_adjacencies(rooms)
        
        # Calculate overall score
        overall_score = (room_validity_score + overlap_score) / 2
        is_valid = overall_score >= 80
        
        if return_report:
            return {
                'is_valid': is_valid,
                'overall_score': overall_score,
                'room_validity_score': room_validity_score,
                'overlap_score': overlap_score,
                'valid_rooms': valid_count,
                'total_rooms': len(rooms),
                'overlap_count': len(overlaps),
                'overlaps': overlaps,
                'adjacent_pairs': len(adjacent_rooms),
                'adjacencies': adjacent_rooms,
                'issues': self.issues[:10],  # Top 10 issues
                'issue_count': len(self.issues),
                'warnings': self.warnings
            }
        
        return {'is_valid': is_valid, 'overall_score': overall_score}


class GeometricMetrics:
    """
    Compute geometric consistency metrics beyond BLEU.
    
    Metrics computed:
    - IoU (Intersection over Union): Layout overlap accuracy
    - Room Count Accuracy: Predicted vs reference room count
    - Room Type Accuracy: Room type matching
    - Adjacency Accuracy: Room relationship preservation
    - Layout Plausibility: Architectural sense check
    """
    
    @staticmethod
    def compute_iou(predicted_rooms: List[Dict], reference_rooms: List[Dict]) -> float:
        """
        Compute IoU between predicted and reference floor plans.
        
        Args:
            predicted_rooms: Generated room list
            reference_rooms: Ground truth room list
            
        Returns:
            IoU score (0-1)
        """
        if not predicted_rooms or not reference_rooms:
            return 0.0
        
        try:
            # Create bounding boxes for overall layouts
            pred_centers = [r.get('center', (0, 0)) for r in predicted_rooms]
            pred_widths = [r.get('width', 0) for r in predicted_rooms]
            pred_heights = [r.get('height', 0) for r in predicted_rooms]
            
            ref_centers = [r.get('center', (0, 0)) for r in reference_rooms]
            ref_widths = [r.get('width', 0) for r in reference_rooms]
            ref_heights = [r.get('height', 0) for r in reference_rooms]
            
            pred_box = box(
                min(c[0] - w / 2 for c, w in zip(pred_centers, pred_widths)),
                min(c[1] - h / 2 for c, h in zip(pred_centers, pred_heights)),
                max(c[0] + w / 2 for c, w in zip(pred_centers, pred_widths)),
                max(c[1] + h / 2 for c, h in zip(pred_centers, pred_heights))
            )
            
            ref_box = box(
                min(c[0] - w / 2 for c, w in zip(ref_centers, ref_widths)),
                min(c[1] - h / 2 for c, h in zip(ref_centers, ref_heights)),
                max(c[0] + w / 2 for c, w in zip(ref_centers, ref_widths)),
                max(c[1] + h / 2 for c, h in zip(ref_centers, ref_heights))
            )
            
            intersection = pred_box.intersection(ref_box).area
            union = pred_box.union(ref_box).area
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing IoU: {e}")
            return 0.0
    
    @staticmethod
    def room_count_accuracy(predicted_count: int, reference_count: int) -> float:
        """
        Measure accuracy of predicted room count.
        
        Args:
            predicted_count: Number of predicted rooms
            reference_count: Number of reference rooms
            
        Returns:
            Accuracy score (0-1)
        """
        if reference_count == 0:
            return 1.0 if predicted_count == 0 else 0.0
        
        diff = abs(predicted_count - reference_count)
        # 1.0 if exact, linearly decreases
        return max(0.0, 1.0 - (diff / reference_count))
    
    @staticmethod
    def room_type_accuracy(predicted_types: List[str], reference_types: List[str]) -> float:
        """
        Measure accuracy of predicted room types.
        
        Args:
            predicted_types: List of predicted room type names
            reference_types: List of reference room type names
            
        Returns:
            Type matching accuracy (0-1)
        """
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
        """
        Measure how well room adjacencies are preserved.
        
        Args:
            predicted_rooms: Generated floor plan rooms
            reference_rooms: Ground truth rooms
            
        Returns:
            Adjacency preservation score (0-1)
        """
        def get_adjacencies(rooms: List[Dict], threshold: float = 1.0) -> set:
            """Get set of adjacent room type pairs."""
            adjacencies = set()
            for i, r1 in enumerate(rooms):
                c1 = r1.get('center', (0, 0))
                for j, r2 in enumerate(rooms[i + 1 :]):
                    c2 = r2.get('center', (0, 0))
                    dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                    
                    if dist < threshold:
                        t1 = r1.get('type', 'unknown')
                        t2 = r2.get('type', 'unknown')
                        pair = tuple(sorted([t1, t2]))
                        adjacencies.add(pair)
            
            return adjacencies
        
        try:
            pred_adj = get_adjacencies(predicted_rooms)
            ref_adj = get_adjacencies(reference_rooms)
            
            if not ref_adj:
                return 1.0
            
            common = len(pred_adj.intersection(ref_adj))
            return common / len(ref_adj) if ref_adj else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing adjacency accuracy: {e}")
            return 0.0


if __name__ == "__main__":
    # Test example
    validator = GeometricValidator()
    
    test_rooms = [
        {'type': 'bedroom', 'center': (5, 5), 'width': 4, 'height': 3},
        {'type': 'kitchen', 'center': (10, 5), 'width': 3, 'height': 2},
        {'type': 'bathroom', 'center': (13, 7), 'width': 2, 'height': 2}
    ]
    
    report = validator.validate_floor_plan(test_rooms)
    print("\nValidation Report:")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Overall Score: {report['overall_score']:.1f}%")
    print(f"  Valid Rooms: {report['valid_rooms']}/{report['total_rooms']}")
    print(f"  Overlaps: {report['overlap_count']}")
    print(f"  Issues: {report['issue_count']}")
    
    # Test metrics
    metrics = GeometricMetrics()
    iou = metrics.compute_iou(test_rooms, test_rooms)
    room_acc = metrics.room_count_accuracy(3, 3)
    
    print("\nGeometric Metrics:")
    print(f"  IoU: {iou:.2f}")
    print(f"  Room Count Accuracy: {room_acc:.2f}")
