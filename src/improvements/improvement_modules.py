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


class AutomatedEvaluator:
    """
    Automated evaluation module that measures geometric consistency alongside BLEU.
    
    Combines:
    - BLEU score (text similarity)
    - Geometric metrics (IoU, room count, adjacency, plausibility)
    - Validation results
    
    Provides comprehensive quality assessment for generated floor plans.
    """
    
    def __init__(self):
        self.validator = GeometricValidator()
        self.metrics = GeometricMetrics()
    
    def evaluate_prediction(self, predicted_code: str, reference_code: str,
                          predicted_rooms: List[Dict], reference_rooms: List[Dict]) -> Dict:
        """
        Comprehensive evaluation of a single prediction.
        
        Args:
            predicted_code: Generated AutoCAD code
            reference_code: Ground truth AutoCAD code
            predicted_rooms: Parsed predicted room structures
            reference_rooms: Parsed reference room structures
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Geometric metrics
        iou = self.metrics.compute_iou(predicted_rooms, reference_rooms)
        
        room_count_acc = self.metrics.room_count_accuracy(
            len(predicted_rooms), len(reference_rooms)
        )
        
        pred_types = [r.get('type', 'unknown') for r in predicted_rooms]
        ref_types = [r.get('type', 'unknown') for r in reference_rooms]
        type_acc = self.metrics.room_type_accuracy(pred_types, ref_types)
        
        adjacency_acc = self.metrics.adjacency_accuracy(predicted_rooms, reference_rooms)
        
        plausibility_score, plausibility_issues = self.metrics.layout_plausibility(predicted_rooms)
        
        # Validation
        validation_report = self.validator.validate_floor_plan(predicted_rooms)
        
        # Combined score
        geometric_score = (iou * 100 + room_count_acc * 100 + 
                          type_acc * 100 + adjacency_acc * 100 + plausibility_score) / 5
        
        return {
            'geometric_metrics': {
                'iou': iou,
                'room_count_accuracy': room_count_acc,
                'room_type_accuracy': type_acc,
                'adjacency_accuracy': adjacency_acc,
                'plausibility_score': plausibility_score / 100,
                'plausibility_issues': plausibility_issues
            },
            'validation': validation_report,
            'scores': {
                'geometric_score': geometric_score,
                'validation_score': validation_report['overall_score'],
                'combined_score': (geometric_score + validation_report['overall_score']) / 2
            },
            'room_counts': {
                'predicted': len(predicted_rooms),
                'reference': len(reference_rooms),
                'difference': abs(len(predicted_rooms) - len(reference_rooms))
            }
        }
    
    def evaluate_batch(self, predictions: List[Dict], references: List[Dict]) -> Dict:
        """
        Evaluate multiple predictions and aggregate metrics.
        
        Args:
            predictions: List of {'code': str, 'rooms': List[Dict]}
            references: List of {'code': str, 'rooms': List[Dict]}
            
        Returns:
            Aggregated evaluation report
        """
        all_metrics = []
        
        for pred, ref in zip(predictions, references):
            metrics = self.evaluate_prediction(
                pred['code'], ref['code'],
                pred['rooms'], ref['rooms']
            )
            all_metrics.append(metrics)
        
        # Aggregate
        avg_iou = np.mean([m['geometric_metrics']['iou'] for m in all_metrics])
        avg_room_count_acc = np.mean([m['geometric_metrics']['room_count_accuracy'] for m in all_metrics])
        avg_type_acc = np.mean([m['geometric_metrics']['room_type_accuracy'] for m in all_metrics])
        avg_adjacency = np.mean([m['geometric_metrics']['adjacency_accuracy'] for m in all_metrics])
        avg_plausibility = np.mean([m['geometric_metrics']['plausibility_score'] for m in all_metrics])
        avg_validation = np.mean([m['scores']['validation_score'] for m in all_metrics])
        avg_combined = np.mean([m['scores']['combined_score'] for m in all_metrics])
        
        return {
            'summary': {
                'total_samples': len(all_metrics),
                'avg_iou': avg_iou,
                'avg_room_count_accuracy': avg_room_count_acc,
                'avg_room_type_accuracy': avg_type_acc,
                'avg_adjacency_accuracy': avg_adjacency,
                'avg_plausibility': avg_plausibility,
                'avg_validation_score': avg_validation,
                'avg_combined_score': avg_combined
            },
            'detailed_results': all_metrics
        }
    
    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate human-readable evaluation report."""
        summary = evaluation_results['summary']
        
        report = f"""
{'='*80}
CADLINGO AUTOMATED EVALUATION REPORT
{'='*80}

Total Samples Evaluated: {summary['total_samples']}

GEOMETRIC METRICS:
  IoU (Layout Overlap):          {summary['avg_iou']:.3f}  ({summary['avg_iou']*100:.1f}%)
  Room Count Accuracy:           {summary['avg_room_count_accuracy']:.3f}  ({summary['avg_room_count_accuracy']*100:.1f}%)
  Room Type Accuracy:            {summary['avg_room_type_accuracy']:.3f}  ({summary['avg_room_type_accuracy']*100:.1f}%)
  Adjacency Accuracy:            {summary['avg_adjacency_accuracy']:.3f}  ({summary['avg_adjacency_accuracy']*100:.1f}%)
  Layout Plausibility:           {summary['avg_plausibility']:.3f}  ({summary['avg_plausibility']*100:.1f}%)

VALIDATION:
  Average Validation Score:      {summary['avg_validation_score']:.1f}%

OVERALL:
  Combined Quality Score:        {summary['avg_combined_score']:.1f}%

{'='*80}
"""
        return report


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
