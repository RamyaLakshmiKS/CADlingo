"""
RPLAN Dataset Loader

This module loads and parses RPLAN pickle files containing floor plan data.
Each pickle file contains structured information about rooms, walls, doors, and boundaries.

"""

import pickle
import os
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path


class RPlanLoader:
    """
    Loads and processes RPLAN pickle files.
    
    The RPLAN dataset contains floor plans with:
    - Rooms: type, coordinates, boundaries
    - Walls: coordinates, connections
    - Doors: positions, types
    - Overall floor plan dimensions
    """
    
    # Room type mapping (RPLAN uses numeric codes)
    ROOM_TYPES = {
        0: "living_room",
        1: "master_bedroom",
        2: "kitchen",
        3: "bathroom",
        4: "dining_room",
        5: "child_bedroom",
        6: "study_room",
        7: "second_living_room",
        8: "guest_bedroom",
        9: "balcony",
        10: "entrance",
        11: "storage",
        12: "walk_in_closet",
        13: "exterior",
        14: "exterior_wall",
        15: "front_door",
        16: "interior_wall",
        17: "interior_door"
    }
    
    def __init__(self, data_root: str = None):
        """
        Initialize the RPLAN loader.
        
        Args:
            data_root: Path to the data root directory (default: project root/data/raw/pickle)
        """
        if data_root is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            data_root = project_root / "data" / "raw" / "pickle"
        
        self.data_root = Path(data_root)
        self.train_path = self.data_root / "train"
        self.val_path = self.data_root / "val"
        
    def load_pickle(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single RPLAN pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Dictionary containing floor plan data
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # RPLAN format: list of [boundary, room_type_mask, close_wall, front_door, rooms_info]
        # data[0]: boundary mask (256x256)
        # data[1]: room type mask (256x256) 
        # data[2]: close wall mask (256x256)
        # data[3]: front door mask (256x256)
        # data[4]: list of room dictionaries with 'category' and 'centroid'
        if isinstance(data, list) and len(data) >= 5:
            return {
                'boundary': data[0] if len(data) > 0 else None,
                'room_type_mask': data[1] if len(data) > 1 else None,
                'close_wall': data[2] if len(data) > 2 else None,
                'front_door': data[3] if len(data) > 3 else None,
                'rooms_info': data[4] if len(data) > 4 else [],
                'raw_data': data
            }
        return data
    
    def get_file_list(self, split: str = "train") -> List[str]:
        """
        Get list of all pickle files in a split.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            List of file paths
        """
        if split == "train":
            path = self.train_path
        elif split == "val":
            path = self.val_path
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'val'")
        
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")
        
        # Get all .pkl files
        files = sorted(list(path.glob("*.pkl")))
        return [str(f) for f in files]
    
    def parse_floor_plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse floor plan data into structured format.
        
        RPLAN dataset structure:
        - data[0]: boundary mask (255=interior, 0=exterior)
        - data[1]: room type visualization (127=walls, 255=doors, 0=rooms)  
        - data[2]: close wall mask
        - data[3]: front door mask
        - data[4]: list of room dicts with 'category' and 'centroid'
        
        Note: RPLAN doesn't have per-room instance masks, only centroids and categories.
        We estimate room regions using Voronoi-like partitioning of the interior space.
        
        Args:
            data: Floor plan data dictionary from RPLAN
            
        Returns:
            Parsed floor plan with rooms, dimensions, etc.
        """
        rooms = []
        
        # Get masks and room metadata
        boundary_mask = data.get('boundary')
        rooms_info = data.get('rooms_info', [])
        
        # Extract room information from metadata
        # RPLAN provides centroid and category for each room
        for room_info in rooms_info:
            try:
                room_category = room_info.get('category', 0)
                centroid = room_info.get('centroid', (0, 0))
                
                # Map category to room type
                room_type = self.ROOM_TYPES.get(room_category, 'unknown')
                
                cy, cx = centroid  # centroid is (y, x) in image coordinates
                
                # Estimate room size using average room area
                # Typical room is ~3-4m x 3-4m (9-16 m²)
                scale = 10.0 / 256.0  # meters per pixel
                
                # Estimate room dimensions based on type
                # These are rough approximations since RPLAN doesn't provide exact boundaries
                typical_sizes = {
                    'living_room': (4.0, 5.0),      # 20 m²
                    'master_bedroom': (3.5, 4.0),   # 14 m²
                    'kitchen': (2.5, 3.0),          # 7.5 m²
                    'bathroom': (2.0, 2.5),         # 5 m²
                    'dining_room': (3.0, 3.5),      # 10.5 m²
                    'child_bedroom': (3.0, 3.5),    # 10.5 m²
                    'balcony': (1.5, 3.0),          # 4.5 m²
                    'study_room': (2.5, 3.0),       # 7.5 m²
                }
                
                width, height = typical_sizes.get(room_type, (3.0, 3.0))
                area = width * height
                
                # Create room data with estimated dimensions
                room_data = {
                    'type': room_type,
                    'category': int(room_category),
                    'center': (float(cx * scale), float(cy * scale)),  # Convert to meters
                    'width': float(width),
                    'height': float(height),
                    'area': float(area),
                    'bbox': {
                        'min_x': float(cx * scale - width / 2),
                        'max_x': float(cx * scale + width / 2),
                        'min_y': float(cy * scale - height / 2),
                        'max_y': float(cy * scale + height / 2)
                    }
                }
                rooms.append(room_data)
                
            except Exception as e:
                print(f"Warning: Error parsing room {room_info}: {e}")
                continue
        
        # Calculate overall dimensions from boundary
        if boundary_mask is not None:
            boundary_points = np.where(boundary_mask == 255)  # interior pixels
            if len(boundary_points[0]) > 0:
                scale = 10.0 / 256.0
                overall_width = (boundary_points[1].max() - boundary_points[1].min() + 1) * scale
                overall_height = (boundary_points[0].max() - boundary_points[0].min() + 1) * scale
            else:
                overall_width = 10.0  # default
                overall_height = 10.0
        else:
            overall_width = 10.0
            overall_height = 10.0
        
        return {
            'rooms': rooms,
            'dimensions': {
                'width': float(overall_width),
                'height': float(overall_height)
            },
            'num_rooms': len(rooms)
        }
    
    def _parse_room(self, room_data: Any, idx: int) -> Dict[str, Any]:
        """Parse individual room data."""
        try:
            # RPLAN format: rooms can be stored as arrays or dictionaries
            if isinstance(room_data, dict):
                room_type = room_data.get('type', 0)
                vertices = np.array(room_data.get('vertices', []))
            else:
                # Array format: [type, vertices]
                room_type = int(room_data[0]) if len(room_data) > 0 else 0
                vertices = np.array(room_data[1:]) if len(room_data) > 1 else np.array([])
            
            # Get room type name
            type_name = self.ROOM_TYPES.get(room_type, f"unknown_{room_type}")
            
            # Calculate area and center
            if len(vertices) > 0:
                if vertices.ndim == 1:
                    vertices = vertices.reshape(-1, 2)
                
                area = self._calculate_polygon_area(vertices)
                center = vertices.mean(axis=0)
                
                # Calculate bounding box
                bbox = {
                    'min_x': float(vertices[:, 0].min()),
                    'min_y': float(vertices[:, 1].min()),
                    'max_x': float(vertices[:, 0].max()),
                    'max_y': float(vertices[:, 1].max()),
                    'width': float(vertices[:, 0].max() - vertices[:, 0].min()),
                    'height': float(vertices[:, 1].max() - vertices[:, 1].min())
                }
            else:
                area = 0
                center = np.array([0, 0])
                bbox = {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0, 'width': 0, 'height': 0}
            
            return {
                'id': idx,
                'type': type_name,
                'type_code': room_type,
                'vertices': vertices.tolist() if len(vertices) > 0 else [],
                'area': float(area),
                'center': center.tolist() if len(center) > 0 else [0, 0],
                'bbox': bbox
            }
        except Exception as e:
            print(f"Warning: Failed to parse room {idx}: {e}")
            return None
    
    def _parse_wall(self, wall_data: Any) -> Dict[str, Any]:
        """Parse individual wall data."""
        try:
            # Wall format: start_point, end_point
            if isinstance(wall_data, dict):
                start = np.array(wall_data.get('start', [0, 0]))
                end = np.array(wall_data.get('end', [0, 0]))
            else:
                start = np.array(wall_data[:2])
                end = np.array(wall_data[2:4]) if len(wall_data) >= 4 else start
            
            length = np.linalg.norm(end - start)
            
            return {
                'start': start.tolist(),
                'end': end.tolist(),
                'length': float(length)
            }
        except Exception as e:
            print(f"Warning: Failed to parse wall: {e}")
            return None
    
    def _parse_door(self, door_data: Any) -> Dict[str, Any]:
        """Parse individual door data."""
        try:
            if isinstance(door_data, dict):
                position = np.array(door_data.get('position', [0, 0]))
                door_type = door_data.get('type', 'interior')
            else:
                position = np.array(door_data[:2])
                door_type = 'interior'
            
            return {
                'position': position.tolist(),
                'type': door_type
            }
        except Exception as e:
            print(f"Warning: Failed to parse door: {e}")
            return None
    
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of a polygon using the shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return float(area)
    
    def load_batch(self, file_paths: List[str], max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Load and parse multiple floor plans.
        
        Args:
            file_paths: List of pickle file paths
            max_samples: Maximum number of samples to load (None = all)
            
        Returns:
            List of parsed floor plans
        """
        floor_plans = []
        
        if max_samples:
            file_paths = file_paths[:max_samples]
        
        for file_path in file_paths:
            try:
                raw_data = self.load_pickle(file_path)
                parsed_data = self.parse_floor_plan(raw_data)
                parsed_data['file_name'] = os.path.basename(file_path)
                floor_plans.append(parsed_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return floor_plans
    
    def get_statistics(self, split: str = "train", max_samples: int = 1000) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Args:
            split: 'train' or 'val'
            max_samples: Maximum samples to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        files = self.get_file_list(split)[:max_samples]
        floor_plans = self.load_batch(files)
        
        stats = {
            'total_samples': len(floor_plans),
            'avg_rooms_per_plan': 0,
            'room_type_counts': {},
            'avg_area': 0,
            'avg_dimensions': {'width': 0, 'height': 0}
        }
        
        total_rooms = 0
        total_area = 0
        total_width = 0
        total_height = 0
        
        for plan in floor_plans:
            num_rooms = len(plan['rooms'])
            total_rooms += num_rooms
            
            for room in plan['rooms']:
                room_type = room['type']
                stats['room_type_counts'][room_type] = stats['room_type_counts'].get(room_type, 0) + 1
                total_area += room['area']
            
            if plan['dimensions']:
                total_width += plan['dimensions']['width']
                total_height += plan['dimensions']['height']
        
        if len(floor_plans) > 0:
            stats['avg_rooms_per_plan'] = total_rooms / len(floor_plans)
            stats['avg_area'] = total_area / total_rooms if total_rooms > 0 else 0
            stats['avg_dimensions']['width'] = total_width / len(floor_plans)
            stats['avg_dimensions']['height'] = total_height / len(floor_plans)
        
        return stats


def main():
    """Test the RPLAN loader."""
    loader = RPlanLoader()
    
    # Get file lists
    train_files = loader.get_file_list("train")
    val_files = loader.get_file_list("val")
    
    print(f"Found {len(train_files)} training files")
    print(f"Found {len(val_files)} validation files")
    
    # Load a sample
    if train_files:
        print("\nLoading sample file...")
        sample_data = loader.load_pickle(train_files[0])
        print(f"Raw data keys: {list(sample_data.keys())}")
        
        parsed = loader.parse_floor_plan(sample_data)
        print(f"\nParsed data:")
        print(f"  Rooms: {len(parsed['rooms'])}")
        print(f"  Walls: {len(parsed['walls'])}")
        print(f"  Doors: {len(parsed['doors'])}")
        
        if parsed['rooms']:
            print(f"\nFirst room:")
            print(f"  Type: {parsed['rooms'][0]['type']}")
            print(f"  Area: {parsed['rooms'][0]['area']:.2f}")
            print(f"  Center: {parsed['rooms'][0]['center']}")
    
    # Get statistics
    print("\nComputing dataset statistics...")
    stats = loader.get_statistics("train", max_samples=100)
    print(f"\nDataset Statistics (sample of 100):")
    print(f"  Avg rooms per plan: {stats['avg_rooms_per_plan']:.2f}")
    print(f"  Avg room area: {stats['avg_area']:.2f}")
    print(f"  Avg plan dimensions: {stats['avg_dimensions']['width']:.2f} x {stats['avg_dimensions']['height']:.2f}")
    print(f"\nRoom type distribution:")
    for room_type, count in sorted(stats['room_type_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {room_type}: {count}")


if __name__ == "__main__":
    main()
