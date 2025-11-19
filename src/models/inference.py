"""
Inference Module for CADlingo

Generates AutoCAD code from natural language descriptions using trained CodeT5 model.

This module:
1. Loads trained model checkpoint
2. Generates AutoCAD code from text prompts
3. Exports to DXF format
4. Provides visualization

Author: Ramya Lakshmi KS
Date: November 2025
"""

import torch
import ezdxf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from geometric_validator import GeometricValidator, GeometricMetrics


class CADGenerator:
    """
    Generates AutoCAD code and DXF files from text descriptions.
    """
    
    def __init__(self, model_path: str = None, device: str = None, enable_validation: bool = True):
        """
        Initialize CAD generator.
        
        Args:
            model_path: Path to trained model checkpoint (default: best_model)
            device: 'cuda' or 'cpu' (auto-detect if None)
            enable_validation: Whether to enable geometric validation (default: True)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Setup model path
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "results" / "models" / "best_model"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize geometric validator
        self.enable_validation = enable_validation
        if self.enable_validation:
            self.validator = GeometricValidator()
            self.metrics = GeometricMetrics()
        
        print(f"Model loaded successfully on {self.device}")
        if self.enable_validation:
            print("Geometric validation: ENABLED")
    
    def generate_code(
        self,
        description: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 1.0
    ) -> str:
        """
        Generate AutoCAD code from text description.
        
        Args:
            description: Natural language description of floor plan
            max_length: Maximum length of generated code
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated AutoCAD code as string
        """
        # Tokenize input
        inputs = self.tokenizer(
            description,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        generated_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_code
    
    def generate_with_validation(
        self,
        description: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 1.0
    ) -> Tuple[str, Dict]:
        """
        Generate AutoCAD code with geometric validation.
        
        Args:
            description: Natural language description of floor plan
            max_length: Maximum length of generated code
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated_code, validation_report)
        """
        code = self.generate_code(description, max_length, num_beams, temperature)
        
        if not self.enable_validation:
            return code, {'validation_enabled': False}
        
        # Parse code into rooms
        elements = self.parse_code_to_elements(code)
        rooms = self._extract_rooms_from_elements(elements)
        
        # Validate floor plan
        validation_report = self.validator.validate_floor_plan(rooms)
        
        # Compute geometric metrics
        if rooms:
            metrics = {
                'room_count': len(rooms),
                'room_types': [r.get('type', 'unknown') for r in rooms],
                'avg_room_area': np.mean([r.get('width', 0) * r.get('height', 0) for r in rooms]) if rooms else 0
            }
            validation_report['metrics'] = metrics
        
        return code, validation_report
    
    def _extract_rooms_from_elements(self, elements: Dict[str, List]) -> List[Dict]:
        """
        Extract room information from parsed elements.
        
        Args:
            elements: Parsed code elements from parse_code_to_elements
            
        Returns:
            List of room dictionaries
        """
        rooms = []
        for rect in elements.get('rectangles', []):
            x1, y1 = rect['x1'], rect['y1']
            x2, y2 = rect['x2'], rect['y2']
            
            room = {
                'type': rect['label'].lower() if rect['label'] else 'unknown',
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1),
                'area': abs((x2 - x1) * (y2 - y1))
            }
            rooms.append(room)
        
        return rooms
    
    def parse_code_to_elements(self, code: str) -> Dict[str, List]:
        """
        Parse generated code into drawable elements.
        
        Args:
            code: Generated AutoCAD code
            
        Returns:
            Dictionary with rectangles, text labels
        """
        elements = {
            'rectangles': [],
            'labels': []
        }
        
        lines = code.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse RECT commands
            if line.startswith('RECT'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        x1 = float(parts[1])
                        y1 = float(parts[2])
                        x2 = float(parts[3])
                        y2 = float(parts[4])
                        
                        # Extract label from comment
                        label = ""
                        if ';' in line:
                            label = line.split(';')[1].strip()
                        
                        elements['rectangles'].append({
                            'x1': x1, 'y1': y1,
                            'x2': x2, 'y2': y2,
                            'label': label
                        })
                    except (ValueError, IndexError):
                        pass
            
            # Parse TEXT commands
            elif line.startswith('TEXT'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        # Extract text between quotes
                        text = line.split('"')[1] if '"' in line else ""
                        
                        elements['labels'].append({
                            'x': x, 'y': y,
                            'text': text
                        })
                    except (ValueError, IndexError):
                        pass
        
        return elements
    
    def code_to_dxf(self, code: str, output_file: str = "output.dxf"):
        """
        Convert generated code to DXF file.
        
        Args:
            code: Generated AutoCAD code
            output_file: Path to save DXF file
        """
        # Create new DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Parse code into elements
        elements = self.parse_code_to_elements(code)
        
        # Add rectangles (rooms)
        for rect in elements['rectangles']:
            # Create rectangle using LWPOLYLINE
            points = [
                (rect['x1'], rect['y1']),
                (rect['x2'], rect['y1']),
                (rect['x2'], rect['y2']),
                (rect['x1'], rect['y2']),
                (rect['x1'], rect['y1'])  
            ]
            msp.add_lwpolyline(points)
        
        # Add text labels
        for label in elements['labels']:
            msp.add_text(
                label['text'],
                dxfattribs={
                    'insert': (label['x'], label['y']),
                    'height': 2.5
                }
            )
        
        # Save DXF file
        doc.saveas(output_file)
        print(f"DXF file saved to: {output_file}")
    
    def visualize_floor_plan(
        self,
        code: str,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Visualize generated floor plan.
        
        Args:
            code: Generated AutoCAD code
            output_file: Path to save visualization (None = display only)
            figsize: Figure size
        """
        elements = self.parse_code_to_elements(code)
        
        if not elements['rectangles']:
            print("No rectangles found in code")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for different room types
        room_colors = {
            'living_room': '#FFE5B4',
            'bedroom': '#B4D7FF',
            'master_bedroom': '#9BB4FF',
            'kitchen': '#FFB4B4',
            'bathroom': '#B4FFD7',
            'dining_room': '#FFD4B4',
            'balcony': '#D4FFB4',
            'storage': '#E5E5E5',
            'entrance': '#FFE4F0'
        }
        
        # Draw rectangles
        for rect in elements['rectangles']:
            x = rect['x1']
            y = rect['y1']
            width = rect['x2'] - rect['x1']
            height = rect['y2'] - rect['y1']
            
            # Get color based on label
            label = rect['label'].lower()
            color = '#F0F0F0'  # Default
            for room_type, room_color in room_colors.items():
                if room_type in label:
                    color = room_color
                    break
            
            # Draw rectangle
            rectangle = patches.Rectangle(
                (x, y), width, height,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rectangle)
            
            # Add label in center
            if rect['label']:
                center_x = x + width / 2
                center_y = y + height / 2
                ax.text(
                    center_x, center_y,
                    rect['label'].replace('_', ' ').title(),
                    ha='center', va='center',
                    fontsize=10,
                    fontweight='bold'
                )
        
        # Add text labels
        for label in elements['labels']:
            ax.text(
                label['x'], label['y'],
                label['text'],
                ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Set aspect and limits
        ax.set_aspect('equal')
        ax.autoscale()
        ax.margins(0.1)
        
        # Styling
        ax.set_xlabel('X (feet)', fontsize=12)
        ax.set_ylabel('Y (feet)', fontsize=12)
        ax.set_title('Generated Floor Plan', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save or display
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_from_prompt(
        self,
        description: str,
        output_dir: str = None,
        save_code: bool = True,
        save_dxf: bool = True,
        save_visualization: bool = True
    ) -> Dict[str, str]:
        """
        Complete pipeline: text -> code -> DXF -> visualization.
        
        Args:
            description: Text description of floor plan
            output_dir: Directory to save outputs
            save_code: Whether to save generated code
            save_dxf: Whether to save DXF file
            save_visualization: Whether to save visualization
            
        Returns:
            Dictionary with paths to generated files
        """
        # Setup output directory
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "outputs"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating floor plan from description:")
        print(f"{'='*60}")
        print(f"{description}")
        print(f"{'='*60}\n")
        
        # Generate code
        print("Generating AutoCAD code...")
        code = self.generate_code(description)
        
        results = {'code': code}
        
        # Save code
        if save_code:
            code_file = output_dir / "generated_code.txt"
            with open(code_file, 'w') as f:
                f.write(f"Description: {description}\n\n")
                f.write("Generated Code:\n")
                f.write("="*60 + "\n")
                f.write(code)
            results['code_file'] = str(code_file)
            print(f"Code saved to: {code_file}")
        
        # Save DXF
        if save_dxf:
            dxf_file = output_dir / "generated_floor_plan.dxf"
            try:
                self.code_to_dxf(code, str(dxf_file))
                results['dxf_file'] = str(dxf_file)
                print(f"DXF file saved to: {dxf_file}")
            except Exception as e:
                print(f"Error creating DXF: {e}")
        
        # Save visualization
        if save_visualization:
            vis_file = output_dir / "floor_plan_visualization.png"
            try:
                self.visualize_floor_plan(code, str(vis_file))
                results['visualization'] = str(vis_file)
                print(f"Visualization saved to: {vis_file}")
            except Exception as e:
                print(f"Error creating visualization: {e}")
        
        print(f"\n{'='*60}")
        print("Generation complete!")
        print(f"{'='*60}\n")
        
        return results


def main():
    """Test inference module."""
    # Create generator
    generator = CADGenerator()
    
    # Test prompts
    test_prompts = [
        "A 1200 sq ft floor plan with 2 bedrooms and 1 kitchen",
        "Floor plan featuring 3 bedrooms, 1 kitchen, 1 living room, and 2 bathrooms",
        "Small apartment with 1 bedroom, kitchen, and bathroom, approximately 800 sq ft"
    ]
    
    # Generate for each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"\n\n{'='*60}")
        print(f"Test {i+1}/{len(test_prompts)}")
        print(f"{'='*60}")
        
        results = generator.generate_from_prompt(prompt)
        
        print("\nGenerated Code Preview:")
        print("-" * 60)
        print(results['code'][:500] + "...")


if __name__ == "__main__":
    main()
