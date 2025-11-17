"""
CADlingo FastAPI Production API

RESTful API endpoint for text-to-CAD generation.
Supports JSON requests and responses for integration with other systems.

Endpoints:
- POST /generate: Generate AutoCAD code from text description
- POST /generate-with-validation: Generate with quality assessment
- GET /health: Health check endpoint
- GET /metrics: Model performance metrics
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
from pathlib import Path
import tempfile
import json
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'models'))
sys.path.insert(0, str(project_root / 'src' / 'improvements'))

from inference import CADGenerator
from improvement_modules import GeometricValidator, AutomatedEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="CADlingo API",
    description="AI-powered Text-to-CAD Floor Plan Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on deployment needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance (loaded once at startup)
generator = None
validator = GeometricValidator()
evaluator = AutomatedEvaluator()


# Request/Response models
class GenerationRequest(BaseModel):
    description: str = Field(
        ...,
        description="Natural language description of the floor plan",
        example="A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen, and 1 living room"
    )
    num_beams: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of beams for generation quality (higher = better)"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature for creativity"
    )
    max_length: int = Field(
        default=512,
        ge=128,
        le=1024,
        description="Maximum code length"
    )


class GenerationResponse(BaseModel):
    success: bool
    code: str
    description: str
    generation_time: float
    metadata: Dict


class ValidationResponse(BaseModel):
    success: bool
    code: str
    description: str
    validation_report: Dict
    quality_score: float
    generation_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# Startup event - load model
@app.on_event("startup")
async def startup_event():
    """Load the CAD generator model on startup."""
    global generator
    try:
        generator = CADGenerator()
        print("✓ CADlingo model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        generator = None


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        Health status and model availability
    """
    return HealthResponse(
        status="healthy" if generator is not None else "unhealthy",
        model_loaded=generator is not None,
        version="1.0.0"
    )


# Main generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_floor_plan(request: GenerationRequest):
    """
    Generate AutoCAD code from natural language description.
    
    Args:
        request: Generation request with description and parameters
        
    Returns:
        Generated AutoCAD code and metadata
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the API service."
        )
    
    start_time = time.time()
    
    try:
        # Generate code
        code = generator.generate_code(
            request.description,
            num_beams=request.num_beams,
            temperature=request.temperature,
            max_length=request.max_length
        )
        
        generation_time = time.time() - start_time
        
        # Extract metadata
        import re
        room_count = len(re.findall(r'RECT', code))
        
        return GenerationResponse(
            success=True,
            code=code,
            description=request.description,
            generation_time=generation_time,
            metadata={
                "room_count": room_count,
                "code_length": len(code),
                "num_beams": request.num_beams,
                "temperature": request.temperature
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {str(e)}"
        )


# Generation with validation endpoint
@app.post("/generate-with-validation", response_model=ValidationResponse)
async def generate_with_validation(request: GenerationRequest):
    """
    Generate AutoCAD code with geometric validation.
    
    Args:
        request: Generation request with description and parameters
        
    Returns:
        Generated code, validation report, and quality score
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the API service."
        )
    
    start_time = time.time()
    
    try:
        # Generate code
        code = generator.generate_code(
            request.description,
            num_beams=request.num_beams,
            temperature=request.temperature,
            max_length=request.max_length
        )
        
        # Parse rooms for validation
        import re
        rect_pattern = r'RECT\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"([^"]+)"'
        room_matches = re.findall(rect_pattern, code)
        
        parsed_rooms = []
        for x, y, w, h, room_type in room_matches:
            x, y, w, h = float(x), float(y), float(w), float(h)
            parsed_rooms.append({
                'type': room_type,
                'center': (x + w/2, y + h/2),
                'width': w,
                'height': h,
                'area': w * h
            })
        
        # Validate
        validation_report = validator.validate_floor_plan(parsed_rooms)
        generation_time = time.time() - start_time
        
        return ValidationResponse(
            success=True,
            code=code,
            description=request.description,
            validation_report=validation_report,
            quality_score=validation_report['overall_score'],
            generation_time=generation_time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation/validation error: {str(e)}"
        )


# Download DXF endpoint
@app.post("/generate-dxf")
async def generate_dxf(request: GenerationRequest):
    """
    Generate and download DXF file.
    
    Args:
        request: Generation request
        
    Returns:
        DXF file download
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Generate code
        code = generator.generate_code(
            request.description,
            num_beams=request.num_beams,
            temperature=request.temperature
        )
        
        # Create temporary DXF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dxf', delete=False) as tmp:
            generator.code_to_dxf(code, tmp.name)
            tmp_path = tmp.name
        
        return FileResponse(
            tmp_path,
            media_type="application/dxf",
            filename="floor_plan.dxf"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DXF generation error: {str(e)}"
        )


# Batch generation endpoint
@app.post("/batch-generate")
async def batch_generate(descriptions: List[str], num_beams: int = 4):
    """
    Generate multiple floor plans in batch.
    
    Args:
        descriptions: List of floor plan descriptions
        num_beams: Generation quality parameter
        
    Returns:
        List of generated codes with metadata
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(descriptions) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 descriptions per batch"
        )
    
    results = []
    
    for desc in descriptions:
        try:
            code = generator.generate_code(desc, num_beams=num_beams)
            results.append({
                "success": True,
                "description": desc,
                "code": code
            })
        except Exception as e:
            results.append({
                "success": False,
                "description": desc,
                "error": str(e)
            })
    
    return {"results": results, "total": len(descriptions)}


# Model metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """
    Get model performance metrics.
    
    Returns:
        Model statistics and capabilities
    """
    return {
        "model": "CodeT5-base",
        "parameters": "220M",
        "training_samples": "1000+",
        "supported_rooms": [
            "living_room", "bedroom", "master_bedroom", "kitchen",
            "dining_room", "bathroom", "balcony", "storage",
            "hallway", "office", "laundry"
        ],
        "capabilities": {
            "text_to_code": True,
            "code_to_dxf": True,
            "visualization": True,
            "validation": True
        },
        "api_version": "1.0.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """
    API root endpoint with welcome message.
    """
    return {
        "message": "CADlingo API - Text to CAD Floor Plan Generation",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "generate": "POST /generate",
            "generate_with_validation": "POST /generate-with-validation",
            "generate_dxf": "POST /generate-dxf",
            "batch_generate": "POST /batch-generate",
            "metrics": "GET /metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("CADlingo API Server")
    print("=" * 80)
    print("Starting server on http://0.0.0.0:8000")
    print("API Documentation: http://0.0.0.0:8000/docs")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
