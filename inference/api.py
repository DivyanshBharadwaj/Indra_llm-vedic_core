"""
API server for INDRA LLM inference with FastAPI
(c) Divyansh Bharadwaj
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .generation import InferenceEngine, GenerationConfig
from .vedic_inference import VedicInferenceEngine, VedicGenerationConfig, VedicMode
from model import INDRATransformer

# Global inference engines
inference_engine: Optional[InferenceEngine] = None
vedic_engine: Optional[VedicInferenceEngine] = None

# Request/Response models
class GenerationRequest(BaseModel):
    """Standard generation request."""
    prompt: Union[str, List[str]] = Field(..., description="Input prompt(s)")
    max_new_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, le=1000, description="Top-k sampling")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Nucleus sampling")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Use sampling vs greedy decoding")
    num_beams: int = Field(1, ge=1, le=8, description="Number of beams for beam search")
    use_cache: bool = Field(True, description="Use KV caching")

class VedicGenerationRequest(GenerationRequest):
    """Vedic-guided generation request."""
    vedic_mode: str = Field("balanced", description="Vedic reasoning mode")
    vedic_weight: float = Field(1.0, ge=0.1, le=3.0, description="Vedic guidance weight")
    philosophical_depth: int = Field(3, ge=1, le=5, description="Philosophical depth")
    sanskrit_preference: float = Field(0.2, ge=0.0, le=1.0, description="Sanskrit integration preference")
    include_analysis: bool = Field(True, description="Include Vedic analysis in response")
    context_texts: Optional[List[str]] = Field(None, description="Additional Vedic context texts")

class GenerationResponse(BaseModel):
    """Generation response model."""
    generated_text: Union[str, List[str]]
    generation_stats: Dict[str, Any]
    model_info: Dict[str, str]
    processing_time: float

class VedicGenerationResponse(BaseModel):
    """Vedic generation response model."""
    generated_text: str
    vedic_analysis: Dict[str, Any]
    enhanced_prompt: str
    vedic_config: Dict[str, Any]
    generation_stats: Dict[str, Any]
    model_info: Dict[str, str]
    processing_time: float

class VedicAnalysisRequest(BaseModel):
    """Request for Vedic text analysis."""
    text: str = Field(..., description="Text to analyze")
    include_insights: bool = Field(True, description="Include philosophical insights")
    suggest_practices: bool = Field(False, description="Suggest related practices")

class VedicDialogueRequest(BaseModel):
    """Request for Vedic philosophical dialogue."""
    topic: str = Field(..., description="Topic for discussion")
    participants: Optional[List[str]] = Field(None, description="Dialogue participants")
    turns: int = Field(6, ge=2, le=20, description="Number of dialogue turns")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    vedic_engine_available: bool
    uptime_seconds: float
    stats: Dict[str, Any]

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global inference_engine, vedic_engine
    
    # Startup
    logging.info("Starting INDRA LLM API server...")
    
    # Initialize engines (these would be loaded from saved models)
    # For now, we'll create placeholders
    # In production, you would load actual trained models here
    
    yield
    
    # Shutdown
    logging.info("Shutting down INDRA LLM API server...")
    if inference_engine:
        inference_engine.clear_cache()
    if vedic_engine:
        vedic_engine.clear_cache()

# Create FastAPI app
app = FastAPI(
    title="INDRA LLM API",
    description="Vedic-aligned Large Language Model API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time
startup_time = time.time()

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "INDRA LLM API - Vedic-aligned Language Model",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    
    stats = {}
    if inference_engine:
        stats = inference_engine.get_stats()
    
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine is not None,
        vedic_engine_available=vedic_engine is not None,
        uptime_seconds=uptime,
        stats=stats
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using standard inference."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    start_time = time.time()
    
    try:
        # Create generation config
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            num_beams=request.num_beams,
            use_cache=request.use_cache
        )
        
        # Generate
        result = inference_engine.generate(
            prompt=request.prompt,
            generation_config=config,
            return_dict=True
        )
        
        processing_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=result["generated_texts"],
            generation_stats=result["generation_stats"],
            model_info={
                "model_name": "INDRA LLM",
                "model_type": "Decoder-only Transformer",
                "vedic_aligned": "true"
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/vedic", response_model=VedicGenerationResponse)
async def generate_vedic_text(request: VedicGenerationRequest):
    """Generate text with Vedic guidance."""
    if not vedic_engine:
        raise HTTPException(status_code=503, detail="Vedic inference engine not loaded")
    
    start_time = time.time()
    
    try:
        # Parse Vedic mode
        try:
            vedic_mode = VedicMode(request.vedic_mode.lower())
        except ValueError:
            vedic_mode = VedicMode.BALANCED
        
        # Create Vedic generation config
        config = VedicGenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            num_beams=request.num_beams,
            use_cache=request.use_cache,
            vedic_mode=vedic_mode,
            vedic_weight=request.vedic_weight,
            philosophical_depth=request.philosophical_depth,
            sanskrit_preference=request.sanskrit_preference
        )
        
        # Generate with Vedic guidance
        result = vedic_engine.generate_vedic(
            prompt=request.prompt,
            vedic_config=config,
            context_texts=request.context_texts
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "generated_text": result["generated_text"],
            "enhanced_prompt": result["enhanced_prompt"],
            "generation_stats": result["generation_stats"],
            "vedic_config": {
                "vedic_mode": vedic_mode.value,
                "vedic_weight": request.vedic_weight,
                "philosophical_depth": request.philosophical_depth,
                "sanskrit_preference": request.sanskrit_preference
            },
            "model_info": {
                "model_name": "INDRA LLM",
                "model_type": "Vedic-aligned Decoder-only Transformer",
                "vedic_core": "active"
            },
            "processing_time": processing_time
        }
        
        # Include analysis if requested
        if request.include_analysis:
            response_data["vedic_analysis"] = result["vedic_analysis"]
        else:
            response_data["vedic_analysis"] = {"overall_alignment": result["vedic_analysis"]["overall_alignment"]}
        
        return VedicGenerationResponse(**response_data)
        
    except Exception as e:
        logging.error(f"Vedic generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Vedic generation failed: {str(e)}")

@app.post("/generate/stream")
async def stream_generate(request: GenerationRequest):
    """Stream text generation."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    try:
        # Create generation config
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            use_cache=request.use_cache
        )
        
        async def generate_stream():
            try:
                for token in inference_engine.stream_generate(request.prompt, config):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR: {str(e)}]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logging.error(f"Stream generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Stream generation failed: {str(e)}")

@app.post("/analyze/vedic")
async def analyze_vedic_text(request: VedicAnalysisRequest):
    """Analyze text for Vedic alignment and insights."""
    if not vedic_engine:
        raise HTTPException(status_code=503, detail="Vedic inference engine not loaded")
    
    start_time = time.time()
    
    try:
        if request.include_insights:
            result = vedic_engine.get_vedic_insights(request.text)
        else:
            # Just basic analysis
            config = VedicGenerationConfig()
            analysis = vedic_engine._analyze_vedic_alignment("", request.text, config)
            result = {
                "original_text": request.text,
                "vedic_analysis": analysis
            }
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        return result
        
    except Exception as e:
        logging.error(f"Vedic analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Vedic analysis failed: {str(e)}")

@app.post("/dialogue/vedic")
async def create_vedic_dialogue(request: VedicDialogueRequest):
    """Create a Vedic philosophical dialogue."""
    if not vedic_engine:
        raise HTTPException(status_code=503, detail="Vedic inference engine not loaded")
    
    start_time = time.time()
    
    try:
        result = vedic_engine.generate_vedic_dialogue(
            topic=request.topic,
            participants=request.participants,
            turns=request.turns
        )
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        return result
        
    except Exception as e:
        logging.error(f"Vedic dialogue error: {e}")
        raise HTTPException(status_code=500, detail=f"Vedic dialogue failed: {str(e)}")

@app.post("/explain/vedic")
async def explain_vedic_reasoning(
    question: str,
    principles: Optional[List[str]] = None,
    depth: int = 3
):
    """Explain Vedic reasoning for a question."""
    if not vedic_engine:
        raise HTTPException(status_code=503, detail="Vedic inference engine not loaded")
    
    start_time = time.time()
    
    try:
        result = vedic_engine.explain_vedic_reasoning(
            question=question,
            principles=principles,
            depth=min(max(depth, 1), 5)  # Clamp between 1-5
        )
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        return result
        
    except Exception as e:
        logging.error(f"Vedic reasoning error: {e}")
        raise HTTPException(status_code=500, detail=f"Vedic reasoning failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    info = {
        "base_model": {
            "name": "INDRA LLM",
            "type": "Decoder-only Transformer", 
            "architecture": "GPT-style with FlashAttention, GQA, and MoE",
            "loaded": inference_engine is not None
        },
        "vedic_engine": {
            "name": "INDRA Vedic Engine",
            "type": "Philosophically-guided inference",
            "features": ["Dharmic alignment", "Sanskrit integration", "Concept reasoning"],
            "loaded": vedic_engine is not None
        }
    }
    
    if inference_engine:
        stats = inference_engine.get_stats()
        info["base_model"]["stats"] = stats
    
    return info

@app.post("/cache/clear")
async def clear_cache(background_tasks: BackgroundTasks):
    """Clear model caches."""
    def clear_all_caches():
        if inference_engine:
            inference_engine.clear_cache()
        if vedic_engine:
            vedic_engine.clear_cache()
        logging.info("All caches cleared")
    
    background_tasks.add_task(clear_all_caches)
    return {"message": "Cache clearing initiated"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logging.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": time.time()
    }

# Utility functions for creating API server
def create_api_server(
    model: Optional[INDRATransformer] = None,
    tokenizer = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    **kwargs
) -> None:
    """
    Create and run API server.
    
    Args:
        model: INDRA transformer model
        tokenizer: Tokenizer instance
        host: Host address
        port: Port number
        workers: Number of worker processes
        **kwargs: Additional uvicorn arguments
    """
    global inference_engine, vedic_engine
    
    # Initialize inference engines if model provided
    if model and tokenizer:
        inference_engine = InferenceEngine(model, tokenizer)
        
        # Create Vedic engine if model has Vedic capabilities
        if hasattr(model.config, 'vedic') and model.config.vedic.use_vedic_core:
            from ..tokenization import VedicTokenizer
            if isinstance(tokenizer, VedicTokenizer):
                vedic_engine = VedicInferenceEngine(model, tokenizer)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        **kwargs
    )

def load_models_for_api(
    model_path: str,
    tokenizer_path: str,
    device: Optional[str] = None
) -> tuple[INDRATransformer, Any]:
    """
    Load models for API server.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device to load models on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from ..model import INDRATransformer
    from ..tokenization import VedicTokenizer, SentencePieceTokenizer
    from ..training.trainer_utils import TrainerUtils
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load tokenizer
    try:
        tokenizer = VedicTokenizer(model_path=tokenizer_path)
        logging.info("Loaded Vedic tokenizer")
    except Exception:
        tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        logging.info("Loaded SentencePiece tokenizer")
    
    # Load model configuration and create model
    # This would need to be implemented based on your saved model format
    # For now, we'll create a placeholder
    
    from ..config import ModelConfig
    
    # You would load the actual config from the checkpoint
    config = ModelConfig()  # This should be loaded from saved model
    
    model = INDRATransformer(config)
    
    # Load model weights
    TrainerUtils.load_checkpoint(
        model_path,
        model=model,
        device=device
    )
    
    model.eval()
    logging.info(f"Loaded model on {device}")
    
    return model, tokenizer

# CLI for running API server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="INDRA LLM API Server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Load models
    model, tokenizer = load_models_for_api(
        args.model_path,
        args.tokenizer_path,
        args.device
    )
    
    # Create and run server
    create_api_server(
        model=model,
        tokenizer=tokenizer,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )
