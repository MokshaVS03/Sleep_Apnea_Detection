from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import torch
import numpy as np
from datetime import datetime
import uvicorn

# Import your model classes from main.py
from main import ApneaCNN, ECGDataCollector, preprocess_ecg

# ===== GLOBAL VARIABLES =====
MODEL = None
DEVICE = None
COLLECTOR = None
MODEL_PATH = "./chestxray_best_model_1_val_loss=-0.1384.pt"
SERIAL_PORT = "COM11"

# ===== PYDANTIC MODELS =====
class ECGData(BaseModel):
    samples: List[int] = Field(..., description="Raw ECG ADC values (0-4095)")
    sampling_rate: Optional[int] = Field(100, description="Sampling rate in Hz")

class PredictionResponse(BaseModel):
    probability: float
    prediction: str
    confidence: float
    timestamp: str
    samples_analyzed: int

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    serial_connected: bool
    device: str

class CollectionRequest(BaseModel):
    num_samples: int = Field(1200, ge=100, le=12000)
    timeout: int = Field(30, ge=10, le=120)

# ===== LIFESPAN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, DEVICE, COLLECTOR
    
    # Startup
    try:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {DEVICE}")
        
        print("🔧 Loading model...")
        MODEL = ApneaCNN(input_channels=1, input_length=12000)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(checkpoint)
        MODEL.to(DEVICE)
        MODEL.eval()
        print("✅ Model loaded successfully!")
        
        print(f"📡 Attempting to connect to {SERIAL_PORT}...")
        try:
            COLLECTOR = ECGDataCollector(SERIAL_PORT)
            print("✅ Serial connection established!")
        except Exception as serial_error:
            print(f"⚠️ Serial connection failed: {serial_error}")
            print("⚠️ Server will start without serial connection.")
            COLLECTOR = None
        
        print("\n✅ Server startup complete!")
        print(f"   Model: {'Loaded' if MODEL else 'Not Loaded'}")
        print(f"   Serial: {'Connected' if COLLECTOR else 'Not Connected'}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Critical startup error: {e}")
    
    yield
    
    # Shutdown
    if COLLECTOR:
        try:
            COLLECTOR.close()
            print("🔌 Serial connection closed")
        except:
            pass

# ===== CREATE APP =====
app = FastAPI(
    title="ECG Sleep Apnea Detection API",
    description="Real-time ECG analysis for sleep apnea detection",
    version="1.0.0",
    lifespan=lifespan
)

# ===== ADD CORS MIDDLEWARE =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS =====
@app.get("/")
async def root():
    return {
        "message": "ECG Sleep Apnea Detection API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/status",
            "predict_from_data": "/predict",
            "collect_and_predict": "/collect-and-predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        status="operational" if MODEL and COLLECTOR else "degraded",
        model_loaded=MODEL is not None,
        serial_connected=COLLECTOR is not None,
        device=str(DEVICE)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_from_data(ecg_data: ECGData):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        raw_ecg = np.array(ecg_data.samples)
        
        if len(raw_ecg) < 100:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient samples: {len(raw_ecg)} (minimum 100)"
            )
        
        processed_ecg = preprocess_ecg(raw_ecg, target_length=12000)
        ecg_tensor = torch.FloatTensor(processed_ecg).unsqueeze(0).unsqueeze(0)
        ecg_tensor = ecg_tensor.to(DEVICE)
        
        with torch.no_grad():
            prediction = MODEL(ecg_tensor)
            prob = prediction.item()
        
        pred_label = "APNEA" if prob > 0.5 else "NORMAL"
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
        
        return PredictionResponse(
            probability=prob,
            prediction=pred_label,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            samples_analyzed=len(raw_ecg)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/collect-and-predict", response_model=PredictionResponse)
async def collect_and_predict(request: CollectionRequest):
    if MODEL is None or COLLECTOR is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or serial connection not available"
        )
    
    try:
        print(f"📡 Collecting {request.num_samples} samples...")
        raw_ecg = COLLECTOR.collect_ecg_segment(
            num_samples=request.num_samples,
            timeout=request.timeout
        )
        
        if raw_ecg is None or len(raw_ecg) == 0:
            print("❌ ERROR: No data collected from ESP32")
            raise HTTPException(
                status_code=500,
                detail="No data collected from ESP32. Check serial connection."
            )
        
        if len(raw_ecg) < 100:
            print(f"⚠️ WARNING: Only {len(raw_ecg)} samples collected")
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: only {len(raw_ecg)} samples"
            )
        
        print(f"✅ Collected {len(raw_ecg)} samples, preprocessing...")
        
        try:
            processed_ecg = preprocess_ecg(raw_ecg, target_length=12000)
        except Exception as prep_error:
            print(f"❌ Preprocessing error: {prep_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Preprocessing failed: {str(prep_error)}"
            )
        
        if processed_ecg is None or len(processed_ecg) == 0:
            print("❌ ERROR: Preprocessing returned empty array")
            raise HTTPException(
                status_code=500,
                detail="Preprocessing failed - empty result"
            )
        
        print(f"✅ Preprocessed to {len(processed_ecg)} samples, running inference...")
        
        ecg_tensor = torch.FloatTensor(processed_ecg).unsqueeze(0).unsqueeze(0)
        ecg_tensor = ecg_tensor.to(DEVICE)
        
        with torch.no_grad():
            prediction = MODEL(ecg_tensor)
            prob = prediction.item()
        
        pred_label = "APNEA" if prob > 0.5 else "NORMAL"
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
        
        print(f"✅ Prediction complete: {pred_label} ({prob:.4f})")
        
        return PredictionResponse(
            probability=prob,
            prediction=pred_label,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            samples_analyzed=len(raw_ecg)
        )
        
    except HTTPException:
        raise
    except TimeoutError as e:
        print(f"❌ Timeout error: {e}")
        raise HTTPException(status_code=408, detail="Data collection timeout")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {type(e).__name__}: {str(e)}"
        )

@app.post("/collect-only")
async def collect_ecg_data(request: CollectionRequest):
    if COLLECTOR is None:
        raise HTTPException(status_code=503, detail="Serial connection not available")
    
    try:
        print(f"📡 Collecting {request.num_samples} samples (test mode)...")
        raw_ecg = COLLECTOR.collect_ecg_segment(
            num_samples=request.num_samples,
            timeout=request.timeout
        )
        
        if raw_ecg is None or len(raw_ecg) == 0:
            raise HTTPException(
                status_code=500,
                detail="No data collected from ESP32"
            )
        
        stats = {
            "min": int(np.min(raw_ecg)),
            "max": int(np.max(raw_ecg)),
            "mean": float(np.mean(raw_ecg)),
            "std": float(np.std(raw_ecg))
        }
        
        print(f"✅ Collection test complete: {len(raw_ecg)} samples")
        
        return {
            "samples": raw_ecg.tolist()[:100],
            "num_samples": len(raw_ecg),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "statistics": stats
        }
        
    except TimeoutError as e:
        print(f"❌ Timeout: {e}")
        raise HTTPException(status_code=408, detail="Data collection timeout")
    except Exception as e:
        print(f"❌ Collection error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/test-serial")
async def test_serial_connection():
    if COLLECTOR is None:
        raise HTTPException(status_code=503, detail="Serial connection not available")
    
    try:
        is_open = COLLECTOR.ser.is_open
        
        port_info = {
            "port": COLLECTOR.ser.port,
            "baudrate": COLLECTOR.ser.baudrate,
            "is_open": is_open,
            "in_waiting": COLLECTOR.ser.in_waiting if is_open else 0,
            "timeout": COLLECTOR.ser.timeout
        }
        
        print(f"✅ Serial test: {port_info}")
        
        return {
            "status": "connected" if is_open else "disconnected",
            "port_info": port_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Serial test error: {e}")
        raise HTTPException(status_code=500, detail=f"Serial test error: {str(e)}")

@app.post("/reconnect-serial")
async def reconnect_serial(port: str = SERIAL_PORT):
    global COLLECTOR
    
    try:
        if COLLECTOR is not None:
            try:
                COLLECTOR.close()
                print(f"🔌 Closed existing connection")
            except:
                pass
            COLLECTOR = None
        
        print(f"📡 Attempting to connect to {port}...")
        COLLECTOR = ECGDataCollector(port)
        print(f"✅ Connected to {port}")
        
        return {
            "status": "success",
            "message": f"Connected to {port}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Reconnection failed: {e}")
        COLLECTOR = None
        raise HTTPException(status_code=500, detail=f"Failed to connect: {str(e)}")

# ===== RUN SERVER =====
if __name__ == "__main__":
    print("🚀 Starting ECG Apnea Detection API Server")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Serial Port: {SERIAL_PORT}")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )