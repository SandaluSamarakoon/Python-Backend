import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import uvicorn
from typing import Dict, Any
import logging
import soundfile as sf
import wave
import audioop
import base64
import subprocess

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Emotion Recognition API")

# Configure CORS for production - Allow your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000", 
        "https://*.vercel.app",  # Allow all Vercel domains
        "https://mindbridge-alpha.vercel.app",  # Replace with your actual Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Model path - Azure App Service will have the model in the root directory
model_path = "./model.keras"

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Check if model file exists
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # You might want to download the model from Azure Blob Storage or another location
            # For now, we'll handle this in the prediction endpoint
    except Exception as e:
        logger.error(f"Error loading model during startup: {e}")

def is_valid_wav(file_path):
    """Check if a WAV file is valid by trying to open it with wave"""
    try:
        with wave.open(file_path, 'rb') as wf:
            return True
    except Exception as e:
        logger.error(f"Invalid WAV file: {str(e)}")
        return False

def fix_wav_header(file_path):
    """Attempt to fix WAV file header issues by rewriting it"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        if not data.startswith(b'RIFF'):
            logger.warning("WAV file doesn't start with RIFF header. Attempting to fix...")
            
            with open(file_path + '.fixed.wav', 'wb') as f:
                channels = 1
                sample_width = 2
                sample_rate = 44100
                
                audio_data = data
                
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
            return file_path + '.fixed.wav'
    except Exception as e:
        logger.error(f"Error fixing WAV header: {str(e)}")
    
    return file_path

def extract_features(file_path, n_mfcc=40):
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing audio file: {file_path}, size: {file_size} bytes")
        
        if not is_valid_wav(file_path):
            fixed_path = fix_wav_header(file_path)
            if fixed_path != file_path:
                logger.info(f"Using fixed WAV file: {fixed_path}")
                file_path = fixed_path
        
        try:
            audio, sample_rate = sf.read(file_path)
            logger.info(f"Loaded audio with soundfile, shape: {audio.shape}, sample_rate: {sample_rate}")
        except Exception as sf_error:
            logger.warning(f"Soundfile failed: {str(sf_error)}, trying librosa...")
            try:
                audio, sample_rate = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
                logger.info(f"Loaded audio with librosa, length: {len(audio)}, sample_rate: {sample_rate}")
            except Exception as librosa_error:
                logger.error(f"Librosa also failed: {str(librosa_error)}")
                try:
                    with wave.open(file_path, 'rb') as wf:
                        frames = wf.getnframes()
                        buffer = wf.readframes(frames)
                        sample_rate = wf.getframerate()
                        sample_width = wf.getsampwidth()
                        channels = wf.getnchannels()
                        
                        if channels == 2:
                            buffer = audioop.tomono(buffer, sample_width, 0.5, 0.5)
                        
                        if sample_width == 2:
                            audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:
                            audio = np.frombuffer(buffer, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            audio = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        
                        logger.info(f"Loaded audio with wave, length: {len(audio)}, sample_rate: {sample_rate}")
                except Exception as wave_error:
                    logger.error(f"All audio loading methods failed: {str(wave_error)}")
                    raise Exception(f"Could not load audio with any method: {str(sf_error)} | {str(librosa_error)} | {str(wave_error)}")
        
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        logger.info(f"Audio prepared: shape={audio.shape}, sample_rate={sample_rate}")
        
        if len(audio) < sample_rate * 0.5:
            logger.warning("Audio too short, padding")
            audio = np.pad(audio, (0, int(sample_rate * 0.5) - len(audio)), 'constant')
        
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            logger.info(f"MFCCs extracted: shape={mfccs.shape}")
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {str(e)}")
            logger.warning("Using fallback dummy features")
            mfccs = np.random.rand(n_mfcc, max(1, int(len(audio) / 512)))
        
        mfccs = np.mean(mfccs, axis=1).reshape(n_mfcc, 1)
        return mfccs
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Voice Emotion Recognition API is running on Azure App Service",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Voice Emotion Recognition API"
    }

def convert_webm_to_wav(webm_path, wav_path):
    """Convert WebM audio to WAV format using ffmpeg"""
    try:
        subprocess.run([
            'ffmpeg', '-i', webm_path, '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-ac', '1', wav_path, '-y'
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg.")
        return False

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

@app.post("/predict")
async def predict_emotion(
    audio_file: UploadFile = File(...), 
    expected_word: str = Form(None), 
    category: str = Form(None),
    return_audio: bool = Form(False)
) -> Dict[str, Any]:
    global model
    
    logger.info(f"Received prediction request: file={audio_file.filename}, content_type={audio_file.content_type}")
    logger.info(f"Parameters: expected_word={expected_word}, category={category}, return_audio={return_audio}")
    
    # Try to load model if not already loaded
    if model is None:
        try:
            logger.info(f"Attempting to load model from {model_path}")
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model file not found at {model_path}")
                raise HTTPException(
                    status_code=500, 
                    detail="Model file not found. Please ensure model.keras is uploaded to your repository."
                )
        except Exception as e:
            logger.error(f"Model could not be loaded: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Model could not be loaded: {str(e)}"
            )
    
    valid_audio_types = [".wav", ".webm", ".mp3", ".ogg", ".m4a"]
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    if not file_ext and (audio_file.content_type == 'audio/webm' or 
                        audio_file.content_type == 'video/webm'):
        file_ext = '.webm'
    
    if not file_ext and audio_file.content_type == 'audio/wav':
        file_ext = '.wav'
    
    if not file_ext and not any(audio_file.content_type.startswith(f"audio/{t.replace('.', '')}") for t in valid_audio_types):
        logger.warning(f"Unsupported content type: {audio_file.content_type}")
        file_ext = '.webm'
    
    temp_file_path = None
    wav_file_path = None
    processed_wav_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)
        
        logger.info(f"Saved uploaded file to {temp_file_path}, size: {len(content)} bytes")
        
        if file_ext == '.webm':
            wav_file_path = temp_file_path.replace('.webm', '_converted.wav')
            logger.info(f"Converting WebM to WAV: {temp_file_path} -> {wav_file_path}")
            
            if convert_webm_to_wav(temp_file_path, wav_file_path):
                logger.info("WebM to WAV conversion successful")
                processing_file_path = wav_file_path
            else:
                logger.warning("WebM conversion failed, trying direct processing")
                processing_file_path = temp_file_path
        else:
            processing_file_path = temp_file_path
        
        if return_audio:
            processed_wav_path = processing_file_path.replace(file_ext, '_processed.wav')
            try:
                audio_data, sample_rate = librosa.load(processing_file_path, sr=22050, mono=True)
                sf.write(processed_wav_path, audio_data, sample_rate)
                logger.info(f"Created processed WAV file: {processed_wav_path}")
            except Exception as e:
                logger.warning(f"Could not create processed WAV: {str(e)}")
                processed_wav_path = None
        
        try:
            features = extract_features(processing_file_path)
            features = np.expand_dims(features, axis=0)
            logger.info(f"Features extracted successfully, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {
                "filename": audio_file.filename,
                "predicted_emotion": "neutral",
                "confidence_scores": {
                    "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                    "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
                },
                "error": str(e),
                "matches": False,
                "audio_base64": None
            }
        
        logger.info("Making prediction")
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)
        
        emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        confidence_scores = prediction[0].tolist()
        
        logger.info(f"Prediction result: {emotions[predicted_label]}")
        
        audio_base64 = None
        if return_audio and processed_wav_path and os.path.exists(processed_wav_path):
            try:
                with open(processed_wav_path, 'rb') as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                logger.info("Audio file encoded to base64")
            except Exception as e:
                logger.warning(f"Could not encode audio to base64: {str(e)}")
        
        # Cleanup
        cleanup_files = [temp_file_path]
        if wav_file_path:
            cleanup_files.append(wav_file_path)
        if processed_wav_path:
            cleanup_files.append(processed_wav_path)
        if os.path.exists(temp_file_path + '.fixed.wav'):
            cleanup_files.append(temp_file_path + '.fixed.wav')
            
        for file_path in cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {file_path}: {str(e)}")
        
        return {
            "filename": audio_file.filename,
            "predicted_emotion": emotions[predicted_label],
            "confidence_scores": {
                emotions[i]: float(confidence_scores[i]) for i in range(len(emotions))
            },
            "text": "placeholder for transcription",
            "matches": True if predicted_label == 0 else False,
            "audio_base64": audio_base64
        }
        
    except Exception as e:
        # Cleanup on error
        cleanup_files = []
        if temp_file_path:
            cleanup_files.extend([temp_file_path, temp_file_path + '.fixed.wav'])
        if wav_file_path:
            cleanup_files.append(wav_file_path)
        if processed_wav_path:
            cleanup_files.append(processed_wav_path)
            
        for file_path in cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except:
                pass
        
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        return {
            "filename": audio_file.filename,
            "predicted_emotion": "neutral",
            "confidence_scores": {
                "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
            },
            "text": "placeholder for transcription",
            "error": str(e),
            "matches": False,
            "audio_base64": None
        }

# For Azure App Service, we need to expose the app for gunicorn
if __name__ == "__main__":
    # This will only run when testing locally
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)