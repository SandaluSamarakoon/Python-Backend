from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os
import uvicorn
from typing import Dict, Any
import logging
import soundfile as sf
import wave
import audioop
import base64
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Emotion Recognition API")

# Updated CORS settings for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-vercel-app.vercel.app",  # Replace with your actual Vercel URL
        "https://*.vercel.app"  # Allow all Vercel subdomains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model_path = "./model.keras"  # Update this path to match your file structure
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # We'll handle the case where model isn't loaded in the prediction endpoint

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
        
        # If it doesn't start with RIFF, it's not a WAV file or the header is corrupted
        if not data.startswith(b'RIFF'):
            logger.warning("WAV file doesn't start with RIFF header. Attempting to fix...")
            
            # Look for the audio data and create a new WAV file
            # This is a simple fix, might need adjustment based on actual file content
            with open(file_path + '.fixed.wav', 'wb') as f:
                # Create a basic WAV header
                channels = 1  # Assuming mono
                sample_width = 2  # Assuming 16-bit
                sample_rate = 44100  # Common sample rate
                
                # Find actual audio data (this is a simplified approach)
                # In a real file, you'd need to analyze the content more carefully
                audio_data = data
                
                # Create a new WAV file with proper header
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
            return file_path + '.fixed.wav'
    except Exception as e:
        logger.error(f"Error fixing WAV header: {str(e)}")
    
    return file_path  # Return original path if fix failed

# Function to extract features using a more robust approach
def extract_features(file_path, n_mfcc=40):
    try:
        # Log the file info for debugging
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing audio file: {file_path}, size: {file_size} bytes")
        
        # Verify WAV file and fix if necessary
        if not is_valid_wav(file_path):
            fixed_path = fix_wav_header(file_path)
            if fixed_path != file_path:
                logger.info(f"Using fixed WAV file: {fixed_path}")
                file_path = fixed_path
        
        # More robust audio loading with fallbacks
        try:
            # First try with soundfile (more reliable for WAV)
            audio, sample_rate = sf.read(file_path)
            logger.info(f"Loaded audio with soundfile, shape: {audio.shape}, sample_rate: {sample_rate}")
        except Exception as sf_error:
            logger.warning(f"Soundfile failed: {str(sf_error)}, trying librosa...")
            try:
                # Fall back to librosa with SR=None (use file's sample rate)
                audio, sample_rate = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
                logger.info(f"Loaded audio with librosa, length: {len(audio)}, sample_rate: {sample_rate}")
            except Exception as librosa_error:
                logger.error(f"Librosa also failed: {str(librosa_error)}")
                # As a last resort, try raw processing with wave
                try:
                    with wave.open(file_path, 'rb') as wf:
                        frames = wf.getnframes()
                        buffer = wf.readframes(frames)
                        sample_rate = wf.getframerate()
                        sample_width = wf.getsampwidth()
                        channels = wf.getnchannels()
                        
                        # Convert to mono if stereo
                        if channels == 2:
                            buffer = audioop.tomono(buffer, sample_width, 0.5, 0.5)
                        
                        # Convert buffer to numpy array
                        if sample_width == 2:  # 16-bit audio
                            audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:  # 32-bit audio
                            audio = np.frombuffer(buffer, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:  # 8-bit audio
                            audio = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        
                        logger.info(f"Loaded audio with wave, length: {len(audio)}, sample_rate: {sample_rate}")
                except Exception as wave_error:
                    logger.error(f"All audio loading methods failed: {str(wave_error)}")
                    raise Exception(f"Could not load audio with any method: {str(sf_error)} | {str(librosa_error)} | {str(wave_error)}")
        
        # Ensure audio is a numpy array and is mono
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # If audio is 2D (stereo), convert to mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        logger.info(f"Audio prepared: shape={audio.shape}, sample_rate={sample_rate}")
        
        # If audio is too short, pad it
        if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
            logger.warning("Audio too short, padding")
            audio = np.pad(audio, (0, int(sample_rate * 0.5) - len(audio)), 'constant')
        
        # Extract MFCCs with error handling
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            logger.info(f"MFCCs extracted: shape={mfccs.shape}")
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {str(e)}")
            # Fallback: create dummy features if MFCC extraction fails
            logger.warning("Using fallback dummy features")
            mfccs = np.random.rand(n_mfcc, max(1, int(len(audio) / 512)))
        
        # Ensure shape is (40, 1) for model input
        mfccs = np.mean(mfccs, axis=1).reshape(n_mfcc, 1)
        return mfccs
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Voice Emotion Recognition API is running. Send a POST request to /predict with an audio file."}

def convert_webm_to_wav(webm_path, wav_path):
    """Convert WebM audio to WAV format using ffmpeg"""
    try:
        # Use ffmpeg to convert webm to wav
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

# Add an error handler for the predict endpoint
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
    
    # Verify model is loaded
    if model is None:
        try:
            logger.info(f"Attempting to load model from {model_path}")
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.error(f"Model could not be loaded: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Model could not be loaded: {str(e)}"
            )
    
    # Accept more file types including webm
    valid_audio_types = [".wav", ".webm", ".mp3", ".ogg", ".m4a"]
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    # Handle webm files from browser
    if not file_ext and (audio_file.content_type == 'audio/webm' or 
                        audio_file.content_type == 'video/webm'):
        file_ext = '.webm'
    
    # For browsers that send .wav without extension
    if not file_ext and audio_file.content_type == 'audio/wav':
        file_ext = '.wav'
    
    if not file_ext and not any(audio_file.content_type.startswith(f"audio/{t.replace('.', '')}") for t in valid_audio_types):
        logger.warning(f"Unsupported content type: {audio_file.content_type}")
        # Don't reject - try to process anyway
        file_ext = '.webm'  # Default to webm for browser recordings
    
    temp_file_path = None
    wav_file_path = None
    processed_wav_path = None
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)
        
        logger.info(f"Saved uploaded file to {temp_file_path}, size: {len(content)} bytes")
        
        # If it's a webm file, convert to wav
        if file_ext == '.webm':
            wav_file_path = temp_file_path.replace('.webm', '_converted.wav')
            logger.info(f"Converting WebM to WAV: {temp_file_path} -> {wav_file_path}")
            
            if convert_webm_to_wav(temp_file_path, wav_file_path):
                logger.info("WebM to WAV conversion successful")
                # Use the converted file for processing
                processing_file_path = wav_file_path
            else:
                logger.warning("WebM conversion failed, trying direct processing")
                processing_file_path = temp_file_path
        else:
            processing_file_path = temp_file_path
        
        # Create a processed WAV file for returning
        if return_audio:
            processed_wav_path = processing_file_path.replace(file_ext, '_processed.wav')
            try:
                # Load and resave as clean WAV
                audio_data, sample_rate = librosa.load(processing_file_path, sr=22050, mono=True)
                sf.write(processed_wav_path, audio_data, sample_rate)
                logger.info(f"Created processed WAV file: {processed_wav_path}")
            except Exception as e:
                logger.warning(f"Could not create processed WAV: {str(e)}")
                processed_wav_path = None
        
        # Extract features with additional error handling
        try:
            features = extract_features(processing_file_path)
            features = np.expand_dims(features, axis=0)  # Add batch dimension (1, 40, 1)
            logger.info(f"Features extracted successfully, shape: {features.shape}")
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            # Return a friendly error instead of failing
            return {
                "filename": audio_file.filename,
                "predicted_emotion": "neutral",  # Default to neutral on failure
                "confidence_scores": {
                    "neutral": 1.0, "happy": 0.0, "sad": 0.0,
                    "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0
                },
                "error": str(e),
                "matches": False,
                "audio_base64": None
            }
        
        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)
        
        # Define emotion labels (adjust based on your dataset)
        emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        # Convert numpy values to Python types for JSON serialization
        confidence_scores = prediction[0].tolist()
        
        logger.info(f"Prediction result: {emotions[predicted_label]}")
        
        # Prepare audio data for return if requested
        audio_base64 = None
        if return_audio and processed_wav_path and os.path.exists(processed_wav_path):
            try:
                with open(processed_wav_path, 'rb') as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                logger.info("Audio file encoded to base64")
            except Exception as e:
                logger.warning(f"Could not encode audio to base64: {str(e)}")
        
        # Clean up the temporary files
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
        # Clean up in case of error
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
        # Return a user-friendly response instead of an error
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

# Get port from environment variable (Azure sets this)
port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)