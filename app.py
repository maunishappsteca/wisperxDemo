import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
from typing import Optional
from botocore.exceptions import ClientError
import logging
from datetime import datetime

# --- Configuration ---
COMPUTE_TYPE = "float16"  # Changed to float16 for better cuda compatibility
BATCH_SIZE = 16  # Reduced batch size for cuda
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")
S3_OUTPUT_DIR = "output"  # Base directory for outputs

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

def ensure_model_cache_dir():
    """Ensure model cache directory exists and is accessible"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV"""
    try:
        output_path = f"/tmp/{uuid.uuid4()}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed error: {str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    try:
        if not ensure_model_cache_dir():
            logger.error(f"Model cache directory is not accessible")
            raise RuntimeError("Model cache directory is not accessible")
            
        return whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def load_alignment_model(language_code: str):
    """Load alignment model with fallback options"""
    try:
        # Try to load the default model first
        return whisperx.load_align_model(language_code=language_code, device="cuda")
    except Exception as e:
        logger.warning(f"Failed to load default alignment model for {language_code}, trying fallback: {str(e)}")
        
        # Define fallback models for specific languages
        fallback_models = {
            "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",  # Hindi
            "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", # Portuguese
            "he": "imvladikon/wav2vec2-xls-r-300m-hebrew", # Hebrew
        }
        
        if language_code in fallback_models:
            try:
                # Try to load the fallback model
                return whisperx.load_align_model(
                    model_name=fallback_models[language_code],
                    device="cuda"
                )
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback alignment model for {language_code}: {str(fallback_e)}")
                raise RuntimeError(f"Alignment model loading failed for {language_code}")
        else:
            logger.error(f"No alignment model available for language: {language_code}")
            raise RuntimeError(f"No alignment model available for language: {language_code}")
        

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription logic with optional translation"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language else "en")
        
        if align and detected_language != "unknown":
            try:
                align_model, metadata = load_alignment_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device="cuda",
                    return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment skipped: {str(e)}")
                # Continue without alignment if it fails
                result["alignment_error"] = str(e)

        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "segments": result["segments"],
            "language": detected_language,
            "model": model_size,
            "alignment_success": "alignment_error" not in result
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")
    


def save_response_to_s3(job_id, response_data, status="success"):
    """
    Save response to S3 bucket in the appropriate directory structure
    
    Args:
        job_id: The ID of the job
        response_data: The response data to save
        status: Status of the job (success, error, failed)
    """
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not configured, skipping response save")
        return False
    
    try:
        # Create the directory path
        directory_path = f"{S3_OUTPUT_DIR}/{job_id}/"
        file_key = f"{directory_path}response.json"
        
        # Convert response to JSON string
        response_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        
        # Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=response_json,
            ContentType='application/json'
        )
        
        logger.info(f"Response saved to S3: s3://{S3_BUCKET}/{file_key}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save response to S3: {str(e)}")
        return False
    

def handler(job):
    """RunPod serverless handler"""
    try:
        # Validate input

        if not job.get("id"):
            return {"error": "job id not found"}
        
        job_id = job["id"]
        # Initialize response variable
        response = {}

        if not job.get("input"):
            response = {"error": "No input provided", "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            return response

        file_name = input_data.get("file_name")

        jobid = job["id"]
        logger.error(f"new job id is : {str(jobid)}")
        
        if not file_name:
            response = {"error": "No file_name provided in input", "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            return response
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}
        
        # 2. Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                audio_path = local_path
        except Exception as e:
            response = {"error": f"Audio processing failed: {str(e)}", "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            return response
            
        # 3. Transcribe
        try:
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language", None),
                input_data.get("align", False)
            )
            result["job_id"] = job_id  # Include job ID in the result
            result["status"] = "success"
            logger.info(f"Transcription completed for job ID: {job_id}")
            
            # Save successful response
            save_response_to_s3(job_id, result, "success")
            response = result

        except Exception as e:
            response = {"error": str(e), "job_id": job_id, "status": "failed"}
            save_response_to_s3(job_id, response, "failed")
            
        finally:
            # 4. Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()
        
        return response
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    print("Starting WhisperX cuda Endpoint with Translation...")
    
    # Verify model cache directory at startup
    if not ensure_model_cache_dir():
        print("ERROR: Model cache directory is not accessible")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            # In serverless mode, we want to fail fast if model dir isn't available
            raise RuntimeError("Model cache directory is not accessible")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input
        test_result = handler({
            "id": "test-job-id-123",
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "language": "hi",
                "align": True
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
