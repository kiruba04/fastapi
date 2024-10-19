from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import numpy as np
import io
import wave
import noisereduce as nr
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Initialize the FastAPI app
app = FastAPI()

# Request model for text input
class TextInput(BaseModel):
    text: str

# Helper function to read WAV audio
def read_wav(file_bytes):
    with wave.open(io.BytesIO(file_bytes), 'rb') as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
        audio_data = np.frombuffer(frames, dtype=np.int16)  # Assuming 16-bit PCM
        return audio_data, params

# Helper function to write WAV audio
def write_wav(audio_data, params):
    output_io = io.BytesIO()
    with wave.open(output_io, 'wb') as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(audio_data.tobytes())
    output_io.seek(0)
    return output_io


@app.get('/')
async def root():
    return {"message": "Noise reduction endpoint"}

# Noise reduction endpoint    
@app.post("/remove-audio-noise/")
async def remove_audio_noise(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    # Read the audio data
    audio_data, params = read_wav(audio_bytes)

    # Reduce noise
    reduced_noise = nr.reduce_noise(y=audio_data, sr=params.framerate)

    # Convert to integer type (same as original)
    reduced_noise = reduced_noise.astype(np.int16)

    # Write back to WAV format
    output_io = write_wav(reduced_noise, params)

    return StreamingResponse(output_io, media_type="audio/wav")

# Stop words removal endpoint
@app.post("/remove-stop-words/")
async def remove_stop_words(input_data: TextInput):
    # Get the text to process
    text = input_data.text

    # Split the text into words
    words = text.split()

    # Get English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the text
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)

    # Return the text without stop words
    return {"filtered_text": filtered_text}

# Background removal endpoint
@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    # Read the uploaded image
    image_bytes = await file.read()
    input_image = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB (if not already)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    # Remove background
    output_image = remove(image_bytes)

    # Convert the result back to an image
    result_image = Image.open(io.BytesIO(output_image))

    # Save the output image to a byte stream
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
