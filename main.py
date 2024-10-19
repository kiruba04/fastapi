from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import numpy as np
import io
import wave
import noisereduce as nr
import nltk
from nltk.corpus import stopwords
from fastapi.staticfiles import StaticFiles

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

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI Tools</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            h1 { text-align: center; }
            .container { max-width: 600px; margin: 0 auto; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; }
            input, textarea, button { width: 100%; padding: 10px; font-size: 1rem; }
            button { background-color: #007BFF; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .output { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FastAPI Tools</h1>

            <!-- Audio Noise Removal Form -->
            <div class="form-group">
                <label for="audio-file">Upload Audio for Noise Removal:</label>
                <input type="file" id="audio-file">
                <button onclick="removeNoise()">Remove Noise</button>
            </div>
            <div class="output" id="audio-output"></div>

            <!-- Stop Words Removal Form -->
            <div class="form-group">
                <label for="text-input">Enter Text to Remove Stop Words:</label>
                <textarea id="text-input" rows="4"></textarea>
                <button onclick="removeStopWords()">Remove Stop Words</button>
            </div>
            <div class="output" id="text-output"></div>

            <!-- Background Removal Form -->
            <div class="form-group">
                <label for="image-file">Upload Image for Background Removal:</label>
                <input type="file" id="image-file">
                <button onclick="removeBackground()">Remove Background</button>
            </div>
            <div class="output" id="image-output"></div>
        </div>

        <script>
            async function removeNoise() {
                const fileInput = document.getElementById('audio-file');
                const output = document.getElementById('audio-output');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const response = await fetch('/remove-audio-noise/', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    output.innerHTML = `<audio controls src="${url}"></audio>`;
                } else {
                    output.textContent = 'Error removing noise';
                }
            }

            async function removeStopWords() {
                const textInput = document.getElementById('text-input').value;
                const output = document.getElementById('text-output');

                const response = await fetch('/remove-stop-words/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textInput })
                });

                const result = await response.json();
                output.textContent = 'Filtered Text: ' + result.filtered_text;
            }

            async function removeBackground() {
                const fileInput = document.getElementById('image-file');
                const output = document.getElementById('image-output');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const response = await fetch('/remove-background/', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    output.innerHTML = `<img src="${url}" alt="Image with background removed" style="max-width: 100%;">`;
                } else {
                    output.textContent = 'Error removing background';
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

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
