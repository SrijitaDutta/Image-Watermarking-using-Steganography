import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
import io

app = Flask(__name__)

# Embed Watermark using LSB in Blue Channel with Balanced Quality
def embed_watermark_lsb(image, watermark):
    wm_h, wm_w = image.shape[:2]
    watermark = cv2.resize(watermark, (wm_w // 5, wm_h // 5))  # Downscale watermark
    
    # Convert watermark to grayscale and resize back to match image size
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    watermark = cv2.resize(watermark, (wm_w, wm_h))
    
    # Reduce watermark to 3-bit depth for better retrieval
    watermark_3bit = watermark >> 5  # 3-bit instead of 2-bit for better clarity
    watermark_3bit = watermark_3bit.astype(np.uint8)
    
    # Embed in the blue channel only
    watermarked = image.copy()
    watermarked[:, :, 0] = (watermarked[:, :, 0] & 0b11111000) | watermark_3bit  # Use 3-bit embedding
    
    return watermarked, watermark_3bit

# Extract Watermark from Blue Channel
def extract_watermark_lsb(watermarked):
    extracted_wm = (watermarked[:, :, 0] & 0b00000111) * 32  # Normalize extracted watermark
    extracted_wm = cv2.equalizeHist(extracted_wm)  # Enhance contrast
    return extracted_wm.astype(np.uint8)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_watermark():
    file = request.files['image']
    watermark_file = request.files['watermark']

    # Read input images
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    watermark = cv2.imdecode(np.frombuffer(watermark_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Apply watermark
    watermarked_img, processed_wm = embed_watermark_lsb(img, watermark)

    # Save processed watermark separately for verification
    _, wm_encoded = cv2.imencode('.png', processed_wm, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    with open("processed_watermark.png", "wb") as f:
        f.write(wm_encoded.tobytes())

    # Save watermarked image as PNG with compression to balance quality & size
    _, img_encoded = cv2.imencode('.png', watermarked_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')

@app.route('/extract', methods=['POST'])
def extract():
    watermarked_file = request.files['watermarked']
    watermarked = cv2.imdecode(np.frombuffer(watermarked_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Extract watermark
    extracted_wm = extract_watermark_lsb(watermarked)

    # Save extracted watermark separately for comparison
    _, wm_encoded = cv2.imencode('.png', extracted_wm, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return send_file(io.BytesIO(wm_encoded.tobytes()), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
