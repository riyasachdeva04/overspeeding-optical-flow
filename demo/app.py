from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    logging.info("Upload endpoint hit")
    
    try:
        # Debug: Print all incoming request details
        logging.info(f"Request files: {request.files}")
        logging.info(f"Request form: {request.form}")

        if 'videoFile' not in request.files:
            logging.error("No file part in the request")
            return jsonify({
                'success': False, 
                'message': 'No file uploaded', 
                'files_received': list(request.files.keys())
            }), 400
        
        file = request.files['videoFile']
        
        if file.filename == '':
            logging.error("No selected file")
            return jsonify({
                'success': False, 
                'message': 'No selected file'
            }), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logging.info(f"Saving file to {filepath}")
            file.save(filepath)
            with open('v.txt', 'r') as f:
                values = [float(line.strip()) for line in f]
                max_value = max(values)
                logging.info(f"Maximum value from v.txt: {max_value}")
            # get_velocity()
            return jsonify({
                'success': True, 
                'message': f'File uploaded successfully! Maximum Speed: {max_value*20*0.01} m/s',
                'filename': filename
            }), 200
        
        return jsonify({
            'success': False, 
            'message': 'Invalid file type'
        }), 400
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'message': f'Unexpected error: {str(e)}'
        }), 500

    
if __name__ == '__main__':
    app.run(debug=True, port=5500)