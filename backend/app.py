import json
import logging
import traceback
from typing import Tuple
from flask import url_for, Flask, send_from_directory, request, jsonify, session
from dotenv import load_dotenv
import os
import werkzeug
from tools.cleanup import cleanUserDescription
from main import generate_answer
from flask_cors import CORS

load_dotenv()

#llm_model = "ollama"
#llm_model = "groq"
llm_model = "openai"

app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
app.secret_key = 'supersecretkey'
CORS(app)

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/process', methods=['POST'])
def process():
    question = request.form['name']
    answer, image_file_paths = generate_answer(question, llm_model)
    print(f"answer = {answer}, image_file_paths in app.py= {image_file_paths}")  # Log the paths
    return jsonify({'answer': answer, 'images': image_file_paths})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        save_path = os.path.join(app.root_path, 'static/data/')
        os.makedirs(save_path, exist_ok=True)
        file.save(os.path.join(save_path, filename))
        
        session['uploaded_filename'] = filename
        session.modified = True
        
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/save_description', methods=['POST'])
def save_description():
    description = request.json.get('description')
    prompt_directory = os.path.join(app.root_path, 'prompt')
    os.makedirs(prompt_directory, exist_ok=True)
    
    filename = session.get('uploaded_filename', 'unknown_file')

    if filename == 'metadata.csv':
        file_content = f"You can use this meta data file with filename = {filename} to extract information about the other files. Pass the information about the presence of metadata.csv to the python agent. \n{description}\n"
    else:
        file_content = f"You have access to this file with filename = {filename} for further processing\n{description}\n"

    file_path = os.path.join(prompt_directory, 'user_description_of_file.txt')

    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(file_content)
    else:
        with open(file_path, 'w') as file:
            file.write(file_content)
    
    return "Description saved successfully"

@app.route('/static/data/<path:filename>')
def serve_static_data(filename):
    return send_from_directory('static/data', filename)

if __name__ == "__main__":
    current_directory = os.path.abspath(os.path.dirname(__file__))
    root_directory = os.path.dirname(current_directory) + "/"
    cleanUserDescription(root_directory + "backend/")
    app.run(host="0.0.0.0", debug=True, port=8002)
