from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from main import generate_answer
import os
import werkzeug
from tools.cleanup import cleanUserDescription
load_dotenv()

#llm_model = "ollama"
llm_model = "openai"

app = Flask(__name__, static_folder='../static')
app.secret_key = 'supersecretkey'  # Needed for session management

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    question = request.form['name']
    answer, image_file_paths = generate_answer(question, llm_model)  # Assuming this returns a tuple
    print(f"answer = {answer}, image_file_paths = {image_file_paths}")
    return jsonify({'answer': answer, 'images': image_file_paths})  # Include the image paths list in the response

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
        file.save(os.path.join('../static/data/', filename))  # Adjust the directory path as necessary
        
        # Save the filename in the session
        session['uploaded_filename'] = filename
        session.modified = True  # Ensure session changes are saved
        
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/save_description', methods=['POST'])
def save_description():
    description = request.form['description']
    prompt_directory = '../prompt'
    os.makedirs(prompt_directory, exist_ok=True)  # Ensure the directory exists
    
    # Retrieve the filename from the session
    filename = session.get('uploaded_filename', 'unknown_file')
    file_content = f"The name of the file is {filename}\n{description}\n"
    
    # Define the path to the file
    file_path = os.path.join(prompt_directory, 'user_description_of_file.txt')

    # Check if the file exists
    if os.path.exists(file_path):
        # Append the content if the file exists
        with open(file_path, 'a') as file:
            file.write(file_content)
    else:
        # Write the content if the file does not exist
        with open(file_path, 'w') as file:
            file.write(file_content)
    
    return "Description saved successfully"

if __name__ == "__main__":
    # Get the absolute path of the current directory
    current_directory = os.path.abspath(os.path.dirname(__file__))
    # Go one directory up
    root_directory = os.path.dirname(current_directory) + "/"
    cleanUserDescription(root_directory)
    app.run(host="0.0.0.0", debug=True, port=8002)
