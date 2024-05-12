from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from main import generate_answer
import os
import werkzeug
from tools.cleanup import cleanUserDescription
load_dotenv()

#llm_model = "ollama"
llm_model = "openai"

app = Flask(__name__, static_folder='../static')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    question = request.form['name']
    answer, image_file_path = generate_answer(question, llm_model)  # Assuming this returns a tuple
    print(f"answer = {answer}, image_file_path = {image_file_path}")
    return jsonify({'answer': answer, 'image': image_file_path})  # Include the image string and format in the response

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return 'No selected file', 400
    if file and file.filename.endswith('.csv'):
        filename = werkzeug.utils.secure_filename(file.filename)
        file.save(os.path.join('../static/data/', filename))  # Adjust the directory path as necessary
        return 'File uploaded successfully', 200
    else:
        return 'Invalid file type', 400

@app.route('/save_description', methods=['POST'])
def save_description():
    description = request.form['description']
    prompt_directory = '../prompt'
    os.makedirs(prompt_directory, exist_ok=True)  # Ensure the directory exists
    with open(os.path.join(prompt_directory, 'user_description_of_file.txt'), 'w') as file:
        file.write(description)
    return "Description saved successfully"

if __name__ == "__main__":
    # Get the absolute path of the current directory
    current_directory = os.path.abspath(os.path.dirname(__file__))
    # Go one directory up
    root_directory = os.path.dirname(current_directory) + "/"
    cleanUserDescription(root_directory)
    app.run(host="0.0.0.0", debug=True, port=8002)
