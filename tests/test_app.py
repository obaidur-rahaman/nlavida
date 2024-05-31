import os
import pytest
from flask import url_for
import sys
import io  # Added for file upload testing

# Ensure the src directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SERVER_NAME'] = 'localhost'  # Set SERVER_NAME for URL building
    with app.app_context():
        with app.test_client() as client:
            yield client

@pytest.fixture(autouse=True)
def ensure_directories_exist():
    os.makedirs('../static/data/', exist_ok=True)
    yield
    # Clean up any files created during tests if necessary

def test_index(client):
    """Test the index route."""
    with app.app_context():
        response = client.get(url_for('index'))
        assert response.status_code == 200
        assert b"NLAVIDA" in response.data  # Replace with actual content expected

def test_process(client, mocker):
    """Test the process route."""
    with app.app_context():
        mock_generate_answer = mocker.patch('app.generate_answer')
        mock_generate_answer.return_value = ("Test Answer", ["image1.png", "image2.png"])

        response = client.post(url_for('process'), data={'name': 'Test Question'})
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['answer'] == "Test Answer"
        assert json_data['images'] == ["image1.png", "image2.png"]

def test_upload_file(client):
    """Test the file upload route."""
    with app.app_context():
        data = {
            'file': (io.BytesIO(b"fake file content"), 'test.txt')
        }
        response = client.post(url_for('upload_file'), content_type='multipart/form-data', data=data)
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['message'] == 'File uploaded successfully'
        assert json_data['filename'] == 'test.txt'

def test_save_description(client):
    """Test the save_description route."""
    with app.app_context():
        with client.session_transaction() as sess:
            sess['uploaded_filename'] = 'test.txt'

        response = client.post(url_for('save_description'), data={'description': 'Test Description'})
        assert response.status_code == 200
        assert response.data == b"Description saved successfully"

        # Check the file content
        with open('../prompt/user_description_of_file.txt', 'r') as f:
            content = f.read()
            assert 'Test Description' in content
            assert 'test.txt' in content
