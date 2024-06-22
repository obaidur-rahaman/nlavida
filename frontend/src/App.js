import React, { useState } from 'react';
import './App.css';

function App() {
  const [name, setName] = useState('');
  const [response, setResponse] = useState('');
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [description, setDescription] = useState('');
  const [showDescriptionInput, setShowDescriptionInput] = useState(false);
  const [descriptionSubmitted, setDescriptionSubmitted] = useState(false);

  const handleFileChange = async (e) => {
    setShowDescriptionInput(false);
    setUploadStatus('');
    setDescriptionSubmitted(false);

    const formData = new FormData();
    formData.append('file', e.target.files[0]);

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.error) {
        setUploadStatus(data.error);
      } else {
        setUploadStatus('File uploaded successfully');
        setShowDescriptionInput(true);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('Error uploading file');
    }
  };

  const handleNameChange = (e) => {
    setName(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); // Show the spinner
    setUploadStatus('');
    setResponse('');
    setImages([]);
    setShowDescriptionInput(false);

    const formData = new FormData();
    formData.append('name', name);

    try {
      const res = await fetch('/process', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      console.log('Image paths:', data.images); // Log image paths for debugging
      setResponse(data.answer);
      setImages(data.images || []);
      setLoading(false); // Hide the spinner
    } catch (error) {
      console.error('Failed to fetch:', error);
      setResponse('Error fetching answer.');
      setLoading(false); // Hide the spinner
    }
  };

  const handleDescriptionSubmit = async () => {
    try {
      const res = await fetch('/save_description', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ description }),
      });
      if (res.ok) {
        setUploadStatus('Description saved successfully');
        setDescriptionSubmitted(true); // Hide the description input after submission
      } else {
        setUploadStatus('Error saving description');
      }
    } catch (error) {
      console.error('Error saving description:', error);
      setUploadStatus('Error saving description');
    }
  };

  return (
    <div className="App">
      <main>
        <div id="answer" style={{ display: response ? '' : 'none' }}>
          <h2>Answer:</h2>
          <p>{response}</p>
          <div id="imageContainer">
            {images.map((src, index) => (
              <img key={index} src={src} alt={`Result ${index}`} />
            ))}
          </div>
        </div>
      </main>
      <footer>
        <form id="name-form" onSubmit={handleSubmit} style={{ width: '70%', margin: '0 auto' }}>
          <button className="file-upload-btn" type="button" onClick={() => document.getElementById('file').click()}>
            <span>&#128206;</span>
            <input
              className="file-upload-input"
              type="file"
              name="file"
              id="file"
              accept=".csv"
              onChange={handleFileChange}
            />
          </button>
          <input
            type="text"
            name="name"
            placeholder="Enter your question here"
            required
            value={name}
            onChange={handleNameChange}
            style={{ flexGrow: 1, fontSize: '1.2em' }}
          />
          <button type="submit">Ask NLAVIDA</button>
        </form>
        <div id="uploadStatus">{uploadStatus}</div>
        <div id="spinner" className={loading ? 'visible' : ''}>
          <div className="spinner"></div>
        </div>
        {showDescriptionInput && !descriptionSubmitted && (
          <div id="descriptionInput">
            <input
              type="text"
              placeholder="Describe your file..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              style={{ fontSize: '1.2em', width: '70%' }}
            />
            <button type="button" onClick={handleDescriptionSubmit} style={{ padding: '10px 20px', background: '#ff8800', color: 'white', border: 'none', cursor: 'pointer' }}>Submit</button>
          </div>
        )}
      </footer>
    </div>
  );
}

export default App;
