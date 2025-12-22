import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:3000';

function App() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, completed, failed
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      // Reset state for re-uploads
      setJobId(null);
      setStatus('idle');
      setDownloadUrl(null);
      setError(null);
    }
  };

  const uploadFile = async () => {
    if (!file) return;

    setStatus('uploading');
    setError(null);

    const formData = new FormData();
    formData.append('processing_job[original_image]', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/processing_jobs`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setJobId(response.data.id);
      setStatus('processing');
    } catch (err) {
      console.error('Upload failed:', err);
      setError('File upload failed. Please try again.');
      setStatus('failed');
    }
  };
  
  // Effect for triggering upload when file is selected
  useEffect(() => {
    if (file) {
      uploadFile();
    }
  }, [file]);

  // Effect for polling job status
  useEffect(() => {
    if (status !== 'processing' || !jobId) {
      return;
    }

    const intervalId = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/processing_jobs/${jobId}`);
        const { status: jobStatus, prepped_psd_url } = response.data;

        if (jobStatus === 'completed') {
          clearInterval(intervalId);
          setStatus('completed');
          setDownloadUrl(prepped_psd_url);
        } else if (jobStatus === 'failed') {
          clearInterval(intervalId);
          setStatus('failed');
          setError('Image processing failed.');
        }
        // If 'processing', do nothing and let the interval continue
      } catch (err) {
        clearInterval(intervalId);
        setStatus('failed');
        setError('Could not get job status.');
        console.error('Polling failed:', err);
      }
    }, 2000);

    // Cleanup function to clear interval
    return () => clearInterval(intervalId);
  }, [status, jobId]);

  const renderContent = () => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return (
          <div className="progress-indicator">
            <p>Processing...</p>
            <div className="spinner"></div>
          </div>
        );
      case 'completed': {
        if (!downloadUrl) {
          return <p className="error-message">Could not get download URL.</p>;
        }
        const finalUrl = new URL(downloadUrl, API_BASE_URL).href;
        return (
          <a href={finalUrl} className="download-button" download>
            Download PSD
          </a>
        );
      }
      case 'failed':
        return <p className="error-message">{error}</p>;
      case 'idle':
      default:
        return (
          <div className="file-input-area">
            <p>Select a file to begin</p>
             <input type="file" onChange={handleFileChange} accept="image/*" />
          </div>
        );
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <header>
          <h1>PSD Auto Prepper</h1>
        </header>
        <main>
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

export default App;