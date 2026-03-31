import React, { useEffect, useState } from "react";
import axios from "axios";

function AudioPredict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [history, setHistory] = useState([]);

  const API_BASE = `${window.location.protocol}//${window.location.hostname}:5000`;

  const handlePredict = async () => {
    if (!file) {
      alert("Upload an audio file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/predict-audio`, formData);
      setResult(
        `🐾 Animal: ${res.data.animal} (Confidence: ${res.data.confidence}%)`
      );
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }
  };

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE}/history?limit=10`);
        if (res.data.status === "success") {
          setHistory(res.data.history || []);
        }
      } catch (err) {
        console.error("History fetch error:", err);
      }
    };
    fetchHistory();
  }, [API_BASE]);

  return (
    <div className="page-container">
      <h2 className="page-title">Animal Sound Prediction</h2>

      <div className="card-box">
        <input
          type="file"
          accept="audio/*"
          className="input-field"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <br /><br />

        <button className="save-btn" onClick={handlePredict}>Predict</button>

        <h3>{result}</h3>
      </div>

      <div className="card-box">
        <h3>Prediction Results (Database)</h3>
        {history.length === 0 ? (
          <p>No prediction history available.</p>
        ) : (
          <ul className="info-list">
            {history.map((record) => (
              <li key={record._id}>
                {record.animal_name} — {record.detection_type} — {Math.round(record.confidence * 100)}%
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default AudioPredict;
