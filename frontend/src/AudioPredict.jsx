import React, { useEffect, useState } from "react";
import axios from "axios";

function AudioPredict() {
  const [history, setHistory] = useState([]);

  const API_BASE = `${window.location.protocol}//${window.location.hostname}:5000`;

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
        <h3>Prediction Results (Database)</h3>
        {history.length === 0 ? (
          <p>No prediction history available.</p>
        ) : (
          <ul className="info-list">
            {history.map((record) => (
              <li key={record._id}>
                {record.animal_name} - {record.detection_type} - {Math.round(record.confidence * 100)}%
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default AudioPredict;
