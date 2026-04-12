import React, { useEffect, useState } from "react";
import axios from "axios";
import "./AudioPredict.css";

function AudioPredict() {
  const [history, setHistory] = useState([]);

  const API_BASE = `${window.location.protocol}//${window.location.hostname}:5000`;

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE}/history?limit=20`);
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
    <div className="page-container audio-predict">
      <h2 className="page-title">Prediction Results History</h2>

      <div className="card-box prediction-card">
        {history.length === 0 ? (
          <p>No prediction history available.</p>
        ) : (
          <ul className="info-list prediction-list">
            {history.map((record) => (
              (() => {
                const animalText = String(record.animal_name || "").toLowerCase();
                const detectionText = String(record.detection_type || "").toLowerCase();
                const dangerousAnimals = [
                  "tiger",
                  "lion",
                  "leopard",
                  "cheetah",
                  "panther",
                  "wolf",
                  "bear",
                  "elephant",
                  "crocodile",
                  "alligator",
                  "hippo",
                  "rhino",
                  "rhinoceros",
                  "boar",
                  "wild boar",
                  "snake",
                  "cobra",
                  "python",
                  "hyena",
                  "buffalo",
                  "bison",
                ];

                const isExplicitSafe =
                  detectionText.includes("safe") ||
                  detectionText.includes("non-danger") ||
                  detectionText.includes("nondanger") ||
                  detectionText.includes("not danger");
                const isExplicitDanger =
                  detectionText.includes("danger") ||
                  detectionText.includes("threat") ||
                  detectionText.includes("high risk");
                const isAnimalDanger = dangerousAnimals.some((name) =>
                  animalText.includes(name)
                );
                const isDangerous =
                  isExplicitDanger || (!isExplicitSafe && isAnimalDanger);

                return (
                  <li
                    key={record._id}
                    className={`prediction-item ${
                      isDangerous ? "is-danger" : "is-safe"
                    }`}
                    data-type={record.detection_type}
                  >
                <span className="prediction-name">{record.animal_name}</span>
                <span className="prediction-type">{record.detection_type}</span>
                <span className="prediction-confidence">
                  {Math.round(record.confidence * 100)}%
                </span>
                  </li>
                );
              })()
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default AudioPredict;
