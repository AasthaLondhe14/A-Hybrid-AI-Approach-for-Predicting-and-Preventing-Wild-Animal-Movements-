import React, { useEffect, useState, useRef } from "react";
import "./App.css";
import axios from "axios";
import Contacts from "./Contacts";
import Help from "./Help";
import AudioPredict from "./AudioPredict";

function App() {
  const [detected, setDetected] = useState([]);
  const [counts, setCounts] = useState({});
  const [videoScores, setVideoScores] = useState({});
  const [audioDetected, setAudioDetected] = useState([]);
  const [audioScores, setAudioScores] = useState({});
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [activeView, setActiveView] = useState("home");
  const audioRef = useRef(null);
  const previousDetectedRef = useRef([]);
  const API_BASE = `${window.location.protocol}//${window.location.hostname}:5000`;
  const ipCameraStreamUrl = "http://100.107.105.168:8080/video";
  const dangerousAnimals = [
    "tiger",
    "leopard",
    "lion",
    "bear",
    "elephant",
    "wild boar",
    "boar",
    "wolf",
    "panther",
    "crocodile",
    "rhino",
    "hippo",
    "snake",
  ];
  const isDangerous = detected.some((animal) =>
    dangerousAnimals.includes(String(animal).toLowerCase())
  );
  const onlyHumanDetected =
    detected.length > 0 &&
    detected.every((animal) => String(animal).toLowerCase() === "human");
  const nonHumanDetected = detected.filter(
    (animal) => String(animal).toLowerCase() !== "human"
  );
  const videoDisplayAnimals = nonHumanDetected;
  const animalDetected = videoDisplayAnimals.length > 0;
  const cleanLabel = (label) => String(label).toLowerCase();
  const isHuman = (label) => cleanLabel(label) === "human";
  const topAudioLabels = Object.entries(audioScores)
    .filter(([label, score]) => !isHuman(label) && (score ?? 0) > 0)
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
    .slice(0, 3)
    .map(([label, score]) => `${label} (${Math.round((score ?? 0) * 100)}%)`);

  const finalIsDangerous = false;
  const audioDangerousHigh = Object.entries(audioScores).some(
    ([label, score]) =>
      dangerousAnimals.includes(String(label).toLowerCase()) && score >= 0.8
  );
  const shouldAlert = isDangerous || finalIsDangerous || audioDangerousHigh;
  const lastAlertRef = useRef(false);

  useEffect(() => {
    const fetchDetection = async () => {
      try {
        const res = await axios.get(`${API_BASE}/detect`);
        const isSuccess = res.data.status === "success";
        const newDetected = isSuccess ? (res.data.detected || []) : [];
        const newCounts = isSuccess ? (res.data.counts || {}) : {};
        const newScores = isSuccess ? (res.data.scores || {}) : {};

        const prevDetected = previousDetectedRef.current;
        const newAnimals = newDetected.filter(
          (animal) => !prevDetected.includes(animal)
        );

        setDetected(newDetected);
        setCounts(newCounts);
        setVideoScores(newScores);
        previousDetectedRef.current = newDetected;

        if (newAnimals.length > 0) {
          // keep newAnimals for future use if needed
        }
      } catch (err) {
        console.error("❌ Detection error:", err);
      }
    };

    fetchDetection();
    const interval = setInterval(fetchDetection, 10000);

    return () => clearInterval(interval);
  }, [soundEnabled, API_BASE]); // re-run when soundEnabled changes

  useEffect(() => {
    const fetchAudioDetection = async () => {
      try {
        const res = await axios.get(`${API_BASE}/audio_detect`);
        const newAudioDetected = res.data.detected || [];
        const newAudioScores = res.data.scores || {};
        setAudioDetected(newAudioDetected);
        setAudioScores(newAudioScores);
      } catch (err) {
        console.error("❌ Audio detection error:", err);
      }
    };

    fetchAudioDetection();
    const interval = setInterval(fetchAudioDetection, 10000);

    return () => clearInterval(interval);
  }, [API_BASE]);

  useEffect(() => {
    if (!soundEnabled || !audioRef.current) return;
    if (shouldAlert && !lastAlertRef.current) {
      audioRef.current.play().catch(() => {
        console.warn("🔇 User interaction required to play audio.");
      });
    }
    lastAlertRef.current = shouldAlert;
  }, [shouldAlert, soundEnabled]);

  const handleEnableSound = () => {
    if (audioRef.current) {
      audioRef.current.play().then(() => {
        setSoundEnabled(true);
      }).catch(() => {
        console.warn("🔇 User interaction required to play audio.");
      });
    }
  };

  const handleMenuClick = (view) => {
    setActiveView(view);
  };

  return (
    <div>
      {/* Alert sound (path fixed to public folder) */}
      <audio ref={audioRef} src="/sound.wav" preload="auto" />

      <header className="app-header">
        Wild Animal Intrusion Detection and Prevention System
      </header>

      <div className="app">
        <div className="sidebar glass slide-in-left">
          <ul>
            <li onClick={() => handleMenuClick("home")} style={{ cursor: "pointer", fontWeight: activeView === "home" ? "bold" : "normal" }}>Home</li>
            <li onClick={() => handleMenuClick("prediction")} style={{ cursor: "pointer", fontWeight: activeView === "prediction" ? "bold" : "normal" }}>Prediction Results</li>
            <li onClick={() => handleMenuClick("help")} style={{ cursor: "pointer", fontWeight: activeView === "help" ? "bold" : "normal" }}>Help</li>
            <li onClick={() => handleMenuClick("contacts")} style={{ cursor: "pointer", fontWeight: activeView === "contacts" ? "bold" : "normal" }}>Contacts</li>
          </ul>
        </div>

        <div className="main glass fade-in">
          {activeView === "contacts" ? (
            <Contacts />
          ) : activeView === "prediction" ? (
            <AudioPredict />
          ) : activeView === "help" ? (
            <Help />
          ) : (
            <>
          <div className="section">
            <h3>Live Intrusion Alerts</h3>

            {/* Enable sound button */}
            {!soundEnabled && (
              <button onClick={handleEnableSound} style={{ marginBottom: "10px", padding: "6px 12px" }}>
                🔊 Enable Alert Sound
              </button>
            )}

            {animalDetected ? (
              <>
                <div className="alert-box danger">⚠ Animal Detected</div>
                <div className="detected-name" style={{ fontWeight: "bold" }}>
                  {videoDisplayAnimals.map((animal, index) => (
                    <div key={index}>🔸 {animal}</div>
                  ))}
                </div>
              </>
            ) : (
              <div className="alert-box">
                {onlyHumanDetected ? "Monitoring..." : "🔄 Detecting..."}
              </div>
            )}
          </div>

          <div className="section live-risk-container">
            <div className="live-feed">
              <h3>Live Camera Feed</h3>
              <div className="alert-box">Video Stream</div>
              <div style={{ position: "relative" }}>
                <div className="live-badge">LIVE</div>
                <img
                  alt="IP Camera Feed"
                  className="video-box"
                  src={ipCameraStreamUrl}
                  style={{
                    width: "100%",
                    height: "360px",
                    borderRadius: "15px",
                    border: "none",
                    background: "#000",
                    objectFit: "contain",
                  }}
                />
              </div>
            </div>

            <div className="risk-prediction">
              <h3>Village & Crop Safety</h3>
              {animalDetected === false ? (
                <div className="alert-box">Monitoring...</div>
              ) : (
                <>
                  <div className="alert-box danger">Animal Detected</div>
                  <div className="detected-name" style={{ fontWeight: "bold", marginBottom: "8px" }}>
                    {videoDisplayAnimals.map((animal, index) => (
                      <div key={index}>▶ {animal} ({Math.round((videoScores[animal] ?? 0) * 100)}%)</div>
                    ))}
                  </div>
                  {Object.keys(counts).length > 0 && (
                    <div>
                      <div style={{ fontWeight: "bold", marginBottom: "6px" }}>Detection Chart</div>
                      {Object.entries(counts)
                        .filter(([label]) => !isHuman(label))
                        .map(([label, count]) => {
                        const width = Math.min(100, count * 12);
                        return (
                          <div key={label} style={{ display: "flex", alignItems: "center", marginBottom: "6px" }}>
                            <div style={{ width: "90px", fontSize: "12px" }}>{label}</div>
                            <div style={{ flex: 1, background: "#f1f1f1", borderRadius: "6px", overflow: "hidden" }}>
                              <div
                                style={{
                                  width: `${width}%`,
                                  background: "#2f60ff",
                                  color: "#fff",
                                  padding: "4px 6px",
                                  fontSize: "12px",
                                }}
                              >
                                {count}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {isDangerous ? (
                    <div className="alert-box danger" style={{ marginTop: "10px" }}>
                      Dangerous Animal Detected
                    </div>
                  ) : (
                    <div className="alert-box" style={{ marginTop: "10px" }}>
                      Not Dangerous
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          <div className="section">
            <h3>Audio Detection (IP Camera)</h3>
            {audioDetected.length === 0 ? (
              <>
                <div className="alert-box">Listening...</div>
                {topAudioLabels.length > 0 && (
                  <div className="alert-box">
                    Top 3: [{topAudioLabels.join(", ")}]
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="alert-box danger">Animal Detected (Audio)</div>
                {topAudioLabels.length > 0 && (
                  <div className="alert-box">
                    Top 3: [{topAudioLabels.join(", ")}]
                  </div>
                )}
                <div className="detected-name" style={{ fontWeight: "bold", marginBottom: "8px" }}>
                  {audioDetected.map((animal, index) => (
                    <div key={index}>▶ {animal} ({Math.round((audioScores[animal] ?? 0) * 100)}%)</div>
                  ))}
                </div>
              </>
            )}

            {Object.keys(audioScores).length > 0 && (
              <div style={{ marginTop: "10px" }}>
                <div style={{ fontWeight: "bold", marginBottom: "6px" }}>Audio Detection Chart</div>
                {Object.entries(audioScores)
                  .filter(([label, score]) => (score ?? 0) > 0)
                  .map(([label, score]) => {
                  const width = Math.min(100, Math.round(score * 100));
                  return (
                    <div key={label} style={{ display: "flex", alignItems: "center", marginBottom: "6px" }}>
                      <div style={{ width: "90px", fontSize: "12px" }}>{label}</div>
                      <div style={{ flex: 1, background: "#f1f1f1", borderRadius: "6px", overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${width}%`,
                            background: "#2f60ff",
                            color: "#fff",
                            padding: "4px 6px",
                            fontSize: "12px",
                          }}
                        >
                          {Math.round(score * 100)}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
