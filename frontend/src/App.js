import React, { useEffect, useState, useRef } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [detected, setDetected] = useState([]);
  const [counts, setCounts] = useState({});
  const [soundEnabled, setSoundEnabled] = useState(false);
  const audioRef = useRef(null);
  const previousDetectedRef = useRef([]);
  const ipCameraStreamUrl = "http://192.0.0.4:8080/video";
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

  useEffect(() => {
    const fetchDetection = async () => {
      try {
        const res = await axios.get("http://localhost:5000/detect");
        const newDetected = res.data.detected || [];
        const newCounts = res.data.counts || {};

        const prevDetected = previousDetectedRef.current;
        const newAnimals = newDetected.filter(
          (animal) => !prevDetected.includes(animal)
        );

        if (newAnimals.length > 0) {
          setDetected((prev) => [...prev, ...newAnimals]);
          previousDetectedRef.current = [...prevDetected, ...newAnimals];
          setCounts((prev) => ({ ...prev, ...newCounts }));

          // Play alert sound if user has enabled sound
          if (soundEnabled && audioRef.current) {
            audioRef.current.play().catch((e) => {
              console.warn("🔇 Autoplay blocked or user interaction required.");
            });
          }
        }
      } catch (err) {
        console.error("❌ Detection error:", err);
      }
    };

    fetchDetection();
    const interval = setInterval(fetchDetection, 10000);

    return () => clearInterval(interval);
  }, [soundEnabled]); // re-run when soundEnabled changes

  const handleEnableSound = () => {
    if (audioRef.current) {
      audioRef.current.play().then(() => {
        setSoundEnabled(true);
      }).catch(() => {
        console.warn("🔇 User interaction required to play audio.");
      });
    }
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
            <li>User Profile</li>
            <li>System Status</li>
            <li>Settings</li>
            <li>Help</li>
            <li>Contacts</li>
          </ul>
        </div>

        <div className="main glass fade-in">
          <div className="section">
            <h3>Live Intrusion Alerts</h3>

            {/* Enable sound button */}
            {!soundEnabled && (
              <button onClick={handleEnableSound} style={{ marginBottom: "10px", padding: "6px 12px" }}>
                🔊 Enable Alert Sound
              </button>
            )}

            {detected.length > 0 ? (
              <>
                <div className="alert-box danger">⚠ Animal Detected</div>
                <div className="detected-name" style={{ fontWeight: "bold" }}>
                  {detected.map((animal, index) => (
                    <div key={index}>🔸 {animal}</div>
                  ))}
                </div>
              </>
            ) : (
              <div className="alert-box">🔄 Detecting...</div>
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
              {detected.length === 0 ? (
                <div className="alert-box">Monitoring...</div>
              ) : (
                <>
                  <div className="alert-box danger">Animal Detected</div>
                  <div className="detected-name" style={{ fontWeight: "bold", marginBottom: "8px" }}>
                    {detected.map((animal, index) => (
                      <div key={index}>▶ {animal}</div>
                    ))}
                  </div>
                  {Object.keys(counts).length > 0 && (
                    <div>
                      <div style={{ fontWeight: "bold", marginBottom: "6px" }}>Detection Chart</div>
                      {Object.entries(counts).map(([label, count]) => {
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
        </div>
      </div>
    </div>
  );
}

export default App;
