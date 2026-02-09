import React, { useEffect, useState, useRef } from "react";
import "./App.css";
import axios from "axios";

function App() {
  const [detected, setDetected] = useState([]);
  const [counts, setCounts] = useState({});
  const [videoScores, setVideoScores] = useState({});
  const [audioDetected, setAudioDetected] = useState([]);
  const [audioScores, setAudioScores] = useState({});
  const [soundEnabled, setSoundEnabled] = useState(false);
  const audioRef = useRef(null);
  const previousDetectedRef = useRef([]);
  const ipCameraStreamUrl = "http://100.104.143.1:8080/video";
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
  const animalDetected = nonHumanDetected.length > 0;
  const TOP_K = 3;
  const cleanLabel = (label) => String(label).toLowerCase();
  const isHuman = (label) => cleanLabel(label) === "human";
  const MIN_VIDEO_SCORE = 0.15;
  const MIN_AUDIO_SCORE = 0.06;
  const topVideo = Object.entries(videoScores)
    .filter(([label, score]) => !isHuman(label) && (score ?? 0) >= MIN_VIDEO_SCORE)
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
    .slice(0, TOP_K)
    .map(([label]) => label);
  const topAudio = Object.entries(audioScores)
    .filter(([label, score]) => !isHuman(label) && (score ?? 0) >= MIN_AUDIO_SCORE)
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
    .slice(0, TOP_K)
    .map(([label]) => label);
  const intersection = topVideo.filter((a) =>
    topAudio.some((b) => cleanLabel(b) === cleanLabel(a))
  );
  const finalIntersection = intersection.sort((a, b) => {
    const scoreA = Math.max(videoScores[a] ?? 0, audioScores[a] ?? 0);
    const scoreB = Math.max(videoScores[b] ?? 0, audioScores[b] ?? 0);
    return scoreB - scoreA;
  });
  const finalWinner = finalIntersection.length > 0 ? finalIntersection[0] : null;
  const topAudioLabels = Object.entries(audioScores)
    .filter(([label]) => !isHuman(label))
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
    .slice(0, 3)
    .map(([label, score]) => `${label} (${Math.round((score ?? 0) * 100)}%)`);

  const finalIsDangerous = finalIntersection.some((animal) =>
    dangerousAnimals.includes(String(animal).toLowerCase())
  );
  const audioDangerousHigh = Object.entries(audioScores).some(
    ([label, score]) =>
      dangerousAnimals.includes(String(label).toLowerCase()) && score >= 0.8
  );
  const shouldAlert = isDangerous || finalIsDangerous || audioDangerousHigh;
  const lastAlertRef = useRef(false);

  useEffect(() => {
    const fetchDetection = async () => {
      try {
        const res = await axios.get("http://localhost:5000/detect");
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
  }, [soundEnabled]); // re-run when soundEnabled changes

  useEffect(() => {
    const fetchAudioDetection = async () => {
      try {
        const res = await axios.get("http://localhost:5000/audio_detect");
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
  }, []);

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

            {animalDetected ? (
              <>
                <div className="alert-box danger">⚠ Animal Detected</div>
                <div className="detected-name" style={{ fontWeight: "bold" }}>
                  {nonHumanDetected.map((animal, index) => (
                    <div key={index}>🔸 {animal}</div>
                  ))}
                </div>
              </>
            ) : (
              <div className="alert-box">
                {onlyHumanDetected ? "✅ No animal detected (human only)" : "🔄 Detecting..."}
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
                    {topVideo.map((animal, index) => (
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
                {Object.entries(audioScores).map(([label, score]) => {
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

          <div className="section">
            <h3>Final Detection</h3>
            {finalWinner === null ? (
              <div className="alert-box">Waiting for detection...</div>
            ) : (
              <>
                <div className="alert-box danger">Final Detection</div>
                <div className="detected-name" style={{ fontWeight: "bold", marginBottom: "8px" }}>
                  <div>▶ {finalWinner}</div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
