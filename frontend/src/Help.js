import React from "react";
import "./Help.css";

function Help() {
  return (
    <div className="page-container help-guide">
      <header className="guide-hero">
        <h2 className="page-title">Help & User Guide</h2>
        <p className="guide-subtitle">
          Quick setup, alert rules, and what to expect during monitoring.
        </p>
      </header>

      <div className="guide-grid">
        <section className="card-box guide-card">
          <h3 className="guide-title">Quick Start</h3>
          <ol className="guide-steps">
            <li>Connect the IP camera and verify the live feed appears.</li>
            <li>Keep the system running to analyze video and audio in real time.</li>
            <li>Check the Prediction Results tab for the latest detections.</li>
          </ol>
        </section>

        <section className="card-box guide-card">
          <h3 className="guide-title">Alerts & Severity</h3>
          <div className="status-row">
            <span className="status-chip danger">Danger</span>
            <p>Dangerous animals trigger alert sound and warnings.</p>
          </div>
          <div className="status-row">
            <span className="status-chip safe">Safe</span>
            <p>Non-dangerous detections remain green and informational.</p>
          </div>
        </section>

        <section className="card-box guide-card">
          <h3 className="guide-title">Notifications</h3>
          <ul className="guide-list">
            <li>Alert sound plays when a dangerous animal is detected.</li>
            <li>Email notifications are sent to the registered user.</li>
          </ul>
        </section>

        <section className="card-box guide-card">
          <h3 className="guide-title">History & Analysis</h3>
          <ul className="guide-list">
            <li>Detection history is stored for later review.</li>
            <li>Use history to identify patterns and high-risk times.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}

export default Help;
