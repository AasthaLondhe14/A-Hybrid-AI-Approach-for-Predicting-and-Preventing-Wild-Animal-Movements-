import React from "react";

function Help() {
  return (
    <div className="page-container">
      <h2 className="page-title">❓ Help & User Guide</h2>

      <div className="card-box">
        <ul className="info-list">
          <li>📷 Connect an IP camera to monitor wildlife activity.</li>
          <li>🤖 The AI model analyzes video and audio data.</li>
          <li>⚠️ When a dangerous animal is detected, an alert sound is triggered.</li>
          <li>📧 Email notifications are sent to the registered user.</li>
          <li>📊 Detection history is saved for future analysis.</li>
        </ul>
      </div>
    </div>
  );
}

export default Help;
