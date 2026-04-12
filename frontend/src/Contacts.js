import React from "react";
import "./Contacts.css";

function Contacts() {
  return (
    <div className="page-container contacts-page">
      <header className="contacts-hero">
        <h2 className="page-title">Emergency Contacts</h2>
        <p className="contacts-subtitle">
          Reach the right team quickly during wildlife intrusion alerts.
        </p>
      </header>

      <div className="contacts-grid">
        <div className="card-box contact-card">
          <h3 className="contact-title">Forest Department</h3>
          <p className="contact-label">Phone</p>
          <p className="contact-value">+91 98765 43210</p>
        </div>

        <div className="card-box contact-card">
          <h3 className="contact-title">Farmer Helpline</h3>
          <p className="contact-label">Phone</p>
          <p className="contact-value">+91 91234 56789</p>
        </div>

        <div className="card-box contact-card developers-card">
          <h3 className="contact-title">Project Developers</h3>
          <ul className="developer-list">
            <li>
              <span className="developer-name">Aastha Londhe</span>
              <span className="developer-email">nnm22ad001@nmamit.in</span>
            </li>
            <li>
              <span className="developer-name">Ambika Jayashanthi</span>
              <span className="developer-email">nnm22ad006@nmamit.in</span>
            </li>
            <li>
              <span className="developer-name">Ankitha</span>
              <span className="developer-email">nnm22ad009@nmamit.in</span>
            </li>
            <li>
              <span className="developer-name">Ruchitha Prabhu</span>
              <span className="developer-email">nnm22ad045@nmamit.in</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Contacts;
