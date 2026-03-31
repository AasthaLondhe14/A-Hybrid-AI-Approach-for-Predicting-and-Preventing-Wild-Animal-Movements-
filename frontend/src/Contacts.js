import React from "react";

function Contacts() {
  return (
    <div className="page-container">
      <h2 className="page-title">📞 Emergency Contacts</h2>

      <div className="card-box">
        <p><strong>Forest Department</strong></p>
        <p>Phone: +91 98765 43210</p>
      </div>

      <div className="card-box">
        <p><strong>Farmer Helpline</strong></p>
        <p>Phone: +91 91234 56789</p>
      </div>

      <div className="card-box">
        <p><strong>Project Developer</strong></p>
        <p>Name: Ankitha</p>
        <p>Email: ankitha@email.com</p>
      </div>
    </div>
  );
}

export default Contacts;
