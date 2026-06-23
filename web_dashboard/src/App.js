// FILE: App.js
// PATH: AgroSaarthi_AI/web_dashboard/src/App.js

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { collection, onSnapshot } from 'firebase/firestore';
import { db } from './firebase'; // Imported our new DB connection
import './App.css';

// Fix for default Leaflet markers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

function App() {
  // State is now empty initially
  const [outbreaks, setOutbreaks] = useState([]);

  // useEffect runs once when the dashboard opens
  useEffect(() => {
    // onSnapshot creates a live connection. Any change in DB instantly triggers this.
    const unsubscribe = onSnapshot(collection(db, 'disease_outbreaks'), (snapshot) => {
      const liveData = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      setOutbreaks(liveData);
      console.log("Live Data Fetched:", liveData);
    });

    // Cleanup listener when component unmounts
    return () => unsubscribe();
  }, []);

  const mapCenter = [26.4499, 80.3319]; // Default focus
  const zoomLevel = 10;

  return (
    <div className="App">
      <header className="dashboard-header">
        <h1>AgroSaarthi AI - Live Outbreak Dashboard</h1>
        <p>Real-time tracking of crop diseases connected to Firebase</p>
      </header>

      <div className="map-container-wrapper">
        <MapContainer center={mapCenter} zoom={zoomLevel} scrollWheelZoom={true} style={{ height: '70vh', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {outbreaks.map((point) => (
            <Marker key={point.id} position={[point.latitude, point.longitude]}>
              <Popup>
                <div>
                  <h3>🚨 {point.disease}</h3>
                  <p><strong>Risk Level:</strong> {point.risk_level}</p>
                  <p><strong>Time:</strong> {new Date(point.timestamp).toLocaleString()}</p>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;