import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import VideoUpload from './pages/VideoUpload';
import DetectionLogs from './pages/DetectionLogs';
import LiveFeed from './pages/LiveFeed';

function App() {
  const [activeRoute, setActiveRoute] = useState('/');

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        {/* Navigation */}
        <nav className="gradient-bg text-white shadow-lg">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <h1 className="text-2xl font-bold">🛡️ PPE Compliance System</h1>
              </div>
              <div className="flex space-x-4">
                <NavLink to="/" label="Dashboard" active={activeRoute === '/'} onClick={() => setActiveRoute('/')} />
                <NavLink to="/live" label="Live Feed" active={activeRoute === '/live'} onClick={() => setActiveRoute('/live')} />
                <NavLink to="/upload" label="Upload Video" active={activeRoute === '/upload'} onClick={() => setActiveRoute('/upload')} />
                <NavLink to="/logs" label="Detection Logs" active={activeRoute === '/logs'} onClick={() => setActiveRoute('/logs')} />
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveFeed />} />
            <Route path="/upload" element={<VideoUpload />} />
            <Route path="/logs" element={<DetectionLogs />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-gray-800 text-white py-4 mt-12">
          <div className="container mx-auto px-4 text-center">
            <p>© 2026 PPE Safety Compliance System - IIIT Nagpur</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

function NavLink({ to, label, active, onClick }) {
  return (
    <Link
      to={to}
      onClick={onClick}
      className={`px-4 py-2 rounded-md transition ${
        active
          ? 'bg-white bg-opacity-20 font-semibold'
          : 'hover:bg-white hover:bg-opacity-10'
      }`}
    >
      {label}
    </Link>
  );
}

export default App;
