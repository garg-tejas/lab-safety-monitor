import React, { useState, useRef, useEffect } from 'react';

function LiveFeed() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [stats, setStats] = useState({
    personCount: 0,
    compliantCount: 0,
    violationCount: 0,
  });
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Could not access webcam. Please check permissions.');
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  };

  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold text-gray-800">Live Webcam Feed</h2>
        <div className="flex gap-4">
          {!isStreaming ? (
            <button
              onClick={startWebcam}
              className="px-6 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition"
            >
              📹 Start Webcam
            </button>
          ) : (
            <button
              onClick={stopWebcam}
              className="px-6 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition"
            >
              ⏹️ Stop Webcam
            </button>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm font-medium">Persons Detected</p>
              <p className="text-3xl font-bold mt-2">{stats.personCount}</p>
            </div>
            <div className="text-4xl">👥</div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm font-medium">Compliant</p>
              <p className="text-3xl font-bold mt-2 text-green-600">{stats.compliantCount}</p>
            </div>
            <div className="text-4xl">✅</div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm font-medium">Violations</p>
              <p className="text-3xl font-bold mt-2 text-red-600">{stats.violationCount}</p>
            </div>
            <div className="text-4xl">⚠️</div>
          </div>
        </div>
      </div>

      {/* Video Feed */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="relative bg-black rounded-lg overflow-hidden" style={{ paddingTop: '56.25%' }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="absolute top-0 left-0 w-full h-full object-contain"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            style={{ pointerEvents: 'none' }}
          />
          
          {!isStreaming && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center text-white">
                <p className="text-xl mb-4">📹 Webcam Not Active</p>
                <p className="text-gray-400">Click "Start Webcam" to begin live detection</p>
              </div>
            </div>
          )}
        </div>

        {isStreaming && (
          <div className="mt-4 p-4 bg-blue-50 rounded-md">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ Info:</strong> Live detection is active. The system is analyzing the video feed in real-time for PPE compliance.
            </p>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-3">Instructions</h3>
        <ul className="list-disc list-inside space-y-2 text-gray-700">
          <li>Click "Start Webcam" to begin live PPE detection</li>
          <li>Ensure proper lighting for best detection results</li>
          <li>Person detection works best when individuals are clearly visible</li>
          <li>The system will highlight detected persons and their PPE compliance status</li>
          <li>Green bounding boxes indicate compliance, red indicates violations</li>
          <li>All detections are logged and can be viewed in the Detection Logs page</li>
        </ul>
      </div>
    </div>
  );
}

export default LiveFeed;
