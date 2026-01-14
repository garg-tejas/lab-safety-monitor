import React, { useState } from 'react';
import { uploadVideo, processVideo, getVideoSessions } from '../services/api';

function VideoUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedSession, setUploadedSession] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [sessions, setSessions] = useState([]);

  React.useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const data = await getVideoSessions();
      setSessions(data);
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validFormats = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
      if (!validFormats.includes(file.type)) {
        alert('Please select a valid video file (MP4, AVI, MOV, MKV)');
        return;
      }
      
      const maxSize = 500 * 1024 * 1024; // 500 MB
      if (file.size > maxSize) {
        alert('File size exceeds 500 MB limit');
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      setUploadedSession(result);
      setSelectedFile(null);
      alert('Video uploaded successfully!');
      loadSessions();
    } catch (err) {
      console.error('Upload failed:', err);
      alert('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleProcess = async () => {
    if (!uploadedSession) return;

    setProcessing(true);
    try {
      await processVideo(uploadedSession.session_id);
      alert('Video processing started! Check the detection logs for results.');
      setTimeout(loadSessions, 2000);
    } catch (err) {
      console.error('Processing failed:', err);
      alert('Failed to start processing. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800">Upload Video for Analysis</h2>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4">Upload New Video</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Video File
            </label>
            <input
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              disabled={uploading}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <p className="text-sm text-gray-500 mt-1">
              Supported formats: MP4, AVI, MOV, MKV (Max size: 500 MB)
            </p>
          </div>

          {selectedFile && (
            <div className="bg-gray-50 p-4 rounded-md">
              <p className="text-sm text-gray-700">
                <strong>Selected:</strong> {selectedFile.name}
              </p>
              <p className="text-sm text-gray-700">
                <strong>Size:</strong> {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          )}

          {uploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-600">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}

          <div className="flex gap-4">
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className="px-6 py-2 bg-primary text-white rounded-md hover:bg-opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? 'Uploading...' : '📤 Upload Video'}
            </button>

            {uploadedSession && (
              <button
                onClick={handleProcess}
                disabled={processing}
                className="px-6 py-2 bg-green-500 text-white rounded-md hover:bg-opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {processing ? 'Starting...' : '▶️ Process Video'}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Video Sessions */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold">Recent Video Sessions</h3>
          <button
            onClick={loadSessions}
            className="text-primary hover:underline"
          >
            🔄 Refresh
          </button>
        </div>

        {sessions.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No videos uploaded yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Video Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Upload Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Detections
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Violations
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sessions.map((session) => (
                  <tr key={session.session_id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-sm text-gray-900">{session.video_name}</td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {new Date(session.upload_time).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <StatusBadge status={session.processing_status} />
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">{session.total_detections}</td>
                    <td className="px-6 py-4 text-sm text-red-600">{session.total_violations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const colors = {
    pending: 'bg-yellow-100 text-yellow-800',
    processing: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  };

  return (
    <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${colors[status] || 'bg-gray-100 text-gray-800'}`}>
      {status}
    </span>
  );
}

export default VideoUpload;
