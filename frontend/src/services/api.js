import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Statistics
export const getStatistics = async (startDate, endDate) => {
  const params = {};
  if (startDate) params.start_date = startDate;
  if (endDate) params.end_date = endDate;
  
  const response = await api.get('/statistics', { params });
  return response.data;
};

// Detection Logs
export const getDetections = async (filters = {}) => {
  const response = await api.get('/detections', { params: filters });
  return response.data;
};

export const getDetection = async (eventId) => {
  const response = await api.get(`/detections/${eventId}`);
  return response.data;
};

// Persons
export const getPersons = async (limit = 50, offset = 0) => {
  const response = await api.get('/persons', { params: { limit, offset } });
  return response.data;
};

export const getPerson = async (personId) => {
  const response = await api.get(`/persons/${personId}`);
  return response.data;
};

export const getPersonEvents = async (personId, limit = 100, offset = 0) => {
  const response = await api.get(`/persons/${personId}/events`, {
    params: { limit, offset }
  });
  return response.data;
};

// Video Upload & Processing
export const uploadVideo = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload-video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      if (onProgress) onProgress(percentCompleted);
    },
  });
  
  return response.data;
};

export const processVideo = async (sessionId) => {
  const response = await api.post(`/process-video/${sessionId}`);
  return response.data;
};

export const getVideoSessions = async (limit = 10, offset = 0) => {
  const response = await api.get('/video-sessions', { params: { limit, offset } });
  return response.data;
};

export const getVideoSession = async (sessionId) => {
  const response = await api.get(`/video-sessions/${sessionId}`);
  return response.data;
};

export default api;
