const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SummaryStats {
  total_events: number;
  today_events: number;
  total_violations: number;
  today_violations: number;
  total_persons: number;
  compliance_rate: number;
  last_updated: string;
}

export interface ComplianceEvent {
  id: string;
  person_id: string | null;
  timestamp: string;
  video_source: string | null;
  frame_number?: number;
  detected_ppe: string[];
  missing_ppe: string[];
  action_violations?: string[];  // Drinking/Eating violations
  is_violation: boolean;
  // Event deduplication fields
  start_frame?: number | null;
  end_frame?: number | null;
  end_timestamp?: string | null;
  duration_frames?: number;
  is_ongoing?: boolean;
}

export interface Person {
  id: string;
  name: string | null;
  first_seen: string;
  last_seen: string;
  total_events: number;
  violation_count: number;
  compliance_rate: number;
}

export interface TimelineData {
  date: string;
  violations: number;
}

export interface PPEBreakdown {
  ppe_type: string;
  count: number;
}

export interface ProcessingJob {
  id: string;
  video_path: string;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  total_frames: number;
  processed_frames: number;
  violations_count: number;
  persons_count: number;
  unique_events?: number;  // Deduplicated event count
  error: string | null;
  output_video: string | null;  // Path to annotated video
}

export interface VideoFile {
  filename: string;
  path: string;
  size_mb: number;
}

export interface ProcessingStatus {
  active_jobs: number;
  total_jobs: number;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Stats endpoints
  async getSummaryStats(): Promise<SummaryStats> {
    return this.fetch<SummaryStats>('/api/stats/summary');
  }

  async getViolationTimeline(days: number = 7): Promise<TimelineData[]> {
    return this.fetch<TimelineData[]>(`/api/stats/timeline?days=${days}`);
  }

  async getViolationsByPPE(): Promise<PPEBreakdown[]> {
    return this.fetch<PPEBreakdown[]>('/api/stats/by-ppe');
  }

  // Events endpoints
  async getEvents(params: {
    page?: number;
    page_size?: number;
    person_id?: string;
    violations_only?: boolean;
  } = {}): Promise<{ events: ComplianceEvent[]; total: number; page: number; page_size: number }> {
    const searchParams = new URLSearchParams();
    if (params.page) searchParams.set('page', params.page.toString());
    if (params.page_size) searchParams.set('page_size', params.page_size.toString());
    if (params.person_id) searchParams.set('person_id', params.person_id);
    if (params.violations_only) searchParams.set('violations_only', 'true');

    return this.fetch(`/api/events?${searchParams.toString()}`);
  }

  async getRecentViolations(limit: number = 10): Promise<ComplianceEvent[]> {
    return this.fetch<ComplianceEvent[]>(`/api/events/recent/violations?limit=${limit}`);
  }

  // Persons endpoints
  async getPersons(page: number = 1, pageSize: number = 20): Promise<{ persons: Person[]; total: number }> {
    return this.fetch(`/api/persons?page=${page}&page_size=${pageSize}`);
  }

  async getPerson(personId: string): Promise<Person> {
    return this.fetch<Person>(`/api/persons/${personId}`);
  }

  async getTopViolators(limit: number = 5): Promise<Person[]> {
    return this.fetch<Person[]>(`/api/persons/top/violators?limit=${limit}`);
  }

  async updatePerson(personId: string, name: string | null): Promise<Person> {
    return this.fetch<Person>(`/api/persons/${personId}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    });
  }

  // Video processing endpoints
  async getProcessingStatus(): Promise<ProcessingStatus> {
    return this.fetch<ProcessingStatus>('/api/stream/status');
  }

  async uploadVideo(file: File): Promise<{ message: string; filename: string; path: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/stream/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    return response.json();
  }

  async startProcessing(videoPath: string): Promise<{ message: string; job_id: string; video_path: string }> {
    return this.fetch('/api/stream/process', {
      method: 'POST',
      body: JSON.stringify({ video_path: videoPath }),
    });
  }

  async getProcessingJobs(): Promise<{ jobs: ProcessingJob[] }> {
    return this.fetch<{ jobs: ProcessingJob[] }>('/api/stream/jobs');
  }

  async getJobStatus(jobId: string): Promise<ProcessingJob> {
    return this.fetch<ProcessingJob>(`/api/stream/jobs/${jobId}`);
  }

  async getUploadedVideos(): Promise<{ videos: VideoFile[] }> {
    return this.fetch<{ videos: VideoFile[] }>('/api/stream/videos');
  }

  async deleteVideo(filename: string): Promise<{ message: string }> {
    return this.fetch<{ message: string }>(`/api/stream/videos/${filename}`, {
      method: 'DELETE',
    });
  }

  // Get the URL for the processed/annotated video
  getProcessedVideoUrl(jobId: string): string {
    return `${this.baseUrl}/api/stream/processed/${jobId}`;
  }

  // Get the URL for streaming the processed video as MJPEG
  getProcessedVideoStreamUrl(jobId: string): string {
    return `${this.baseUrl}/api/stream/processed/${jobId}/stream`;
  }

  // Get the URL for live webcam feed with real-time detection
  getLiveFeedUrl(): string {
    return `${this.baseUrl}/api/stream/live/feed`;
  }
}

export const api = new ApiClient();
export default api;
