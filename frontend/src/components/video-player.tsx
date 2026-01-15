"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Upload,
  Play,
  Trash2,
  FileVideo,
  CheckCircle,
  XCircle,
  Loader2,
  RefreshCw,
  Eye,
  Download,
  X,
} from "lucide-react";
import { api, ProcessingJob, VideoFile } from "@/lib/api";

export function VideoProcessor() {
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [viewingJobId, setViewingJobId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const fetchVideos = useCallback(async () => {
    try {
      const result = await api.getUploadedVideos();
      setVideos(result.videos);
    } catch (err) {
      console.error("Failed to fetch videos:", err);
    }
  }, []);

  const fetchJobs = useCallback(async () => {
    try {
      const result = await api.getProcessingJobs();
      setJobs(result.jobs);

      // Stop polling if no active jobs
      const hasActiveJobs = result.jobs.some((j) => j.status === "processing");
      if (!hasActiveJobs && pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    } catch (err) {
      console.error("Failed to fetch jobs:", err);
    }
  }, []);

  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;
    pollIntervalRef.current = setInterval(() => {
      fetchJobs();
    }, 1000);
  }, [fetchJobs]);

  useEffect(() => {
    fetchVideos();
    fetchJobs();

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [fetchVideos, fetchJobs]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      // Simulate upload progress (actual progress would need XHR)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 100);

      await api.uploadVideo(file);

      clearInterval(progressInterval);
      setUploadProgress(100);

      await fetchVideos();
      setIsUploading(false);
      setUploadProgress(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setIsUploading(false);
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleProcess = async (videoPath: string) => {
    try {
      setError(null);
      await api.startProcessing(videoPath);
      await fetchJobs();
      startPolling();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start processing");
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await api.deleteVideo(filename);
      await fetchVideos();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete video");
    }
  };

  const getJobForVideo = (videoPath: string): ProcessingJob | undefined => {
    return jobs.find((j) => j.video_path === videoPath || j.filename === videoPath.split("/").pop());
  };

  const getStatusBadge = (job: ProcessingJob | undefined) => {
    if (!job) return null;

    switch (job.status) {
      case "pending":
        return (
          <Badge variant="secondary">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Pending
          </Badge>
        );
      case "processing":
        return (
          <Badge variant="default">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Processing {job.progress}%
          </Badge>
        );
      case "completed":
        return (
          <Badge variant="outline" className="text-green-600 border-green-600">
            <CheckCircle className="w-3 h-3 mr-1" />
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <XCircle className="w-3 h-3 mr-1" />
            Failed
          </Badge>
        );
      default:
        return null;
    }
  };

  const handleViewProcessed = (jobId: string) => {
    setViewingJobId(jobId);
    setIsStreaming(false);
  };

  const handleCloseViewer = () => {
    setViewingJobId(null);
    setIsStreaming(false);
    if (videoRef.current) {
      videoRef.current.pause();
    }
  };

  const handleDownloadProcessed = (jobId: string) => {
    const url = api.getProcessedVideoUrl(jobId);
    window.open(url, '_blank');
  };

  // Get the viewing job details
  const viewingJob = viewingJobId ? jobs.find(j => j.id === viewingJobId) : null;

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <FileVideo className="w-5 h-5" />
          Video Analysis
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={() => { fetchVideos(); fetchJobs(); }}>
          <RefreshCw className="w-4 h-4" />
        </Button>
      </CardHeader>
      <CardContent>
        {/* Annotated Video Viewer Modal */}
        {viewingJobId && viewingJob && (
          <div className="mb-6 border rounded-lg overflow-hidden bg-black">
            <div className="flex items-center justify-between bg-gray-900 px-4 py-2">
              <div className="flex items-center gap-2 text-white">
                <Eye className="w-4 h-4" />
                <span className="text-sm font-medium">
                  Processed Video: {viewingJob.filename}
                </span>
                <Badge variant="secondary" className="ml-2">
                  {viewingJob.unique_events || viewingJob.violations_count} violations detected
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:text-white hover:bg-gray-700"
                  onClick={() => handleDownloadProcessed(viewingJobId)}
                >
                  <Download className="w-4 h-4" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white hover:text-white hover:bg-gray-700"
                  onClick={handleCloseViewer}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
            <div className="relative aspect-video bg-gray-900">
              {isStreaming ? (
                // MJPEG streaming mode for real-time playback
                <img
                  src={api.getProcessedVideoStreamUrl(viewingJobId)}
                  alt="Processed video stream"
                  className="w-full h-full object-contain"
                />
              ) : (
                // Standard HTML5 video player
                <video
                  ref={videoRef}
                  src={api.getProcessedVideoUrl(viewingJobId)}
                  controls
                  autoPlay
                  className="w-full h-full object-contain"
                  onError={() => {
                    setError("Failed to load processed video. Try streaming mode.");
                  }}
                >
                  Your browser does not support the video tag.
                </video>
              )}
            </div>
            <div className="bg-gray-900 px-4 py-2 flex justify-between items-center">
              <div className="text-xs text-gray-400">
                {viewingJob.persons_count} person(s) tracked | {viewingJob.processed_frames} frames processed
              </div>
              <Button
                size="sm"
                variant="outline"
                className="text-xs"
                onClick={() => setIsStreaming(!isStreaming)}
              >
                {isStreaming ? "Switch to Video" : "Switch to Stream"}
              </Button>
            </div>
          </div>
        )}

        {/* Upload section */}
        <div className="mb-6">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/avi,video/mov,video/mkv,video/webm"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />
          <Button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full"
            variant="outline"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Uploading... {uploadProgress}%
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Upload Video
              </>
            )}
          </Button>
          {isUploading && (
            <Progress value={uploadProgress} className="mt-2" />
          )}
        </div>

        {/* Error display */}
        {error && (
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded-lg mb-4">
            {error}
          </div>
        )}

        {/* Videos list */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-muted-foreground">
            Uploaded Videos ({videos.length})
          </h4>

          {videos.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <FileVideo className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No videos uploaded yet</p>
              <p className="text-sm">Upload a video to start analysis</p>
            </div>
          ) : (
            videos.map((video) => {
              const job = getJobForVideo(video.path);
              const isProcessing = job?.status === "processing";
              const isCompleted = job?.status === "completed";
              const hasProcessedVideo = isCompleted && job?.output_video;

              return (
                <div
                  key={video.path}
                  className="flex items-center justify-between p-3 border rounded-lg"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <FileVideo className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
                      <span className="font-medium truncate">{video.filename}</span>
                      {getStatusBadge(job)}
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      {video.size_mb} MB
                      {job?.status === "completed" && (
                        <span className="ml-2">
                          | {job.unique_events || job.violations_count} violations | {job.persons_count} persons
                        </span>
                      )}
                    </div>
                    {job?.status === "processing" && (
                      <Progress value={job.progress} className="mt-2 h-1" />
                    )}
                    {job?.status === "failed" && job.error && (
                      <p className="text-sm text-red-600 mt-1">{job.error}</p>
                    )}
                  </div>

                  <div className="flex items-center gap-2 ml-4">
                    {/* View processed video button */}
                    {hasProcessedVideo && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleViewProcessed(job.id)}
                        className="text-blue-600 border-blue-600 hover:bg-blue-50"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        View
                      </Button>
                    )}
                    {(!job || job.status === "failed") && (
                      <Button
                        size="sm"
                        onClick={() => handleProcess(video.path)}
                        disabled={isProcessing}
                      >
                        <Play className="w-4 h-4 mr-1" />
                        Analyze
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDelete(video.filename)}
                      disabled={isProcessing}
                    >
                      <Trash2 className="w-4 h-4 text-muted-foreground" />
                    </Button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Active jobs summary */}
        {jobs.filter((j) => j.status === "processing").length > 0 && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center gap-2 text-blue-700">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm font-medium">
                {jobs.filter((j) => j.status === "processing").length} video(s) being analyzed...
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Keep old export name for compatibility
export const VideoPlayer = VideoProcessor;
