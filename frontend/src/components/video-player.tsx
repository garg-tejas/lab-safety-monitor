"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
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
  Film,
  Sparkles,
} from "lucide-react";
import { api, ProcessingJob, VideoFile } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useDemoMode } from "@/providers/demo-context";
import { demoVideos, demoProcessingJobs } from "@/lib/demo-data";

export function VideoProcessor() {
  const { isDemoMode } = useDemoMode();
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [viewingJobId, setViewingJobId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const fetchVideos = useCallback(async () => {
    if (isDemoMode) {
      setVideos(demoVideos);
      return;
    }
    try {
      const result = await api.getUploadedVideos();
      setVideos(result.videos);
    } catch (err) {
      console.error("Failed to fetch videos:", err);
    }
  }, [isDemoMode]);

  const fetchJobs = useCallback(async () => {
    if (isDemoMode) {
      setJobs(demoProcessingJobs);
      return;
    }
    try {
      const result = await api.getProcessingJobs();
      setJobs(result.jobs);

      const hasActiveJobs = result.jobs.some((j) => j.status === "processing");
      if (!hasActiveJobs && pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    } catch (err) {
      console.error("Failed to fetch jobs:", err);
    }
  }, [isDemoMode]);

  const startPolling = useCallback(() => {
    if (isDemoMode) return; // Don't poll in demo mode
    if (pollIntervalRef.current) return;
    pollIntervalRef.current = setInterval(() => {
      fetchJobs();
    }, 1000);
  }, [fetchJobs, isDemoMode]);

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

    if (isDemoMode) {
      setError("Video upload is disabled in demo mode");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
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

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleProcess = async (videoPath: string) => {
    if (isDemoMode) {
      setError("Video processing is disabled in demo mode");
      return;
    }
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
    if (isDemoMode) {
      setError("Video deletion is disabled in demo mode");
      return;
    }
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
          <Badge variant="secondary" className="gap-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            Pending
          </Badge>
        );
      case "processing":
        return (
          <Badge variant="info" className="gap-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            Processing {job.progress}%
          </Badge>
        );
      case "completed":
        return (
          <Badge variant="success" className="gap-1">
            <CheckCircle className="w-3 h-3" />
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="danger" className="gap-1">
            <XCircle className="w-3 h-3" />
            Failed
          </Badge>
        );
      default:
        return null;
    }
  };

  const handleViewProcessed = (jobId: string) => {
    setViewingJobId(jobId);
    setIsStreaming(true);
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

  const viewingJob = viewingJobId ? jobs.find(j => j.id === viewingJobId) : null;

  return (
    <Card variant="glass" hover className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10">
            <Film className="w-5 h-5 text-primary" />
          </div>
          <div>
            <span>Video Analysis</span>
            <p className="text-xs font-normal text-muted-foreground mt-0.5">
              Upload and process videos for safety analysis
            </p>
          </div>
        </CardTitle>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => { fetchVideos(); fetchJobs(); }}
          className="h-8 w-8"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
      </CardHeader>
      <CardContent>
        {/* Annotated Video Viewer Modal */}
        {viewingJobId && viewingJob && (
          <div className="mb-6 rounded-xl overflow-hidden border border-border/50 shadow-2xl">
            <div className="flex items-center justify-between glass px-4 py-3">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
                  <Eye className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <span className="text-sm font-medium">
                    {viewingJob.filename}
                  </span>
                  <div className="flex items-center gap-2 mt-0.5">
                    <Badge variant="danger-soft" className="text-[10px]">
                      {viewingJob.unique_events || viewingJob.violations_count} violations
                    </Badge>
                    <Badge variant="info-soft" className="text-[10px]">
                      {viewingJob.persons_count} persons
                    </Badge>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  size="icon-sm"
                  variant="ghost"
                  className="h-8 w-8"
                  onClick={() => handleDownloadProcessed(viewingJobId)}
                >
                  <Download className="w-4 h-4" />
                </Button>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  className="h-8 w-8"
                  onClick={handleCloseViewer}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
            <div className="relative aspect-video bg-black">
              {isDemoMode ? (
                /* Demo mode: use local video file */
                <video
                  key="demo-video"
                  ref={videoRef}
                  controls
                  autoPlay
                  muted
                  playsInline
                  loop
                  className="w-full h-full object-contain"
                  onLoadedData={() => console.log("Demo video loaded successfully")}
                  onError={(e) => {
                    console.error("Demo video error:", e);
                    setError("Failed to load demo video");
                  }}
                >
                  <source src="/demo/demo_h264.mp4" type="video/mp4" />
                  <source src="/demo/demo.webm" type="video/webm" />
                  Your browser does not support the video tag.
                </video>
              ) : isStreaming ? (
                <img
                  src={api.getProcessedVideoStreamUrl(viewingJobId)}
                  alt="Processed video stream"
                  className="w-full h-full object-contain"
                />
              ) : (
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
              {/* Streaming indicator */}
              {(isStreaming || isDemoMode) && (
                <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1.5 bg-black/60 rounded-full text-xs text-white backdrop-blur-md">
                  <Sparkles className="w-3 h-3 text-primary" />
                  {isDemoMode ? 'Demo Video' : 'AI Annotated Stream'}
                </div>
              )}
            </div>
            <div className="glass px-4 py-2 flex justify-between items-center">
              <div className="text-xs text-muted-foreground">
                {viewingJob.processed_frames} frames processed
              </div>
              <Button
                size="sm"
                variant="outline"
                className="text-xs h-7"
                onClick={() => setIsStreaming(!isStreaming)}
              >
                {isStreaming ? "Switch to Video File" : "Switch to Stream"}
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
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className={cn(
              "w-full py-8 border-2 border-dashed rounded-xl transition-all duration-200",
              "flex flex-col items-center justify-center gap-3",
              isUploading
                ? "border-primary/50 bg-primary/5 cursor-wait"
                : "border-border hover:border-primary/50 hover:bg-primary/5 cursor-pointer"
            )}
          >
            {isUploading ? (
              <>
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
                <div className="text-center">
                  <p className="font-medium">Uploading... {uploadProgress}%</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Please wait while your video is being uploaded
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-primary/10">
                  <Upload className="w-6 h-6 text-primary" />
                </div>
                <div className="text-center">
                  <p className="font-medium">Upload Video</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Drag and drop or click to select a video file
                  </p>
                </div>
              </>
            )}
          </button>
          {isUploading && (
            <div className="mt-3">
              <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Error display */}
        {error && (
          <div className="flex items-center gap-3 bg-danger/10 text-danger px-4 py-3 rounded-xl mb-4 border border-danger/20">
            <XCircle className="w-5 h-5 shrink-0" />
            <p className="text-sm">{error}</p>
            <Button
              size="icon-sm"
              variant="ghost"
              className="ml-auto h-6 w-6 text-danger hover:text-danger hover:bg-danger/10"
              onClick={() => setError(null)}
            >
              <X className="w-3.5 h-3.5" />
            </Button>
          </div>
        )}

        {/* Videos list */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <FileVideo className="w-4 h-4" />
            Uploaded Videos ({videos.length})
          </h4>

          {videos.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center border-2 border-dashed border-border rounded-xl bg-muted/10">
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-muted mb-3">
                <FileVideo className="w-6 h-6 text-muted-foreground" />
              </div>
              <p className="font-medium">No videos uploaded yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Upload a video to start analysis
              </p>
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
                  className={cn(
                    "flex items-center justify-between p-4 rounded-xl border transition-all duration-200",
                    isProcessing
                      ? "border-primary/30 bg-primary/5"
                      : isCompleted
                        ? "border-success/30 bg-success/5 hover:border-success/50"
                        : "border-border/50 hover:border-border"
                  )}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "flex items-center justify-center w-10 h-10 rounded-lg shrink-0",
                          isCompleted ? "bg-success/10" : "bg-muted"
                        )}
                      >
                        <FileVideo
                          className={cn(
                            "w-5 h-5",
                            isCompleted ? "text-success" : "text-muted-foreground"
                          )}
                        />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-medium truncate">{video.filename}</span>
                          {getStatusBadge(job)}
                        </div>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground mt-0.5">
                          <span>{video.size_mb} MB</span>
                          {job?.status === "completed" && (
                            <>
                              <span className="text-border">|</span>
                              <span>{job.unique_events || job.violations_count} violations</span>
                              <span className="text-border">|</span>
                              <span>{job.persons_count} persons</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    {job?.status === "processing" && (
                      <div className="mt-3 ml-13">
                        <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary rounded-full transition-all duration-300"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {job?.status === "failed" && job.error && (
                      <p className="text-sm text-danger mt-2 ml-13">{job.error}</p>
                    )}
                  </div>

                  <div className="flex items-center gap-2 ml-4">
                    {hasProcessedVideo && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleViewProcessed(job.id)}
                        className="text-primary border-primary/30 hover:bg-primary/10 hover:border-primary/50"
                      >
                        <Eye className="w-4 h-4 mr-1.5" />
                        View
                      </Button>
                    )}
                    {(!job || job.status === "failed") && (
                      <Button
                        size="sm"
                        onClick={() => handleProcess(video.path)}
                        disabled={isProcessing}
                        className="shadow-lg"
                      >
                        <Play className="w-4 h-4 mr-1.5" />
                        Analyze
                      </Button>
                    )}
                    <Button
                      size="icon-sm"
                      variant="ghost"
                      onClick={() => handleDelete(video.filename)}
                      disabled={isProcessing}
                      className="h-8 w-8 text-muted-foreground hover:text-danger hover:bg-danger/10"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Active jobs summary */}
        {jobs.filter((j) => j.status === "processing").length > 0 && (
          <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-primary/10 to-info/10 border border-primary/20">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/20">
                <Loader2 className="w-5 h-5 text-primary animate-spin" />
              </div>
              <div>
                <p className="font-medium text-foreground">
                  {jobs.filter((j) => j.status === "processing").length} video(s) being analyzed
                </p>
                <p className="text-sm text-muted-foreground">
                  AI is processing your videos. This may take a few minutes.
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Keep old export name for compatibility
export const VideoPlayer = VideoProcessor;
