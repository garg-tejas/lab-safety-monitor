"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api, {
  SummaryStats,
  ComplianceEvent,
  Person,
  TimelineData,
  PPEBreakdown,
  ProcessingJob,
  VideoFile,
} from "@/lib/api";

// Query keys for cache management
export const queryKeys = {
  stats: {
    summary: ["stats", "summary"] as const,
    timeline: (days: number) => ["stats", "timeline", days] as const,
    byPPE: ["stats", "by-ppe"] as const,
  },
  events: {
    all: ["events"] as const,
    list: (params: {
      page?: number;
      pageSize?: number;
      personId?: string;
      violationsOnly?: boolean;
    }) => ["events", "list", params] as const,
    recentViolations: (limit: number) => ["events", "recent", limit] as const,
  },
  persons: {
    all: ["persons"] as const,
    list: (page: number, pageSize: number) => ["persons", "list", page, pageSize] as const,
    detail: (id: string) => ["persons", "detail", id] as const,
    topViolators: (limit: number) => ["persons", "top", limit] as const,
  },
  videos: {
    all: ["videos"] as const,
    list: ["videos", "list"] as const,
    jobs: ["videos", "jobs"] as const,
    job: (id: string) => ["videos", "job", id] as const,
  },
};

// ============ Stats Queries ============

export function useSummaryStats() {
  return useQuery({
    queryKey: queryKeys.stats.summary,
    queryFn: () => api.getSummaryStats(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

export function useViolationTimeline(days: number = 7) {
  return useQuery({
    queryKey: queryKeys.stats.timeline(days),
    queryFn: () => api.getViolationTimeline(days),
    staleTime: 60000, // Data is fresh for 1 minute
  });
}

export function useViolationsByPPE() {
  return useQuery({
    queryKey: queryKeys.stats.byPPE,
    queryFn: () => api.getViolationsByPPE(),
    staleTime: 60000,
  });
}

// ============ Events Queries ============

export function useEvents(params: {
  page?: number;
  pageSize?: number;
  personId?: string;
  violationsOnly?: boolean;
} = {}) {
  return useQuery({
    queryKey: queryKeys.events.list(params),
    queryFn: () =>
      api.getEvents({
        page: params.page,
        page_size: params.pageSize,
        person_id: params.personId,
        violations_only: params.violationsOnly,
      }),
  });
}

export function useRecentViolations(limit: number = 10) {
  return useQuery({
    queryKey: queryKeys.events.recentViolations(limit),
    queryFn: () => api.getRecentViolations(limit),
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

// ============ Persons Queries ============

export function usePersons(page: number = 1, pageSize: number = 20) {
  return useQuery({
    queryKey: queryKeys.persons.list(page, pageSize),
    queryFn: () => api.getPersons(page, pageSize),
  });
}

export function usePerson(personId: string) {
  return useQuery({
    queryKey: queryKeys.persons.detail(personId),
    queryFn: () => api.getPerson(personId),
    enabled: !!personId,
  });
}

export function useTopViolators(limit: number = 5) {
  return useQuery({
    queryKey: queryKeys.persons.topViolators(limit),
    queryFn: () => api.getTopViolators(limit),
  });
}

export function useUpdatePerson() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ personId, name }: { personId: string; name: string | null }) =>
      api.updatePerson(personId, name),
    onSuccess: (updatedPerson) => {
      // Update the person in the cache
      queryClient.setQueryData(
        queryKeys.persons.detail(updatedPerson.id),
        updatedPerson
      );
      // Invalidate the persons list to trigger a refetch
      queryClient.invalidateQueries({ queryKey: queryKeys.persons.all });
    },
  });
}

// ============ Video Queries ============

export function useUploadedVideos() {
  return useQuery({
    queryKey: queryKeys.videos.list,
    queryFn: () => api.getUploadedVideos(),
  });
}

export function useProcessingJobs() {
  return useQuery({
    queryKey: queryKeys.videos.jobs,
    queryFn: () => api.getProcessingJobs(),
    refetchInterval: (query) => {
      // Refetch every second if there are active jobs
      const hasActiveJobs = query.state.data?.jobs?.some((j) => j.status === "processing");
      return hasActiveJobs ? 1000 : false;
    },
  });
}

export function useJobStatus(jobId: string) {
  return useQuery({
    queryKey: queryKeys.videos.job(jobId),
    queryFn: () => api.getJobStatus(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      // Refetch every second if job is processing
      return query.state.data?.status === "processing" ? 1000 : false;
    },
  });
}

export function useUploadVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => api.uploadVideo(file),
    onSuccess: () => {
      // Invalidate videos list to show the new upload
      queryClient.invalidateQueries({ queryKey: queryKeys.videos.list });
    },
  });
}

export function useStartProcessing() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (videoPath: string) => api.startProcessing(videoPath),
    onSuccess: () => {
      // Invalidate jobs to show the new job
      queryClient.invalidateQueries({ queryKey: queryKeys.videos.jobs });
    },
  });
}

export function useDeleteVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (filename: string) => api.deleteVideo(filename),
    onSuccess: () => {
      // Invalidate videos list
      queryClient.invalidateQueries({ queryKey: queryKeys.videos.list });
    },
  });
}

// ============ Combined Dashboard Query ============

export function useDashboardData() {
  const stats = useSummaryStats();
  const violations = useRecentViolations(5);
  const timeline = useViolationTimeline(7);
  const ppeBreakdown = useViolationsByPPE();

  return {
    stats: stats.data,
    violations: violations.data,
    timeline: timeline.data,
    ppeBreakdown: ppeBreakdown.data,
    isLoading:
      stats.isLoading ||
      violations.isLoading ||
      timeline.isLoading ||
      ppeBreakdown.isLoading,
    isRefetching:
      stats.isRefetching ||
      violations.isRefetching ||
      timeline.isRefetching ||
      ppeBreakdown.isRefetching,
    error: stats.error || violations.error || timeline.error || ppeBreakdown.error,
    refetch: () => {
      stats.refetch();
      violations.refetch();
      timeline.refetch();
      ppeBreakdown.refetch();
    },
  };
}
