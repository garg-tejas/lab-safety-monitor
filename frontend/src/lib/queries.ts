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
import {
  demoStats,
  demoTimeline,
  demoPPEBreakdown,
  demoProcessingJobs,
  demoVideos,
  getDemoEvents,
  getDemoPersons,
  getDemoRecentViolations,
  getDemoTopViolators,
  getDemoPerson,
} from "@/lib/demo-data";
import { useDemoMode } from "@/providers/demo-context";

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
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.stats.summary, isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve(demoStats) : api.getSummaryStats(),
    refetchInterval: isDemoMode ? false : 30000,
  });
}

export function useViolationTimeline(days: number = 7) {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.stats.timeline(days), isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve(demoTimeline) : api.getViolationTimeline(days),
    staleTime: 60000,
  });
}

export function useViolationsByPPE() {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.stats.byPPE, isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve(demoPPEBreakdown) : api.getViolationsByPPE(),
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
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.events.list(params), isDemoMode],
    queryFn: () => {
      if (isDemoMode) {
        return Promise.resolve(getDemoEvents({
          page: params.page,
          page_size: params.pageSize,
          violations_only: params.violationsOnly,
        }));
      }
      return api.getEvents({
        page: params.page,
        page_size: params.pageSize,
        person_id: params.personId,
        violations_only: params.violationsOnly,
      });
    },
  });
}

export function useRecentViolations(limit: number = 10) {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.events.recentViolations(limit), isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve(getDemoRecentViolations(limit)) : api.getRecentViolations(limit),
    refetchInterval: isDemoMode ? false : 30000,
  });
}

// ============ Persons Queries ============

export function usePersons(page: number = 1, pageSize: number = 20) {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.persons.list(page, pageSize), isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve(getDemoPersons(page, pageSize)) : api.getPersons(page, pageSize),
  });
}

export function usePerson(personId: string) {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.persons.detail(personId), isDemoMode],
    queryFn: () => {
      if (isDemoMode) {
        const person = getDemoPerson(personId);
        if (!person) throw new Error("Person not found");
        return Promise.resolve(person);
      }
      return api.getPerson(personId);
    },
    enabled: !!personId,
  });
}

export function useTopViolators(limit: number = 5) {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.persons.topViolators(limit), isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve(getDemoTopViolators(limit)) : api.getTopViolators(limit),
  });
}

export function useUpdatePerson() {
  const queryClient = useQueryClient();
  const { isDemoMode } = useDemoMode();

  return useMutation({
    mutationFn: ({ personId, name }: { personId: string; name: string | null }) => {
      if (isDemoMode) {
        // In demo mode, just return a mock updated person
        const person = getDemoPerson(personId);
        if (!person) throw new Error("Person not found");
        return Promise.resolve({ ...person, name });
      }
      return api.updatePerson(personId, name);
    },
    onSuccess: (updatedPerson) => {
      queryClient.setQueryData(
        queryKeys.persons.detail(updatedPerson.id),
        updatedPerson
      );
      queryClient.invalidateQueries({ queryKey: queryKeys.persons.all });
    },
  });
}

// ============ Video Queries ============

export function useUploadedVideos() {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.videos.list, isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve({ videos: demoVideos }) : api.getUploadedVideos(),
  });
}

export function useProcessingJobs() {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.videos.jobs, isDemoMode],
    queryFn: () => isDemoMode ? Promise.resolve({ jobs: demoProcessingJobs }) : api.getProcessingJobs(),
    refetchInterval: (query) => {
      if (isDemoMode) return false;
      const hasActiveJobs = query.state.data?.jobs?.some((j) => j.status === "processing");
      return hasActiveJobs ? 1000 : false;
    },
  });
}

export function useJobStatus(jobId: string) {
  const { isDemoMode } = useDemoMode();

  return useQuery({
    queryKey: [...queryKeys.videos.job(jobId), isDemoMode],
    queryFn: () => {
      if (isDemoMode) {
        const job = demoProcessingJobs.find(j => j.id === jobId);
        if (!job) throw new Error("Job not found");
        return Promise.resolve(job);
      }
      return api.getJobStatus(jobId);
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      if (isDemoMode) return false;
      return query.state.data?.status === "processing" ? 1000 : false;
    },
  });
}

export function useUploadVideo() {
  const queryClient = useQueryClient();
  const { isDemoMode } = useDemoMode();

  return useMutation({
    mutationFn: (file: File) => {
      if (isDemoMode) {
        return Promise.resolve({ message: "Demo mode - upload simulated", filename: file.name, path: `/demo/${file.name}` });
      }
      return api.uploadVideo(file);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.videos.list });
    },
  });
}

export function useStartProcessing() {
  const queryClient = useQueryClient();
  const { isDemoMode } = useDemoMode();

  return useMutation({
    mutationFn: (videoPath: string) => {
      if (isDemoMode) {
        return Promise.resolve({ message: "Demo mode - processing simulated", job_id: "demo-job-new", video_path: videoPath });
      }
      return api.startProcessing(videoPath);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.videos.jobs });
    },
  });
}

export function useDeleteVideo() {
  const queryClient = useQueryClient();
  const { isDemoMode } = useDemoMode();

  return useMutation({
    mutationFn: (filename: string) => {
      if (isDemoMode) {
        return Promise.resolve({ message: "Demo mode - delete simulated" });
      }
      return api.deleteVideo(filename);
    },
    onSuccess: () => {
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
