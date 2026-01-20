/**
 * Demo data for the frontend when running in demo mode.
 * This provides realistic mock data for all API endpoints.
 */

import type { SummaryStats, ComplianceEvent, Person, TimelineData, PPEBreakdown, ProcessingJob, VideoFile } from "./api";

// Generate timestamps relative to current time
const now = new Date();
const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);

function hoursAgo(hours: number): string {
    return new Date(now.getTime() - hours * 60 * 60 * 1000).toISOString();
}

function daysAgo(days: number): string {
    return new Date(now.getTime() - days * 24 * 60 * 60 * 1000).toISOString();
}

// Demo summary statistics
export const demoStats: SummaryStats = {
    total_events: 247,
    today_events: 23,
    total_violations: 89,
    today_violations: 7,
    total_persons: 12,
    compliance_rate: 78.4,
    last_updated: now.toISOString(),
};

// Demo compliance events (violations)
export const demoEvents: ComplianceEvent[] = [
    {
        id: "demo-event-001",
        person_id: "person-001",
        timestamp: hoursAgo(0.5),
        video_source: "Lab Camera 1",
        frame_number: 1245,
        detected_ppe: ["lab_coat", "gloves"],
        missing_ppe: ["safety_goggles", "face_mask"],
        action_violations: [],
        is_violation: true,
        start_frame: 1200,
        end_frame: 1350,
        duration_frames: 150,
        is_ongoing: true,
    },
    {
        id: "demo-event-002",
        person_id: "person-003",
        timestamp: hoursAgo(1.2),
        video_source: "Lab Camera 1",
        frame_number: 892,
        detected_ppe: ["safety_goggles", "face_mask"],
        missing_ppe: ["lab_coat"],
        action_violations: ["drinking"],
        is_violation: true,
        start_frame: 850,
        end_frame: 920,
        duration_frames: 70,
        is_ongoing: false,
    },
    {
        id: "demo-event-003",
        person_id: "person-002",
        timestamp: hoursAgo(2.5),
        video_source: "Lab Camera 2",
        frame_number: 3421,
        detected_ppe: ["lab_coat"],
        missing_ppe: ["safety_goggles", "face_mask", "gloves"],
        action_violations: [],
        is_violation: true,
        start_frame: 3400,
        end_frame: 3500,
        duration_frames: 100,
        is_ongoing: false,
    },
    {
        id: "demo-event-004",
        person_id: "person-005",
        timestamp: hoursAgo(4.1),
        video_source: "Lab Camera 1",
        frame_number: 5632,
        detected_ppe: ["face_mask", "gloves"],
        missing_ppe: ["safety_goggles", "lab_coat"],
        action_violations: [],
        is_violation: true,
        start_frame: 5600,
        end_frame: 5700,
        duration_frames: 100,
        is_ongoing: false,
    },
    {
        id: "demo-event-005",
        person_id: "person-001",
        timestamp: hoursAgo(6.3),
        video_source: "Lab Camera 2",
        frame_number: 7845,
        detected_ppe: ["safety_goggles", "lab_coat", "gloves"],
        missing_ppe: ["face_mask"],
        action_violations: ["eating"],
        is_violation: true,
        start_frame: 7800,
        end_frame: 7900,
        duration_frames: 100,
        is_ongoing: false,
    },
    {
        id: "demo-event-006",
        person_id: "person-004",
        timestamp: hoursAgo(8.7),
        video_source: "Lab Camera 1",
        frame_number: 9234,
        detected_ppe: ["face_mask"],
        missing_ppe: ["safety_goggles", "lab_coat", "gloves"],
        action_violations: [],
        is_violation: true,
        start_frame: 9200,
        end_frame: 9300,
        duration_frames: 100,
        is_ongoing: false,
    },
    {
        id: "demo-event-007",
        person_id: "person-002",
        timestamp: hoursAgo(12.5),
        video_source: "Lab Camera 2",
        frame_number: 11567,
        detected_ppe: ["safety_goggles", "gloves"],
        missing_ppe: ["face_mask", "lab_coat"],
        action_violations: [],
        is_violation: true,
        start_frame: 11500,
        end_frame: 11650,
        duration_frames: 150,
        is_ongoing: false,
    },
];

// Demo persons
export const demoPersons: Person[] = [
    {
        id: "person-001",
        name: "Dr. Sarah Chen",
        first_seen: daysAgo(14),
        last_seen: hoursAgo(0.5),
        total_events: 45,
        violation_count: 8,
        compliance_rate: 82.2,
    },
    {
        id: "person-002",
        name: "Michael Rodriguez",
        first_seen: daysAgo(12),
        last_seen: hoursAgo(2.5),
        total_events: 38,
        violation_count: 12,
        compliance_rate: 68.4,
    },
    {
        id: "person-003",
        name: "Dr. Emily Watson",
        first_seen: daysAgo(10),
        last_seen: hoursAgo(1.2),
        total_events: 52,
        violation_count: 5,
        compliance_rate: 90.4,
    },
    {
        id: "person-004",
        name: "James Liu",
        first_seen: daysAgo(8),
        last_seen: hoursAgo(8.7),
        total_events: 29,
        violation_count: 15,
        compliance_rate: 48.3,
    },
    {
        id: "person-005",
        name: "Anna Kowalski",
        first_seen: daysAgo(6),
        last_seen: hoursAgo(4.1),
        total_events: 34,
        violation_count: 7,
        compliance_rate: 79.4,
    },
    {
        id: "person-006",
        name: null,
        first_seen: daysAgo(5),
        last_seen: daysAgo(1),
        total_events: 18,
        violation_count: 3,
        compliance_rate: 83.3,
    },
    {
        id: "person-007",
        name: "Dr. Robert Kim",
        first_seen: daysAgo(4),
        last_seen: daysAgo(0.5),
        total_events: 21,
        violation_count: 4,
        compliance_rate: 81.0,
    },
    {
        id: "person-008",
        name: "Lisa Thompson",
        first_seen: daysAgo(3),
        last_seen: daysAgo(0.2),
        total_events: 15,
        violation_count: 2,
        compliance_rate: 86.7,
    },
];

// Demo timeline data (last 7 days)
export const demoTimeline: TimelineData[] = [
    { date: daysAgo(6).split("T")[0], violations: 8 },
    { date: daysAgo(5).split("T")[0], violations: 12 },
    { date: daysAgo(4).split("T")[0], violations: 6 },
    { date: daysAgo(3).split("T")[0], violations: 15 },
    { date: daysAgo(2).split("T")[0], violations: 9 },
    { date: daysAgo(1).split("T")[0], violations: 11 },
    { date: today.toISOString().split("T")[0], violations: 7 },
];

// Demo PPE breakdown
export const demoPPEBreakdown: PPEBreakdown[] = [
    { ppe_type: "safety_goggles", count: 34 },
    { ppe_type: "face_mask", count: 28 },
    { ppe_type: "lab_coat", count: 19 },
    { ppe_type: "gloves", count: 8 },
];

// Demo processing jobs
export const demoProcessingJobs: ProcessingJob[] = [
    {
        id: "demo-job-001",
        video_path: "/demo/demo_h264.mp4",
        filename: "lab_safety_demo.mp4",
        status: "completed",
        progress: 100,
        total_frames: 1800,
        processed_frames: 1800,
        violations_count: 12,
        persons_count: 4,
        unique_events: 8,
        error: null,
        output_video: "/demo/demo_h264.mp4",
    },
];

// Demo uploaded videos
export const demoVideos: VideoFile[] = [
    {
        filename: "lab_safety_demo.mp4",
        path: "/demo/demo_h264.mp4",
        size_mb: 24.5,
    },
];

// Demo live violations for WebSocket simulation
export const demoLiveViolations = [
    {
        type: "violation" as const,
        title: "PPE Violation Detected",
        message: "Person missing safety goggles and face mask in Lab Zone A",
        severity: "error" as const,
        timestamp: hoursAgo(0.1),
        missing_ppe: ["safety_goggles", "face_mask"],
    },
    {
        type: "violation" as const,
        title: "PPE Violation Detected",
        message: "Person missing lab coat in Lab Zone B",
        severity: "warning" as const,
        timestamp: hoursAgo(0.3),
        missing_ppe: ["lab_coat"],
    },
];

// Helper to get paginated events
export function getDemoEvents(params: {
    page?: number;
    page_size?: number;
    violations_only?: boolean;
}): { events: ComplianceEvent[]; total: number; page: number; page_size: number } {
    const page = params.page || 1;
    const pageSize = params.page_size || 20;
    const events = params.violations_only
        ? demoEvents.filter(e => e.is_violation)
        : demoEvents;

    const start = (page - 1) * pageSize;
    const end = start + pageSize;

    return {
        events: events.slice(start, end),
        total: events.length,
        page,
        page_size: pageSize,
    };
}

// Helper to get paginated persons
export function getDemoPersons(page: number = 1, pageSize: number = 20): { persons: Person[]; total: number } {
    const start = (page - 1) * pageSize;
    const end = start + pageSize;

    return {
        persons: demoPersons.slice(start, end),
        total: demoPersons.length,
    };
}

// Helper to get recent violations
export function getDemoRecentViolations(limit: number = 10): ComplianceEvent[] {
    return demoEvents.filter(e => e.is_violation).slice(0, limit);
}

// Helper to get top violators
export function getDemoTopViolators(limit: number = 5): Person[] {
    return [...demoPersons]
        .sort((a, b) => b.violation_count - a.violation_count)
        .slice(0, limit);
}

// Helper to get person by ID
export function getDemoPerson(personId: string): Person | undefined {
    return demoPersons.find(p => p.id === personId);
}
