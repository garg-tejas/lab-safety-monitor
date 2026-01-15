"use client";

import { useEffect, useState } from "react";
import { StatsCard } from "@/components/stats-card";
import { VideoPlayer } from "@/components/video-player";
import { EventsTable } from "@/components/events-table";
import { ViolationTimelineChart, PPEBreakdownChart } from "@/components/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Shield,
  AlertTriangle,
  Users,
  Activity,
  TrendingUp,
  Camera,
} from "lucide-react";
import api, {
  SummaryStats,
  ComplianceEvent,
  TimelineData,
  PPEBreakdown,
} from "@/lib/api";

export default function Dashboard() {
  const [stats, setStats] = useState<SummaryStats | null>(null);
  const [recentViolations, setRecentViolations] = useState<ComplianceEvent[]>([]);
  const [timeline, setTimeline] = useState<TimelineData[]>([]);
  const [ppeBreakdown, setPPEBreakdown] = useState<PPEBreakdown[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [statsData, violationsData, timelineData, ppeData] =
          await Promise.all([
            api.getSummaryStats(),
            api.getRecentViolations(5),
            api.getViolationTimeline(7),
            api.getViolationsByPPE(),
          ]);

        setStats(statsData);
        setRecentViolations(violationsData);
        setTimeline(timelineData);
        setPPEBreakdown(ppeData);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
        setError("Failed to connect to backend. Make sure the API is running.");
        // Set mock data for development
        setStats({
          total_events: 0,
          today_events: 0,
          total_violations: 0,
          today_violations: 0,
          total_persons: 0,
          compliance_rate: 100,
          last_updated: new Date().toISOString(),
        });
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                  MarketWise Lab Safety
                </h1>
                <p className="text-sm text-gray-500">
                  AI-Powered Compliance Monitoring
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Activity className="w-4 h-4" />
              <span>
                Last updated:{" "}
                {stats?.last_updated
                  ? new Date(stats.last_updated).toLocaleTimeString()
                  : "--"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-6">
        {/* Error banner */}
        {error && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg mb-6">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Stats cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatsCard
            title="Compliance Rate"
            value={`${stats?.compliance_rate ?? 0}%`}
            description="Overall safety compliance"
            icon={<TrendingUp className="w-5 h-5" />}
            loading={loading}
          />
          <StatsCard
            title="Today's Violations"
            value={stats?.today_violations ?? 0}
            description={`${stats?.today_events ?? 0} total events today`}
            icon={<AlertTriangle className="w-5 h-5" />}
            loading={loading}
          />
          <StatsCard
            title="Total Violations"
            value={stats?.total_violations ?? 0}
            description="All time"
            icon={<Shield className="w-5 h-5" />}
            loading={loading}
          />
          <StatsCard
            title="Tracked Persons"
            value={stats?.total_persons ?? 0}
            description="Unique individuals"
            icon={<Users className="w-5 h-5" />}
            loading={loading}
          />
        </div>

        {/* Main tabs */}
        <Tabs defaultValue="monitor" className="space-y-4">
          <TabsList>
            <TabsTrigger value="monitor" className="flex items-center gap-2">
              <Camera className="w-4 h-4" />
              Live Monitor
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="events" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Events Log
            </TabsTrigger>
          </TabsList>

          {/* Live Monitor Tab */}
          <TabsContent value="monitor" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <VideoPlayer />
              </div>
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      Recent Violations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {recentViolations.length > 0 ? (
                      <div className="space-y-3">
                        {recentViolations.slice(0, 5).map((event) => (
                          <div
                            key={event.id}
                            className="bg-red-50 border border-red-100 rounded-lg p-3"
                          >
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="font-medium text-sm">
                                  {event.person_id || "Unknown"}
                                </p>
                                <p className="text-xs text-red-600">
                                  Missing: {event.missing_ppe.join(", ")}
                                </p>
                              </div>
                              <span className="text-xs text-gray-500">
                                {new Date(event.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500 text-center py-4">
                        No recent violations
                      </p>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ViolationTimelineChart data={timeline} />
              <PPEBreakdownChart data={ppeBreakdown} />
            </div>
          </TabsContent>

          {/* Events Log Tab */}
          <TabsContent value="events">
            <Card>
              <CardHeader>
                <CardTitle>Compliance Events</CardTitle>
              </CardHeader>
              <CardContent>
                <EventsTable events={recentViolations} loading={loading} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white dark:bg-gray-800 mt-8">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-gray-500">
          MarketWise Lab Safety Monitoring System | Powered by SAM 3 & InsightFace
        </div>
      </footer>
    </div>
  );
}
