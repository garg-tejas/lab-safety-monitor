"use client";

import { Suspense } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { StatsCard } from "@/components/stats-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader, LastUpdated } from "@/components/page-header";
import { MotionWrapper, StaggerContainer, StaggerItem } from "@/components/motion-wrapper";
import { PageLoader, CardLoader } from "@/components/page-loader";
import { DemoBanner } from "@/components/demo-banner";
import { useDashboardData } from "@/lib/queries";
import { useDemoMode } from "@/providers/demo-context";
import {
  Shield,
  AlertTriangle,
  Users,
  TrendingUp,
  Clock,
  CheckCircle2,
  Sparkles,
  Radio,
  Activity,
} from "lucide-react";

// Dynamic imports for heavy chart components
const ViolationTimelineChart = dynamic(
  () => import("@/components/charts").then((m) => ({ default: m.ViolationTimelineChart })),
  {
    loading: () => <CardLoader />,
    ssr: false
  }
);

const PPEBreakdownChart = dynamic(
  () => import("@/components/charts").then((m) => ({ default: m.PPEBreakdownChart })),
  {
    loading: () => <CardLoader />,
    ssr: false
  }
);

export default function DashboardPage() {
  const { stats, violations, timeline, ppeBreakdown, isLoading, isRefetching } = useDashboardData();
  const { isDemoMode } = useDemoMode();

  if (isLoading) {
    return <PageLoader />;
  }

  return (
    <MotionWrapper className="space-y-6">
      {/* Demo Mode Banner */}
      <DemoBanner />

      {/* Industrial Header with System Status */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
      >
        {/* Background tech grid */}
        <div className="absolute inset-0 grid-overlay opacity-20 pointer-events-none" />

        <div className="relative corner-cut bg-gradient-to-r from-card via-card to-card/50 border-2 border-primary/30 p-6 scan-lines">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  className="w-12 h-12 corner-cut bg-primary/20 border-2 border-primary flex items-center justify-center glow-primary"
                >
                  <Shield className="w-6 h-6 text-primary" />
                </motion.div>
                <div>
                  <h1 className="text-3xl font-black neon-text tracking-wider">
                    SAFETY MONITOR
                  </h1>
                  <p className="text-sm text-muted-foreground font-mono uppercase tracking-wide">
                    Real-time Laboratory Compliance System
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* System Status */}
              <div className="corner-cut bg-success/10 border-2 border-success/50 px-4 py-2 flex items-center gap-2">
                <div className="relative">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="w-2 h-2 bg-success rounded-full"
                  />
                  <motion.div
                    animate={{ scale: [1, 2.5, 1], opacity: [0.5, 0, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="absolute inset-0 w-2 h-2 bg-success rounded-full"
                  />
                </div>
                <span className="text-xs font-bold text-success uppercase tracking-wider font-mono">
                  SYSTEM ONLINE
                </span>
              </div>

              <LastUpdated
                timestamp={stats?.last_updated}
                isRefreshing={isRefetching}
              />
            </div>
          </div>
        </div>
      </motion.div>

      {/* Stats Grid - Industrial Panels */}
      <Suspense fallback={
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <CardLoader key={i} />
          ))}
        </div>
      }>
        <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StaggerItem>
            <StatsCard
              title="COMPLIANCE RATE"
              value={`${stats?.compliance_rate ?? 0}%`}
              description="Overall safety compliance"
              icon={<TrendingUp className="w-5 h-5" />}
              variant="success"
            />
          </StaggerItem>
          <StaggerItem>
            <StatsCard
              title="TODAY'S VIOLATIONS"
              value={stats?.today_violations ?? 0}
              description={`${stats?.today_events ?? 0} total events today`}
              icon={<AlertTriangle className="w-5 h-5" />}
              variant="danger"
            />
          </StaggerItem>
          <StaggerItem>
            <StatsCard
              title="TOTAL VIOLATIONS"
              value={stats?.total_violations ?? 0}
              description="All time incidents"
              icon={<Shield className="w-5 h-5" />}
              variant="warning"
            />
          </StaggerItem>
          <StaggerItem>
            <StatsCard
              title="TRACKED PERSONS"
              value={stats?.total_persons ?? 0}
              description="Unique individuals"
              icon={<Users className="w-5 h-5" />}
              variant="info"
            />
          </StaggerItem>
        </StaggerContainer>
      </Suspense>

      {/* Charts - Data Visualization Panels */}
      <Suspense fallback={
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <CardLoader />
          <CardLoader />
        </div>
      }>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <ViolationTimelineChart data={timeline ?? []} />
          <PPEBreakdownChart data={ppeBreakdown ?? []} />
        </motion.div>
      </Suspense>

      {/* Recent Violations - Alert Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card variant="glass" className="border-2 border-danger/30 corner-cut scan-lines overflow-hidden">
          <CardHeader className="border-b-2 border-danger/20 bg-gradient-to-r from-danger/10 via-danger/5 to-transparent">
            <CardTitle className="flex items-center gap-3">
              <div className="relative">
                <motion.div
                  animate={{ rotate: [0, 5, -5, 0] }}
                  transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 2 }}
                  className="flex items-center justify-center w-12 h-12 corner-cut bg-danger/20 border-2 border-danger"
                >
                  <AlertTriangle className="w-6 h-6 text-danger" />
                </motion.div>
                {violations && violations.length > 0 && (
                  <motion.span
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute -top-2 -right-2 flex h-6 w-6 items-center justify-center corner-cut bg-danger text-[11px] font-black text-danger-foreground border border-danger-foreground pulse-danger"
                  >
                    {violations.length}
                  </motion.span>
                )}
              </div>
              <div>
                <span className="text-xl font-black uppercase tracking-wider neon-text">
                  ACTIVE ALERTS
                </span>
                <p className="text-xs font-normal text-muted-foreground mt-0.5 font-mono uppercase">
                  Safety incidents requiring immediate attention
                </p>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            {violations && violations.length > 0 ? (
              <StaggerContainer className="space-y-3">
                {violations.map((event) => (
                  <StaggerItem key={event.id}>
                    <motion.div
                      whileHover={{ x: 4, scale: 1.01 }}
                      transition={{ duration: 0.15 }}
                      className="relative corner-cut bg-gradient-to-r from-danger/10 via-danger/5 to-transparent border-l-4 border-danger p-4 group hover:glow-danger transition-all"
                    >
                      {/* Diagonal stripes background on hover */}
                      <div className="absolute inset-0 diagonal-stripes opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

                      <div className="relative flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="relative">
                            <motion.div
                              whileHover={{ rotate: [0, -15, 15, 0] }}
                              transition={{ duration: 0.3 }}
                              className="p-3 bg-danger/20 corner-cut border border-danger group-hover:bg-danger/30 transition-colors"
                            >
                              <AlertTriangle className="w-5 h-5 text-danger" />
                            </motion.div>
                            {event.is_ongoing && (
                              <div className="absolute -top-1 -right-1">
                                <span className="status-dot bg-danger" />
                              </div>
                            )}
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-1">
                              <p className="font-bold text-base uppercase tracking-wide font-mono">
                                {event.person_id || "UNKNOWN PERSON"}
                              </p>
                              {event.is_ongoing ? (
                                <Badge variant="danger-glow" className="text-[10px] font-black uppercase">
                                  ● ONGOING
                                </Badge>
                              ) : (
                                <Badge variant="secondary" className="text-[10px] font-black uppercase">
                                  RESOLVED
                                </Badge>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              {event.missing_ppe.slice(0, 3).map((ppe) => (
                                <Badge
                                  key={ppe}
                                  variant="danger-soft"
                                  className="text-[10px] font-bold uppercase tracking-wide corner-cut"
                                >
                                  {ppe}
                                </Badge>
                              ))}
                              {event.missing_ppe.length > 3 && (
                                <span className="text-xs text-muted-foreground font-mono">
                                  +{event.missing_ppe.length - 3} MORE
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold flex items-center gap-1.5 justify-end font-mono">
                            <Clock className="w-4 h-4 text-muted-foreground" />
                            {new Date(event.timestamp).toLocaleTimeString()}
                          </p>
                          <p className="text-xs text-muted-foreground mt-0.5 font-mono">
                            {new Date(event.timestamp).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  </StaggerItem>
                ))}
              </StaggerContainer>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="flex flex-col items-center justify-center py-16 text-center"
              >
                <motion.div
                  animate={{ y: [0, -8, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                  className="flex items-center justify-center w-24 h-24 corner-cut bg-success/10 border-2 border-success mb-6 glow-success"
                >
                  <CheckCircle2 className="w-12 h-12 text-success" />
                </motion.div>
                <div className="flex items-center gap-3 mb-2">
                  <motion.div
                    animate={{ rotate: [0, 180, 360] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-6 h-6 text-success" />
                  </motion.div>
                  <p className="text-2xl font-black text-success uppercase tracking-wider neon-text">
                    ALL SYSTEMS CLEAR
                  </p>
                  <motion.div
                    animate={{ rotate: [360, 180, 0] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-6 h-6 text-success" />
                  </motion.div>
                </div>
                <p className="text-sm text-muted-foreground font-mono uppercase tracking-wide">
                  No safety violations detected
                </p>
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </MotionWrapper>
  );
}
