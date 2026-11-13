"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { ArrowUpRight, Shield } from "lucide-react";
import { DemoBanner } from "@/components/demo-banner";
import { EventLedger } from "@/components/event-ledger";
import { PageLoader, CardLoader } from "@/components/page-loader";
import { LastUpdated } from "@/components/page-header";
import { useDashboardData } from "@/lib/queries";
import { useDemoMode } from "@/providers/demo-context";

const ViolationTimelineChart = dynamic(() => import("@/components/charts").then((m) => m.ViolationTimelineChart), { loading: () => <CardLoader />, ssr: false });
const PPEBreakdownChart = dynamic(() => import("@/components/charts").then((m) => m.PPEBreakdownChart), { loading: () => <CardLoader />, ssr: false });

export default function DashboardPage() {
  const { stats, violations, timeline, ppeBreakdown, isLoading, isRefetching, error, refetch } = useDashboardData();
  const { isDemoMode } = useDemoMode();
  if (isLoading) return <PageLoader />;
  const rate = stats && Number.isFinite(stats.compliance_rate) ? stats.compliance_rate : null;
  return (
    <div>
      <DemoBanner />
      <header className="console-header">
        <div><span className="console-eyebrow">Workspace / 01</span><h1>A clearer view of safety.</h1><p>Follow laboratory observations, review reported incidents, and move from a summary to the source.</p></div>
        <div className="console-header-aside"><Link href="/monitor" className="console-link">Open observation console <ArrowUpRight size={16} /></Link></div>
      </header>
      {error && <div role="alert" className="console-query-error">Some records could not be loaded. Missing values are unavailable.<button onClick={refetch} type="button">Retry connection</button></div>}
      <div className="console-overview-grid">
        <section className="console-panel console-summary" aria-label="Compliance summary">
          <div className="console-summary-top"><span className="console-eyebrow">Reported compliance</span><Shield size={21} strokeWidth={1.2} className="text-primary" /></div>
          <div><p className="console-compliance">{rate ?? "—"}{rate !== null && <small>%</small>}</p><p className="console-summary-subtitle">{isDemoMode ? "Illustrative rate from the demonstration dataset" : "Rate reported by the backend across recorded events"}</p><div className="console-summary-bar" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(100, rate ?? 0))}%` }} /></div></div>
          <dl className="console-stat-strip"><div><dt><strong>{stats?.today_violations ?? "—"}</strong></dt><dd><span>Violations today</span></dd></div><div><dt><strong>{stats?.today_events ?? "—"}</strong></dt><dd><span>Events today</span></dd></div><div><dt><strong>{stats?.total_persons ?? "—"}</strong></dt><dd><span>Tracked identities</span></dd></div></dl>
        </section>
        <EventLedger demonstration={isDemoMode} events={(violations || []).map((event) => ({ id: event.id, timestamp: event.timestamp, person: event.person_id || "Unidentified person", missingPpe: event.missing_ppe, ongoing: event.is_ongoing, description: event.video_source || undefined }))} emptyMessage={error ? "Event records are unavailable. Check the backend connection." : "No recent violation records were returned. This is a record view, not a live safety assessment."} footer="Showing up to five recent violation records. Ongoing is the state stored with the event." />
      </div>
      <div className="console-section-heading"><div><h2>Patterns in the record</h2><p>{isDemoMode ? "Illustrative distributions from simulated data." : "Reported incident counts, without inference-speed or accuracy claims."}</p></div><LastUpdated timestamp={stats?.last_updated} isRefreshing={isRefetching} /></div>
      <div className="console-charts"><ViolationTimelineChart data={timeline || []} /><PPEBreakdownChart data={ppeBreakdown || []} /></div>
      <p className="console-info-line">{isDemoMode ? "DEMONSTRATION / All summary values and events on this page are simulated." : "RECORDED EVENTS / A reachable backend does not establish camera or model readiness."} {stats ? `${stats.total_violations} total violations across ${stats.total_events} recorded events.` : "Summary unavailable."}</p>
    </div>
  );
}
