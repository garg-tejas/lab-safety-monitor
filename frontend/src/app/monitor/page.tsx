"use client";

import { useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowUpRight, Camera, Pause, Play, Radio } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CardLoader } from "@/components/page-loader";
import { DemoBanner } from "@/components/demo-banner";
import { EventLedger } from "@/components/event-ledger";
import api from "@/lib/api";
import { useWebSocket, AlertMessage } from "@/providers/websocket-provider";
import { useDemoMode } from "@/providers/demo-context";
import { demoLiveViolations } from "@/lib/demo-data";

const VideoPlayer = dynamic(() => import("@/components/video-player").then((m) => m.VideoPlayer), { loading: () => <CardLoader />, ssr: false });

export default function MonitorPage() {
  const { isDemoMode } = useDemoMode();
  const { recentAlerts, isConnected } = useWebSocket();
  const source = isDemoMode ? "demo" : "live";
  const [feed, setFeed] = useState<{ source: string; enabled: boolean; startedAt: number }>({ source: "demo", enabled: false, startedAt: 0 });
  const feedEnabled = feed.enabled && feed.source === source;
  const [streamError, setStreamError] = useState<string | null>(null);
  const [feedReceived, setFeedReceived] = useState(false);
  const demoVideoRef = useRef<HTMLVideoElement>(null);

  const toggleFeed = () => {
    setStreamError(null);
    setFeedReceived(false);
    setFeed({ source, enabled: !feedEnabled, startedAt: Date.now() });
    if (feedEnabled) {
      demoVideoRef.current?.pause();
    }
  };

  const violations: AlertMessage[] = !feedEnabled ? [] : isDemoMode ? demoLiveViolations : recentAlerts.filter((alert) => alert.receivedAt >= feed.startedAt).slice(0, 5).map((alert) => alert.message);
  const status = !feedEnabled ? "Standby" : streamError ? "Feed unavailable" : isDemoMode ? "Prerecorded playback" : feedReceived ? "Camera feed received" : "Connecting to camera";

  return (
    <div>
      <DemoBanner />
      <header className="console-header">
        <div><span className="console-eyebrow">Observation / 02</span><h1>Keep the scene in view.</h1><p>Inspect a feed, follow reported safety events, and review recorded video in one workspace.</p></div>
        <div className="console-header-aside"><span className="console-source-tag"><i className="console-dot" />{isDemoMode ? "DEMO SOURCE" : "CAMERA SOURCE"}</span><Link href="/events" className="console-link">Event ledger <ArrowUpRight size={15} /></Link></div>
      </header>
      <div className="console-monitor-grid">
        <section className="console-panel" aria-label="Video observation">
          <div className="console-panel-heading"><div><h2>{isDemoMode ? "Recorded laboratory scene" : "Laboratory camera"}</h2><p>{isDemoMode ? "LOCAL VIDEO / DEMO_H264.MP4" : "BACKEND CAMERA STREAM"}</p></div><span className="console-source-tag">{status}</span></div>
          <div className="console-feed">
            {feedEnabled && !streamError ? <>
              {isDemoMode ? <video key={source} ref={demoVideoRef} autoPlay loop muted playsInline controls aria-label="Prerecorded laboratory demonstration" onLoadedData={() => setFeedReceived(true)} onError={() => setStreamError("The prerecorded video could not be loaded. No live camera is running.")}>
                <source src="/demo/demo_h264.mp4" type="video/mp4" /><source src="/demo/demo.webm" type="video/webm" />
              </video> : <img src={api.getLiveFeedUrl()} alt="Camera stream from the connected backend" onLoad={() => setFeedReceived(true)} onError={() => setStreamError("The camera stream is unavailable. Check the backend, camera access, and model readiness.")} />}
              <div className="console-feed-label"><i className="console-dot" />{isDemoMode ? "DEMONSTRATION / PRERECORDED" : "CAMERA STREAM"}</div>
            </> : <div className="console-feed-empty"><Camera size={38} strokeWidth={1} /><strong>{streamError ? "Feed unavailable" : "Ready when you are."}</strong><p>{streamError || (isDemoMode ? "Play the local recording to explore this console. Events are simulated and are not synchronized annotations of this clip." : "Start the camera stream when the backend and its models are ready. No feed is running yet.")}</p>{!feedEnabled && <Button onClick={toggleFeed} size="sm"><Play size={14} />{isDemoMode ? "Play demonstration" : "Start camera feed"}</Button>}</div>}
          </div>
          <div className="console-feed-controls"><p>{isDemoMode ? "DEMO / SIMULATED EVENTS / NO LIVE INFERENCE" : `ALERT CHANNEL / ${isConnected ? "CONNECTED" : "DISCONNECTED"}`}</p><Button onClick={toggleFeed} variant={feedEnabled ? "outline" : "default"} size="sm">{feedEnabled ? <Pause size={14} /> : <Play size={14} />}{feedEnabled ? "Stop feed" : isDemoMode ? "Play recording" : "Start feed"}</Button></div>
          {streamError && <p className="console-stream-error" role="alert">{streamError}</p>}
        </section>
        <EventLedger title="Event timeline" demonstration={isDemoMode} events={violations.map((event, index) => ({ id: `${event.timestamp}-${index}`, timestamp: event.timestamp, person: event.person_id || event.title, description: event.message, missingPpe: event.missing_ppe || [] }))} emptyMessage={isDemoMode ? "Start the demonstration to inspect sample events. These are simulated records, independent of video playback." : "No alerts received in this session. This does not establish that the scene is safe."} footer={isDemoMode ? "Sample events are independent of the recording timeline." : "Most recent five alerts received while this feed is enabled."} />
      </div>
      <div className="console-monitor-notes">
        <div><span>01 / VIDEO SOURCE</span><p>{isDemoMode ? "Local prerecorded footage" : "Backend camera stream"}</p></div>
        <div><span>02 / EVENT SOURCE</span><p>{isDemoMode ? "Simulated demonstration records" : isConnected ? "WebSocket connected" : "WebSocket disconnected"}</p></div>
        <div><span>03 / REVIEW</span><p>Confirm the scene before acting.</p></div>
      </div>
      <div className="console-section-heading"><div><h2>Recorded video workspace</h2><p>Upload, process, and inspect saved recordings. Demo mode keeps processing disabled.</p></div><Radio size={19} strokeWidth={1.4} className="text-muted-foreground" /></div>
      <VideoPlayer key={source} />
    </div>
  );
}
