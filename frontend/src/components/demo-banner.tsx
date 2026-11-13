"use client";

import { Play } from "lucide-react";
import { useDemoMode } from "@/providers/demo-context";

export function DemoBanner() {
  const { isDemoMode, setDemoMode, isBackendAvailable, checkBackendStatus } = useDemoMode();
  if (!isDemoMode) return null;
  return (
    <div className="console-demo-banner" role="status">
      <Play size={14} aria-hidden="true" />
      <strong>DEMONSTRATION</strong>
      <span>Simulated event records · prerecorded video · no live inference</span>
      {isBackendAvailable ? <button type="button" onClick={async () => {
        if (await checkBackendStatus()) setDemoMode(false);
      }}>Connect to backend →</button> : <small>Backend unavailable</small>}
    </div>
  );
}

export function DemoModeIndicator() {
  const { isDemoMode } = useDemoMode();
  return isDemoMode ? <span className="console-source-tag">Demonstration</span> : null;
}
