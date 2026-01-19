"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PageHeader, LiveIndicator } from "@/components/page-header";
import { MotionWrapper, StaggerContainer, StaggerItem } from "@/components/motion-wrapper";
import { CardLoader } from "@/components/page-loader";
import {
  Camera,
  Power,
  PowerOff,
  AlertTriangle,
  Clock,
  Radio,
  Shield,
  Zap,
  Sparkles,
} from "lucide-react";
import api from "@/lib/api";
import { useWebSocket, AlertMessage } from "@/providers/websocket-provider";
import { toast } from "sonner";

// Dynamic import for VideoPlayer (heavy component with video processing)
const VideoPlayer = dynamic(
  () => import("@/components/video-player").then((m) => ({ default: m.VideoPlayer })),
  {
    loading: () => <CardLoader />,
    ssr: false,
  }
);

export default function MonitorPage() {
  const [liveFeedEnabled, setLiveFeedEnabled] = useState(false);
  const { lastMessage } = useWebSocket();
  const [liveViolations, setLiveViolations] = useState<AlertMessage[]>([]);

  // Listen for new violations from WebSocket
  useEffect(() => {
    if (lastMessage && lastMessage.type === "violation") {
      setLiveViolations((prev) => [lastMessage, ...prev].slice(0, 5));
      
      // Show toast notification for new violations
      toast.error(lastMessage.title, {
        description: lastMessage.message,
        icon: <AlertTriangle className="w-4 h-4" />,
        duration: 5000,
      });
    }
  }, [lastMessage]);

  const handleToggleFeed = () => {
    const newState = !liveFeedEnabled;
    setLiveFeedEnabled(newState);
    
    if (newState) {
      toast.success("Live Feed Started", {
        description: "Real-time monitoring is now active",
        icon: <Camera className="w-4 h-4" />,
      });
    } else {
      toast.info("Live Feed Stopped", {
        description: "Real-time monitoring has been paused",
      });
    }
  };

  return (
    <MotionWrapper className="space-y-8">
      {/* Header */}
      <PageHeader
        title="Live Monitor"
        description="Real-time safety monitoring and video processing"
        action={
          <AnimatePresence>
            {liveFeedEnabled && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <LiveIndicator label="Live Stream Active" />
              </motion.div>
            )}
          </AnimatePresence>
        }
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-6"
        >
          {/* Live Feed Card */}
          <Card variant="glass" hover glow={liveFeedEnabled ? "primary" : null}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-3">
                  <motion.div
                    animate={liveFeedEnabled ? { scale: [1, 1.1, 1] } : {}}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10"
                  >
                    <Camera className="w-5 h-5 text-primary" />
                  </motion.div>
                  <div>
                    <span>Live Webcam Feed</span>
                    <p className="text-xs font-normal text-muted-foreground mt-0.5">
                      Real-time detection and analysis
                    </p>
                  </div>
                </CardTitle>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button
                    variant={liveFeedEnabled ? "destructive" : "default"}
                    size="sm"
                    onClick={handleToggleFeed}
                    className="flex items-center gap-2 shadow-lg"
                  >
                    {liveFeedEnabled ? (
                      <>
                        <PowerOff className="w-4 h-4" />
                        Stop Feed
                      </>
                    ) : (
                      <>
                        <Power className="w-4 h-4" />
                        Start Feed
                      </>
                    )}
                  </Button>
                </motion.div>
              </div>
            </CardHeader>
            <CardContent>
              <AnimatePresence mode="wait">
                {liveFeedEnabled ? (
                  <motion.div
                    key="live"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="relative aspect-video rounded-xl overflow-hidden border border-border/50 shadow-2xl"
                  >
                    {/* Glow effect behind video */}
                    <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-danger/20 blur-3xl opacity-50" />
                    <img
                      src={api.getLiveFeedUrl()}
                      alt="Live webcam feed with real-time detection"
                      className="relative w-full h-full object-contain bg-black"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src =
                          "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='480'%3E%3Crect fill='%231a1a1a' width='640' height='480'/%3E%3Ctext fill='%23666' x='50%25' y='50%25' text-anchor='middle' dy='.3em' font-family='Arial' font-size='16'%3EWebcam not available%3C/text%3E%3C/svg%3E";
                      }}
                    />
                    {/* Live indicator overlay */}
                    <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 bg-black/60 text-white rounded-full text-xs backdrop-blur-md border border-white/10">
                      <span className="relative flex h-2.5 w-2.5">
                        <motion.span
                          animate={{ scale: [1, 1.5, 1], opacity: [0.75, 0, 0.75] }}
                          transition={{ duration: 1, repeat: Infinity }}
                          className="absolute inline-flex h-full w-full rounded-full bg-red-400"
                        />
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
                      </span>
                      LIVE
                    </div>
                    {/* Processing indicator */}
                    <div className="absolute bottom-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-black/60 text-white rounded-full text-xs backdrop-blur-md border border-white/10">
                      <Zap className="w-3 h-3 text-primary" />
                      AI Processing
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="relative aspect-video rounded-xl overflow-hidden bg-muted/30 border-2 border-dashed border-border flex items-center justify-center"
                  >
                    <div className="text-center">
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="flex items-center justify-center w-16 h-16 rounded-2xl bg-muted mx-auto mb-4"
                      >
                        <Camera className="w-8 h-8 text-muted-foreground opacity-50" />
                      </motion.div>
                      <p className="font-medium text-foreground">
                        Live feed is disabled
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Click &quot;Start Feed&quot; to enable real-time monitoring
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>

          {/* Recent Live Violations Card */}
          <Card variant="glass" hover glow={liveViolations.length > 0 ? "danger" : null}>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div className="relative">
                  <motion.div
                    animate={liveViolations.length > 0 ? { scale: [1, 1.1, 1] } : {}}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    className="flex items-center justify-center w-10 h-10 rounded-xl bg-danger/10"
                  >
                    <AlertTriangle className="w-5 h-5 text-danger" />
                  </motion.div>
                  <AnimatePresence>
                    {liveViolations.length > 0 && (
                      <motion.span
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                        className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-danger text-[10px] font-bold text-white"
                      >
                        {liveViolations.length}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>
                <div>
                  <span>Live Violations Feed</span>
                  <p className="text-xs font-normal text-muted-foreground mt-0.5">
                    Real-time safety alerts via WebSocket
                  </p>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AnimatePresence mode="popLayout">
                {liveViolations.length > 0 ? (
                  <StaggerContainer className="space-y-3">
                    {liveViolations.map((violation, index) => (
                      <StaggerItem key={`${violation.timestamp}-${index}`}>
                        <motion.div
                          layout
                          initial={{ opacity: 0, x: -20, height: 0 }}
                          animate={{ opacity: 1, x: 0, height: "auto" }}
                          exit={{ opacity: 0, x: 20, height: 0 }}
                          whileHover={{ x: 4 }}
                          className="flex items-start gap-4 p-4 rounded-xl bg-gradient-to-r from-danger/10 to-transparent border border-danger/20 hover:border-danger/30 transition-all duration-200"
                        >
                          <motion.div
                            animate={{ rotate: [0, -5, 5, 0] }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            className="p-2.5 bg-danger/10 rounded-xl shrink-0"
                          >
                            <AlertTriangle className="w-4 h-4 text-danger" />
                          </motion.div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium text-sm text-foreground">
                                {violation.title}
                              </h4>
                              <Badge variant="danger" className="text-[10px]">
                                {violation.severity}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">
                              {violation.message}
                            </p>
                            <div className="flex flex-wrap gap-1.5 mt-2">
                              {violation.missing_ppe?.map((ppe) => (
                                <Badge
                                  key={ppe}
                                  variant="danger-soft"
                                  className="text-[10px]"
                                >
                                  {ppe}
                                </Badge>
                              ))}
                            </div>
                            <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              <span>
                                {new Date(violation.timestamp).toLocaleTimeString()}
                              </span>
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
                    className="flex flex-col items-center justify-center py-16 text-center border-2 border-dashed border-border rounded-xl bg-muted/10"
                  >
                    <motion.div
                      animate={{ y: [0, -5, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="flex items-center justify-center w-16 h-16 rounded-2xl bg-success/10 mb-4"
                    >
                      <Shield className="w-8 h-8 text-success" />
                    </motion.div>
                    <div className="flex items-center gap-2 mb-1">
                      <Sparkles className="w-4 h-4 text-success" />
                      <p className="font-medium text-foreground">No active violations</p>
                      <Sparkles className="w-4 h-4 text-success" />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Real-time safety alerts will appear here
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-6"
        >
          <VideoPlayer />
        </motion.div>
      </div>
    </MotionWrapper>
  );
}
