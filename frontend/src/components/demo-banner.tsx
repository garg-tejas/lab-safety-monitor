"use client";

import { motion } from "framer-motion";
import { Play, X, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useDemoMode } from "@/providers/demo-context";

export function DemoBanner() {
    const { isDemoMode, setDemoMode, isBackendAvailable, checkBackendStatus } = useDemoMode();

    if (!isDemoMode) return null;

    const handleTryLiveMode = async () => {
        const available = await checkBackendStatus();
        if (available) {
            setDemoMode(false);
        } else {
            // Could show a toast here indicating backend is unavailable
            console.log("Backend is still unavailable");
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="relative overflow-hidden corner-cut bg-gradient-to-r from-info/20 via-info/10 to-primary/20 border-2 border-info/50 px-4 py-3 mb-4"
        >
            {/* Animated background pattern */}
            <div className="absolute inset-0 opacity-10">
                <div
                    className="absolute inset-0"
                    style={{
                        backgroundImage: `repeating-linear-gradient(
              45deg,
              transparent,
              transparent 10px,
              currentColor 10px,
              currentColor 11px
            )`,
                    }}
                />
            </div>

            <div className="relative flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <motion.div
                        animate={{ rotate: [0, 360] }}
                        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                        className="flex items-center justify-center w-10 h-10 corner-cut bg-info/20 border border-info"
                    >
                        <Play className="w-5 h-5 text-info" />
                    </motion.div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h3 className="text-sm font-black uppercase tracking-wider text-info">
                                Demo Mode Active
                            </h3>
                            <motion.div
                                animate={{ opacity: [1, 0.5, 1] }}
                                transition={{ duration: 1.5, repeat: Infinity }}
                                className="px-2 py-0.5 text-[10px] font-bold uppercase bg-info/20 text-info rounded corner-cut border border-info/30"
                            >
                                Simulated Data
                            </motion.div>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            You&apos;re viewing pre-recorded demo data. {!isBackendAvailable && "Backend server is unavailable."}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {isBackendAvailable && (
                        <Button
                            size="sm"
                            variant="outline"
                            onClick={handleTryLiveMode}
                            className="text-xs h-8 corner-cut border-info/50 hover:bg-info/10 hover:border-info"
                        >
                            Switch to Live
                        </Button>
                    )}
                    <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => setDemoMode(false)}
                        className="h-8 w-8 text-muted-foreground hover:text-foreground"
                    >
                        <X className="w-4 h-4" />
                    </Button>
                </div>
            </div>
        </motion.div>
    );
}

// Compact version for sidebar
export function DemoModeIndicator() {
    const { isDemoMode } = useDemoMode();

    if (!isDemoMode) return null;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-2 px-3 py-2 corner-cut bg-info/10 border border-info/30"
        >
            <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="w-2 h-2 bg-info rounded-full"
            />
            <span className="text-[10px] font-black text-info uppercase tracking-widest font-mono">
                Demo Mode
            </span>
        </motion.div>
    );
}
