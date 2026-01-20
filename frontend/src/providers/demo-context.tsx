"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { STORAGE_KEYS, getStorageItem, setStorageItem } from "@/lib/storage";

interface DemoModeContextType {
    isDemoMode: boolean;
    toggleDemoMode: () => void;
    setDemoMode: (enabled: boolean) => void;
    isBackendAvailable: boolean;
    checkBackendStatus: () => Promise<boolean>;
}

const DemoModeContext = createContext<DemoModeContextType | undefined>(undefined);

// Add demo mode storage key
const DEMO_MODE_KEY = "sentinelvision_demo_mode_v1";

export function DemoModeProvider({ children }: { children: ReactNode }) {
    const [isDemoMode, setIsDemoMode] = useState(true); // Default to true
    const [isBackendAvailable, setIsBackendAvailable] = useState(true);
    const [mounted, setMounted] = useState(false);

    // Check backend availability
    const checkBackendStatus = useCallback(async (): Promise<boolean> => {
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/stats/summary`, {
                method: "GET",
                signal: AbortSignal.timeout(3000), // 3 second timeout
            });
            const available = response.ok;
            setIsBackendAvailable(available);
            return available;
        } catch {
            setIsBackendAvailable(false);
            return false;
        }
    }, []);

    // Initialize state from localStorage
    useEffect(() => {
        setMounted(true);
        const stored = getStorageItem(DEMO_MODE_KEY);

        // If explicitly disabled in storage, turn it off
        if (stored === "false") {
            setIsDemoMode(false);
        }
        // If explicitly enabled, or no preference (default), keep it on
        else if (stored === "true") {
            setIsDemoMode(true);
        }

        // Always check backend status
        checkBackendStatus().then((available) => {
            // If backend is not available, we force demo mode even if user disabled it previously
            if (!available) {
                setIsDemoMode(true);
                // We don't necessarily update storage here to avoid locking them in if backend comes back
                // But for consistency with previous logic, maybe we should?
                // Actually, if backend is down, we MUST be in demo mode.
            }
        });
    }, [checkBackendStatus]);

    const toggleDemoMode = useCallback(() => {
        setIsDemoMode((prev) => {
            const newValue = !prev;
            setStorageItem(DEMO_MODE_KEY, String(newValue));
            return newValue;
        });
    }, []);

    const setDemoModeValue = useCallback((enabled: boolean) => {
        setIsDemoMode(enabled);
        setStorageItem(DEMO_MODE_KEY, String(enabled));
    }, []);

    // Don't render until mounted to avoid hydration mismatch
    if (!mounted) {
        return <>{children}</>;
    }

    return (
        <DemoModeContext.Provider
            value={{
                isDemoMode,
                toggleDemoMode,
                setDemoMode: setDemoModeValue,
                isBackendAvailable,
                checkBackendStatus,
            }}
        >
            {children}
        </DemoModeContext.Provider>
    );
}

// Default values for when context is not available (SSR/static generation)
const defaultContextValue: DemoModeContextType = {
    isDemoMode: false,
    toggleDemoMode: () => { },
    setDemoMode: () => { },
    isBackendAvailable: true,
    checkBackendStatus: async () => true,
};

export function useDemoMode() {
    const context = useContext(DemoModeContext);
    // Return default values if context is not available (SSR/static generation)
    if (context === undefined) {
        return defaultContextValue;
    }
    return context;
}

