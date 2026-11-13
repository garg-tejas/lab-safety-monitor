"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { getStorageItem, setStorageItem } from "@/lib/storage";

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
    const [isBackendAvailable, setIsBackendAvailable] = useState(false);

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
        const stored = getStorageItem(DEMO_MODE_KEY);

        let cancelled = false;
        // Remain in demo mode until the saved live preference is confirmed reachable.
        // This subscribes to an asynchronous backend check; state changes in its response callback.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        checkBackendStatus().then((available) => {
            if (!cancelled) setIsDemoMode(!(available && stored === "false"));
        });
        return () => { cancelled = true; };
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
    isDemoMode: true,
    toggleDemoMode: () => { },
    setDemoMode: () => { },
    isBackendAvailable: false,
    checkBackendStatus: async () => false,
};

export function useDemoMode() {
    const context = useContext(DemoModeContext);
    // Return default values if context is not available (SSR/static generation)
    if (context === undefined) {
        return defaultContextValue;
    }
    return context;
}

