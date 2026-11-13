"use client";

import React, { createContext, useContext, useEffect, useState, useRef } from "react";
import { useDemoMode } from "@/providers/demo-context";

// Define types for our WebSocket messages
export type AlertMessage = {
  type: "violation" | "system";
  title: string;
  message: string;
  timestamp: string;
  severity: "info" | "warning" | "error";
  // Violation-specific fields (from backend)
  event_id?: string;
  person_id?: string;
  missing_ppe?: string[];
};

type WebSocketContextType = {
  isConnected: boolean;
  lastMessage: AlertMessage | null;
  recentAlerts: { message: AlertMessage; receivedAt: number }[];
};

const WebSocketContext = createContext<WebSocketContextType>({
  isConnected: false,
  lastMessage: null,
  recentAlerts: [],
});

export function useWebSocket() {
  return useContext(WebSocketContext);
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const { isDemoMode } = useDemoMode();
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<AlertMessage | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<{ message: AlertMessage; receivedAt: number }[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (isDemoMode) return;
    // In a real app, URL should come from env vars
    // Using explicit loopback IP to avoid issues
    const wsUrl = "ws://127.0.0.1:8000/api/ws";
    
    // Check if running in browser
    if (typeof window === "undefined") return;

    try {
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket Connected");
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          if (data.type === "violation") {
            setRecentAlerts((previous) => [{ message: data, receivedAt: Date.now() }, ...previous].slice(0, 20));
          }
        } catch (e) {
          console.error("Failed to parse WS message", e);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket Disconnected");
        setIsConnected(false);
      };

      return () => {
        ws.close();
      };
    } catch (err) {
      console.error("WebSocket connection failed", err);
    }
  }, [isDemoMode]);

  return (
    <WebSocketContext.Provider value={{ isConnected: !isDemoMode && isConnected, lastMessage: isDemoMode ? null : lastMessage, recentAlerts: isDemoMode ? [] : recentAlerts }}>
      {children}
    </WebSocketContext.Provider>
  );
}
