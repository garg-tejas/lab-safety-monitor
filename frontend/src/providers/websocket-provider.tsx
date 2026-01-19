"use client";

import React, { createContext, useContext, useEffect, useState, useRef } from "react";
import { toast } from "sonner";

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
};

const WebSocketContext = createContext<WebSocketContextType>({
  isConnected: false,
  lastMessage: null,
});

export function useWebSocket() {
  return useContext(WebSocketContext);
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<AlertMessage | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
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

          // Show toast notification
          // if (data.type === "violation") {
          //   toast.error(data.title, {
          //     description: data.message,
          //     duration: 5000,
          //   });
          // }
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
  }, []);

  return (
    <WebSocketContext.Provider value={{ isConnected, lastMessage }}>
      {children}
    </WebSocketContext.Provider>
  );
}
