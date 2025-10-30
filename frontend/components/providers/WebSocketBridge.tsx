"use client";

import { useEffect } from "react";
import { GMCSWebSocket } from "@/lib/websocket";

let socket: GMCSWebSocket | null = null;

export function WebSocketBridge() {
  useEffect(() => {
    if (!socket) {
      socket = new GMCSWebSocket();
    }
    socket.connect();

    return () => {
      socket?.disconnect();
    };
  }, []);

  return null;
}
