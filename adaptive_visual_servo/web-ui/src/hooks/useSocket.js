import { useEffect, useRef, useState, useCallback } from 'react';
import { io } from 'socket.io-client';

export function useSocket() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const s = io();  // same origin — Vite proxies /socket.io to Flask
    socketRef.current = s;
    s.on('connect', () => setConnected(true));
    s.on('disconnect', () => setConnected(false));
    return () => s.disconnect();
  }, []);

  const on = useCallback((event, handler) => {
    const s = socketRef.current;
    if (!s) return () => {};
    s.on(event, handler);
    return () => s.off(event, handler);
  }, []);

  return { socket: socketRef, connected, on };
}

export async function api(endpoint, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`/api/${endpoint}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
