import type { Cloud, ReplayFrame, Summary } from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export const backendUrl = (path: string) => `${API_BASE}${path}`;

export const backendWsUrl = (path: string) => {
  const configured = (import.meta.env.VITE_WS_URL ?? "").replace(/\/$/, "");
  if (configured) return `${configured}${path}`;
  const origin = API_BASE || window.location.origin;
  return origin.replace(/^http/, "ws") + path;
};

export async function fetchSummary(): Promise<Summary> {
  const response = await fetch(backendUrl("/api/summary"));
  if (!response.ok) throw new Error("Could not load artifact summary");
  return response.json();
}

export async function fetchReplayFrame(frameId: number): Promise<ReplayFrame> {
  const response = await fetch(backendUrl(`/api/replay/frame/${frameId}`));
  if (!response.ok) throw new Error("Could not load replay frame");
  return response.json();
}

export async function fetchCloud(mode: string): Promise<Cloud> {
  const response = await fetch(backendUrl(`/api/pointcloud/${mode}`));
  if (!response.ok) throw new Error("Could not load point cloud");
  return response.json();
}

export async function uploadVideo(file: File): Promise<{ video_id: string; filename: string; frame_count: number; fps: number }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(backendUrl("/api/videos"), { method: "POST", body: form });
  if (!response.ok) throw new Error((await response.json()).detail ?? "Video upload failed");
  return response.json();
}
