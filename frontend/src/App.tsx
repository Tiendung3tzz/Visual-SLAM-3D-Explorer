import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { backendUrl, backendWsUrl, fetchCloud, fetchReplayFrame, fetchSummary, uploadVideo } from "./api";
import type { Cloud, LiveFrame, MapMode, ReplayFrame, RunMode, Summary } from "./types";

const emptyCloud: Cloud = { points: [], colors: [], mode: "", label: "", available: false };

const mapLabels: Record<MapMode, string> = {
  sparse: "Sparse Map",
  dense: "Dense Point Cloud · single frame",
  sensor: "Sensor Depth · single frame",
};

function formatMetric(value: unknown, suffix = "") {
  if (value === undefined || value === null || value === "") return "--";
  return `${value}${suffix}`;
}

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [replayFrame, setReplayFrame] = useState<ReplayFrame | null>(null);
  const [liveFrame, setLiveFrame] = useState<LiveFrame | null>(null);
  const [cloud, setCloud] = useState<Cloud>(emptyCloud);
  const [mapMode, setMapMode] = useState<MapMode>("sparse");
  const [runMode, setRunMode] = useState<RunMode>("replay");
  const [frameId, setFrameId] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showFeatures, setShowFeatures] = useState(true);
  const [error, setError] = useState("");
  const [uploadName, setUploadName] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const playTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    fetchSummary().then(setSummary).catch((reason) => setError(String(reason)));
    return () => {
      socketRef.current?.close();
      if (playTimer.current) window.clearInterval(playTimer.current);
    };
  }, []);

  useEffect(() => {
    if (runMode !== "replay") return;
    Promise.all([fetchReplayFrame(frameId), fetchCloud(mapMode)])
      .then(([nextFrame, nextCloud]) => {
        setReplayFrame(nextFrame);
        setCloud(nextCloud);
      })
      .catch((reason) => setError(String(reason)));
  }, [frameId, mapMode, runMode]);

  useEffect(() => {
    if (!playing || runMode !== "replay" || !summary) return;
    playTimer.current = window.setInterval(() => {
      setFrameId((current) => {
        if (current >= summary.frame_count - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 80);
    return () => {
      if (playTimer.current) window.clearInterval(playTimer.current);
    };
  }, [playing, runMode, summary]);

  const startLive = (videoId: string) => {
    socketRef.current?.close();
    const socket = new WebSocket(backendWsUrl(`/ws/live/${videoId}`));
    socketRef.current = socket;
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as LiveFrame | { type: string; message?: string };
      if (message.type === "frame") {
        const live = message as LiveFrame;
        setLiveFrame(live);
        setFrameId(live.frame_id);
        if (live.dense_available) {
          setCloud({
            points: live.dense_points,
            colors: live.dense_colors,
            mode: "dense",
            label: "Single-frame live AI depth · arbitrary display units",
            available: true,
          });
        } else {
          setCloud({
            points: live.map_points,
            colors: live.map_colors,
            mode: "sparse",
            label: "Live sparse landmarks · ORB geometry",
            available: live.map_points.length > 0,
          });
        }
      } else if (message.type === "complete") {
        setPlaying(false);
      } else if (message.type === "error") {
        setError(message.message ?? "Live processing failed");
      }
    };
    socket.onerror = () => setError("Live WebSocket connection failed");
  };

  const handleUpload = async (file: File) => {
    setError("");
    setUploadName(file.name);
    try {
      const uploaded = await uploadVideo(file);
      setRunMode("live");
      setPlaying(true);
      setReplayFrame(null);
      startLive(uploaded.video_id);
    } catch (reason) {
      setError(String(reason));
    }
  };

  const switchToReplay = () => {
    socketRef.current?.close();
    setRunMode("replay");
    setLiveFrame(null);
    setPlaying(false);
    setFrameId(0);
    setUploadName("");
  };

  const stopLive = () => {
    socketRef.current?.close();
    socketRef.current = null;
    setPlaying(false);
  };

  const activeFrame = runMode === "live" ? liveFrame : replayFrame;
  const activeTrajectory = activeFrame?.trajectory ?? [];
  const activeCamera = activeFrame?.camera_position ?? null;
  const liveDenseReady = runMode === "live" && mapMode === "dense" && Boolean(liveFrame?.dense_available);
  const activePoints = runMode === "live" ? mapMode === "sparse" ? liveFrame?.map_points ?? [] : liveDenseReady ? cloud.points : [] : cloud.points;
  const activeColors = runMode === "live" ? mapMode === "sparse" ? liveFrame?.map_colors ?? [] : liveDenseReady ? cloud.colors : [] : cloud.colors;
  const mapMessage = useMemo(() => {
    if (runMode === "live" && mapMode === "dense") return liveDenseReady ? "Single-frame live AI depth · arbitrary display units" : "AI depth cloud unavailable; set SLAM_LIVE_DEPTH=1 with depth extras on FastAPI";
    if (runMode === "live" && mapMode === "sensor") return "Sensor depth is unavailable for a generic MP4 video";
    if (runMode === "replay" && mapMode === "dense") return "Single-frame AI depth artifact · arbitrary display units";
    if (runMode === "replay" && mapMode === "sensor") return "Single-frame TUM sensor depth artifact";
    return "Live sparse landmarks · ORB geometry";
  }, [liveDenseReady, mapMode, runMode]);

  const metrics = runMode === "live" ? liveFrame?.diagnostics ?? {} : replayFrame?.diagnostics ?? {};
  const videoSource = runMode === "live" && liveFrame ? `data:image/jpeg;base64,${liveFrame.jpeg}` : summary?.rgb_preview_url ? backendUrl(summary.rgb_preview_url) : undefined;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand">◈ Visual SLAM Dashboard</div>
          <div className="subtitle">React + FastAPI · replayable V3 baseline · CPU-compatible live path</div>
        </div>
        <div className={`source-badge ${runMode}`}>{runMode === "live" ? "LIVE · WEBSOCKET" : "REPLAY · V3 ARTIFACT"}</div>
      </header>

      <section className="toolbar">
        <label className="file-button">
          Open video
          <input type="file" accept="video/*" onChange={(event) => event.target.files?.[0] && handleUpload(event.target.files[0])} />
        </label>
        <button onClick={switchToReplay}>Replay artifacts</button>
        <label>Mode</label>
        <select value={runMode} onChange={(event) => event.target.value === "replay" ? switchToReplay() : setRunMode("live")}>
          <option value="replay">Replay Mode</option>
          <option value="live">Live Processing</option>
        </select>
        <label>Map</label>
        <select value={mapMode} onChange={(event) => setMapMode(event.target.value as MapMode)}>
          {(Object.keys(mapLabels) as MapMode[]).map((mode) => <option key={mode} value={mode}>{mapLabels[mode]}</option>)}
        </select>
        <label className="checkbox"><input type="checkbox" checked={showFeatures} onChange={(event) => setShowFeatures(event.target.checked)} /> ORB features</label>
        <span className="toolbar-spacer" />
        {runMode === "replay" ? (
          <button onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Start"}</button>
        ) : (
          <button onClick={stopLive}>Stop live</button>
        )}
        {uploadName && <span className="file-name">{uploadName}</span>}
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="dashboard-grid">
        <ImageCard title="RGB VIDEO" badge={runMode === "live" ? "STREAMING" : "NOTEBOOK ARTIFACT"}>
          {videoSource ? <div className="video-frame"><img src={videoSource} alt="RGB video frame" />{showFeatures && liveFrame?.keypoints.map(([x, y], index) => <i className="keypoint" key={index} style={{ left: `${(x / 640) * 100}%`, top: `${(y / 480) * 100}%` }} />)}</div> : <EmptyState text="Select an MP4 video or load replay artifacts" />}
          <div className="image-meta">Frame {String(frameId).padStart(4, "0")} · {activeFrame?.tracking_method ?? "waiting"}</div>
        </ImageCard>

        <section className="panel map-card">
          <div className="panel-header"><span>3D MAP</span><span className="panel-hint">drag orbit · right-drag pan · wheel zoom</span></div>
          <MapCanvas points={activePoints} colors={activeColors} trajectory={activeTrajectory} camera={activeCamera} message={mapMessage} />
          <div className="map-legend"><span className="dot blue" /> landmarks <span className="dot red" /> trajectory <span className="dot teal" /> camera</div>
        </section>

        <ImageCard title="AI DEPTH · RELATIVE / SINGLE FRAME" badge={runMode === "replay" ? "OFFLINE ARTIFACT" : "OPTIONAL"}>
          {runMode === "replay" && summary?.depth_preview_url ? <img className="depth-image" src={backendUrl(summary.depth_preview_url)} alt="Depth Anything V2 relative depth" /> : liveFrame?.depth_jpeg ? <img className="depth-image" src={`data:image/jpeg;base64,${liveFrame.depth_jpeg}`} alt="Live relative depth" /> : <EmptyState text="Live AI depth is optional; set SLAM_LIVE_DEPTH=1 and install the depth extras on FastAPI" />}
          <div className="image-meta">{liveFrame?.depth_time ? `Frame ${liveFrame.frame_id} · ${liveFrame.depth_time} ms · ` : ""}Relative depth is not metric depth and is not fused across frames</div>
        </ImageCard>

        <section className="panel trajectory-card">
          <div className="panel-header"><span>ESTIMATED TRAJECTORY</span><span className="panel-hint">X · Z projection</span></div>
          <TrajectoryView trajectory={activeTrajectory} />
        </section>
      </section>

      <section className="panel diagnostics-card">
        <div className="panel-header"><span>TRACKING DIAGNOSTICS</span><span className={`status ${activeFrame?.tracking_status?.toLowerCase() ?? "initializing"}`}>● {activeFrame?.tracking_status ?? "INITIALIZING"}</span></div>
        <div className="metrics-grid">
          <Metric name="Frame" value={activeFrame?.frame_id} />
          <Metric name="ORB keypoints" value={metrics.orb_keypoints} />
          <Metric name="Matches" value={activeFrame?.matches} />
          <Metric name="Inliers" value={activeFrame?.inliers} />
          <Metric name="Inlier ratio" value={activeFrame ? `${(activeFrame.inlier_ratio * 100).toFixed(1)}%` : undefined} />
          <Metric name="Map points" value={metrics.map_points} />
          <Metric name="Keyframes" value={metrics.keyframes} />
          <Metric name="Coverage" value={metrics.coverage} suffix="%" />
          <Metric name="Lost frames" value={metrics.lost_frames} />
          <Metric name="Longest streak" value={metrics.longest_streak} />
          <Metric name="Tracking FPS" value={metrics.tracking_fps} />
          <Metric name="Depth time" value={metrics.depth_time} />
        </div>
      </section>

      <div className="timeline">
        <input type="range" min={0} max={Math.max(0, (summary?.frame_count ?? 1) - 1)} value={frameId} disabled={runMode === "live"} onChange={(event) => setFrameId(Number(event.target.value))} />
        <span>Frame {String(frameId).padStart(4, "0")} / {String(Math.max(0, (summary?.frame_count ?? 1) - 1)).padStart(4, "0")}</span>
      </div>
    </main>
  );
}

function Metric({ name, value, suffix = "" }: { name: string; value: unknown; suffix?: string }) {
  return <div className="metric"><span>{name}</span><strong>{formatMetric(value, suffix)}</strong></div>;
}

function ImageCard({ title, badge, children }: { title: string; badge: string; children: ReactNode }) {
  return <section className="panel image-card"><div className="panel-header"><span>{title}</span><span className="panel-hint">{badge}</span></div>{children}</section>;
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>;
}

function MapCanvas({ points, colors, trajectory, camera, message }: { points: number[][]; colors: number[][]; trajectory: number[][]; camera: number[] | null; message: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState({ yaw: 0.65, pitch: -0.35, zoom: 1, panX: 0, panY: 0 });
  const drag = useRef<{ x: number; y: number; button: number } | null>(null);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const ratio = window.devicePixelRatio || 1;
    const width = element.clientWidth;
    const height = element.clientHeight;
    element.width = width * ratio;
    element.height = height * ratio;
    const context = element.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.fillStyle = "#090c10";
    context.fillRect(0, 0, width, height);
    const all = [...points, ...trajectory];
    if (!all.length) {
      context.fillStyle = "#7e8da0";
      context.textAlign = "center";
      context.fillText(message, width / 2, height / 2);
      return;
    }
    const center = [0, 1, 2].map((axis) => all.reduce((sum, point) => sum + point[axis], 0) / all.length);
    const spread = Math.max(...all.map((point) => Math.hypot(point[0] - center[0], point[1] - center[1], point[2] - center[2])), 1);
    const project = (point: number[]) => {
      const x = point[0] - center[0]; const y = point[1] - center[1]; const z = point[2] - center[2];
      const cy = Math.cos(view.yaw); const sy = Math.sin(view.yaw); const cp = Math.cos(view.pitch); const sp = Math.sin(view.pitch);
      const rx = cy * x + sy * z; const rz = -sy * x + cy * z;
      const ry = cp * y - sp * rz; const depth = sp * y + cp * rz;
      const scale = Math.min(width, height) * 0.42 * view.zoom / spread;
      return [width / 2 + rx * scale + view.panX, height / 2 - ry * scale + view.panY, depth];
    };
    points.forEach((point, index) => {
      const [x, y] = project(point); const color = colors[index] ?? [40, 160, 220];
      context.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`; context.fillRect(x, y, 2, 2);
    });
    if (trajectory.length > 1) {
      context.strokeStyle = "#ff6b6b"; context.lineWidth = 2; context.beginPath();
      trajectory.forEach((point, index) => { const [x, y] = project(point); index ? context.lineTo(x, y) : context.moveTo(x, y); });
      context.stroke();
    }
    if (camera) { const [x, y] = project(camera); context.fillStyle = "#27c2b3"; context.beginPath(); context.arc(x, y, 5, 0, Math.PI * 2); context.fill(); }
    context.fillStyle = "#7e8da0"; context.font = "11px Segoe UI"; context.fillText("X red · Y green · Z blue", 12, height - 12);
  }, [camera, colors, message, points, trajectory, view]);

  return <canvas ref={canvas} className="map-canvas" onPointerDown={(event) => { drag.current = { x: event.clientX, y: event.clientY, button: event.button }; event.currentTarget.setPointerCapture(event.pointerId); }} onPointerMove={(event) => { if (!drag.current) return; const dx = event.clientX - drag.current.x; const dy = event.clientY - drag.current.y; drag.current = { ...drag.current, x: event.clientX, y: event.clientY }; setView((current) => drag.current?.button === 2 ? { ...current, panX: current.panX + dx, panY: current.panY + dy } : { ...current, yaw: current.yaw + dx * 0.012, pitch: Math.max(-1.5, Math.min(1.5, current.pitch + dy * 0.012)) }); }} onPointerUp={() => { drag.current = null; }} onContextMenu={(event) => event.preventDefault()} onWheel={(event) => { event.preventDefault(); setView((current) => ({ ...current, zoom: Math.max(0.12, Math.min(8, current.zoom * (event.deltaY < 0 ? 1.15 : 0.87))) })); }} />;
}

function TrajectoryView({ trajectory }: { trajectory: number[][] }) {
  if (trajectory.length < 2) return <div className="empty-state trajectory-empty">Trajectory unavailable</div>;
  const x = trajectory.map((point) => point[0]); const z = trajectory.map((point) => point[2]);
  const minX = Math.min(...x); const maxX = Math.max(...x); const minZ = Math.min(...z); const maxZ = Math.max(...z);
  const sx = (value: number) => 12 + ((value - minX) / Math.max(maxX - minX, 0.001)) * 376;
  const sy = (value: number) => 164 - ((value - minZ) / Math.max(maxZ - minZ, 0.001)) * 140;
  return <svg className="trajectory-svg" viewBox="0 0 400 180"><rect x="8" y="8" width="384" height="164" fill="#090c10" stroke="#2c3948" /> <polyline points={trajectory.map((point) => `${sx(point[0])},${sy(point[2])}`).join(" ")} fill="none" stroke="#27c2b3" strokeWidth="3" /><circle cx={sx(x[x.length - 1])} cy={sy(z[z.length - 1])} r="6" fill="#27c2b3" /></svg>;
}
