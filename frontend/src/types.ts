export type MapMode = "sparse" | "dense" | "sensor";
export type RunMode = "replay" | "live";

export interface Summary {
  frame_count: number;
  tracked_frames: number;
  coverage: number;
  longest_streak: number;
  map_points: number;
  keyframes: number;
  ate_rmse_m: number | null;
  rgbd_tracked_poses: number | null;
  depth_preview_url: string;
  rgb_preview_url: string;
  map_temporal: boolean;
}

export interface ReplayFrame {
  frame_id: number;
  timestamp: number;
  tracking_status: string;
  tracking_method: string;
  camera_position: number[] | null;
  trajectory: number[][];
  matches: number;
  inliers: number;
  inlier_ratio: number;
  diagnostics: Record<string, number | string>;
}

export interface Cloud {
  points: number[][];
  colors: number[][];
  mode: string;
  label: string;
  available: boolean;
}

export interface LiveFrame {
  type: "frame";
  frame_id: number;
  timestamp: number;
  jpeg: string;
  tracking_status: string;
  tracking_method: string;
  camera_position: number[] | null;
  keypoints: number[][];
  trajectory: number[][];
  map_points: number[][];
  map_colors: number[][];
  matches: number;
  inliers: number;
  inlier_ratio: number;
  diagnostics: Record<string, number | string>;
  dense_available: boolean;
  sensor_available: boolean;
  dense_points: number[][];
  dense_colors: number[][];
  depth_error?: string;
  depth_jpeg?: string | null;
  depth_time?: number | null;
}
