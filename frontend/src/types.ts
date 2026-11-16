export type StageStatus = "pending" | "active" | "done" | "skipped";

export type Stage = {
  id: string;
  label: string;
  status: StageStatus;
};

export type Garment = {
  id: string;
  label: string;
  product_tag: string;
  confidence: number;
  area_fraction: number;
  bbox: number[];
  notes: string[];
  overlay_url: string;
};

export type Job = {
  id: string;
  status: "queued" | "running" | "awaiting_selection" | "completed" | "failed";
  stage: string;
  stages: Stage[];
  message: string;
  progress: number;
  image_id: string;
  result_id: string | null;
  error: string | null;
  garments: Garment[];
  options: {
    parser?: string;
    reconstruct?: boolean;
    refine_sam2?: boolean;
    clean_background?: boolean;
    product_tag?: string | null;
  };
  preview_url: string;
  created_at: string;
};

export type Upload = {
  image_id: string;
  filename: string;
  width: number;
  height: number;
  bytes: number;
  preview_url: string;
  original_url: string;
};

export type ResultRecord = {
  id: string;
  image_id: string;
  job_id: string;
  width: number;
  height: number;
  confidence: number;
  low_confidence: boolean;
  reliable: boolean;
  parser: string;
  parser_license: string;
  reconstruction_method: string;
  reconstruction_message: string;
  product_tag: string;
  garment_label: string;
  garment_ids: string[];
  notes: string[];
  manually_edited: boolean;
  created_at: string;
  updated_at: string;
  original_preview_url: string;
  transparent_preview_url: string;
  white_preview_url: string;
  mask_url: string;
  thumbnail_url: string;
  download_png_url: string;
  download_jpeg_url: string;
  upload_preview_url: string;
};

export type HistoryResponse = {
  count: number;
  items: Array<{
    id: string;
    product_tag: string;
    garment_label: string;
    confidence: number;
    low_confidence: boolean;
    parser: string;
    created_at: string;
    thumbnail_url: string;
  }>;
};

export type ModelInfo = {
  id: string;
  ready: boolean;
  installed: boolean;
  license: string;
  purpose: string;
};

export type Health = {
  app: string;
  python: string;
  torch: string | null;
  gpu: { available: boolean; name: string | null; vram_mb: number | null; torch: string | null };
  defaults: {
    parser: "auto" | "segformer" | "mediapipe";
    reconstruct: boolean;
    refine_sam2: boolean;
    clean_background: boolean;
  };
  product_tags: string[];
  models: Record<string, ModelInfo>;
};

export type ExtractOptions = {
  parser: "auto" | "segformer" | "mediapipe";
  reconstruct: boolean;
  refine_sam2: boolean;
  clean_background: boolean;
};
