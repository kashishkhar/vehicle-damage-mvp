// app/types.ts

/** -------- Shared primitives -------- */
export type YoloBoxRel = {
  bbox_rel: [number, number, number, number]; // normalized [0..1]
  confidence: number; // 0..1
};

export type Vehicle = {
  make: string | null;
  model: string | null;
  color: string | null;
  confidence: number; // 0..1
};

export type ApiError = { error: string; error_code?: string };

/** -------- Detect route types -------- */
export type DetectPayload = {
  model: string;               // which classifier was used
  runId: string;
  image_sha256: string;
  yolo_boxes: YoloBoxRel[];    // normalized bboxes (0..1)
  vehicle: Vehicle;            // light guess
  is_vehicle: boolean;         // quick gate
  has_damage: boolean;         // inferred: yolo_boxes.length > 0
  quality_ok: boolean;         // quick gate
  issues: string[];            // e.g., ["blurry","low_light"]
};

/** -------- Analyze route types -------- */
export type Zone =
  | "front" | "front-left" | "left" | "rear-left" | "rear"
  | "rear-right" | "right" | "front-right" | "roof" | "unknown";

export type Part =
  | "bumper" | "fender" | "door" | "hood" | "trunk"
  | "quarter-panel" | "headlight" | "taillight" | "grille"
  | "mirror" | "windshield" | "wheel" | "unknown";

export type DamageType =
  | "dent" | "scratch" | "crack" | "paint-chips"
  | "broken" | "bent" | "missing" | "glass-crack" | "unknown";

export type DamageItem = {
  zone: Zone;
  part: Part;
  damage_type: DamageType;
  severity: 1 | 2 | 3 | 4 | 5;
  confidence: number;          // 0..1
  est_labor_hours: number;
  needs_paint: boolean;
  likely_parts: string[];
  bbox_rel?: [number, number, number, number];
  polygon_rel?: [number, number][];
};

export type CostBand = {
  currency: "USD";
  cost_low: number;
  cost_high: number;
  assumptions: string[];
};

export type Decision =
  | { label: "AUTO-APPROVE"; reasons: string[] }
  | { label: "INVESTIGATE"; reasons: string[] }
  | { label: "SPECIALIST"; reasons: string[] };

export type AnalyzePayload = {
  schema_version: string;
  model: string;
  runId: string;
  image_sha256: string;
  vehicle: Vehicle;
  damage_items: DamageItem[];
  narrative: string;
  normalization_notes: string;
  estimate: CostBand;
  decision: Decision;
  damage_summary: string;
};
