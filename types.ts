// app/types.ts

export type Zone =
  | "front" | "front-left" | "left" | "rear-left"
  | "rear" | "rear-right" | "right" | "front-right"
  | "roof" | "unknown";

export type Part =
  | "bumper" | "fender" | "door" | "hood" | "trunk"
  | "quarter-panel" | "headlight" | "taillight" | "grille"
  | "mirror" | "windshield" | "wheel" | "unknown";

export type DamageType =
  | "dent" | "scratch" | "crack" | "paint-chips"
  | "broken" | "bent" | "missing" | "glass-crack" | "unknown";

export interface DamageItem {
  zone: Zone;
  part: Part;
  damage_type: DamageType;
  severity: number;           // 1..5
  confidence: number;         // 0..1
  est_labor_hours: number;
  needs_paint: boolean;
  likely_parts: string[];
  bbox_rel?: [number, number, number, number];
  polygon_rel?: [number, number][];
}

export interface Vehicle {
  make: string | null;
  model: string | null;
  color: string | null;
  confidence: number; // 0..1
}

export interface Estimate {
  currency: "USD";
  cost_low: number;
  cost_high: number;
  assumptions: string[];
}

export type DecisionLabel = "AUTO-APPROVE" | "SPECIALIST" | "INVESTIGATE";

export interface Decision {
  label: DecisionLabel;
  reasons: string[];
}

export interface AnalyzeResponse {
  schema_version: string;
  model: string;
  runId: string;
  image_sha256: string;
  vehicle: Vehicle;
  damage_items: DamageItem[];
  narrative: string;
  normalization_notes: string;
  estimate: Estimate;
  decision: Decision;
  damage_summary: string;
}
