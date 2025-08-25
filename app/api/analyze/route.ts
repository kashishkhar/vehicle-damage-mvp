// app/api/analyze/route.ts
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";
import {
  AnalyzeResponse,
  DamageItem,
  Estimate,
  Vehicle,
  Decision,
} from "@/app/types";

export const runtime = "nodejs";

/* ---------- Config ---------- */
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
const MODEL_ID = (process.env.MODEL_VISION || "gpt-4o-mini").trim();
const SCHEMA_VERSION = "1.2.0";

const LABOR = Number(process.env.LABOR_RATE ?? 95);
const PAINT = Number(process.env.PAINT_MAT_COST ?? 180);

// Business knobs (routing)
const AUTO_MAX_COST       = Number(process.env.AUTO_MAX_COST ?? 1500);
const AUTO_MAX_SEVERITY   = Number(process.env.AUTO_MAX_SEVERITY ?? 2);
const AUTO_MIN_CONF       = Number(process.env.AUTO_MIN_CONF ?? 0.75);
const SPEC_MIN_COST       = Number(process.env.SPECIALIST_MIN_COST ?? 5000);
const SPEC_MIN_SEVERITY   = Number(process.env.SPECIALIST_MIN_SEVERITY ?? 4);

// Optional YOLO
const YOLO_API_URL = (process.env.YOLO_API_URL || "").trim();
const YOLO_API_KEY = (process.env.YOLO_API_KEY || "").trim();

/* ---------- Types for YOLO ---------- */
type RawPrediction = {
  x?: number; y?: number; width?: number; height?: number; // abs center format
  w?: number; h?: number;                                  // normalized size
  confidence?: number; conf?: number;
  x_min?: number; y_min?: number; x_max?: number; y_max?: number; // normalized corners
  image_width?: number; image_height?: number;
};

type YoloBox = { bbox_rel: [number, number, number, number]; confidence: number };

/* ---------- Helpers ---------- */
function sha256(buf: Buffer) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}
function sha256String(s: string) {
  return crypto.createHash("sha256").update(s, "utf8").digest("hex");
}
function hoursFor(part: string, sev: number): number {
  const base =
    part === "bumper" ? 1.2 :
    part === "door" ? 1.5 :
    part === "fender" ? 1.2 :
    part === "hood" ? 1.4 :
    part === "quarter-panel" ? 2.0 :
    part === "headlight" || part === "taillight" ? 0.6 :
    part === "grille" ? 0.8 :
    part === "mirror" ? 0.5 :
    part === "windshield" ? 1.2 :
    part === "wheel" ? 0.7 :
    part === "trunk" ? 1.4 : 1.0;
  const sevMult = sev <= 1 ? 0.5 : sev === 2 ? 0.8 : sev === 3 ? 1.0 : sev === 4 ? 1.4 : 1.8;
  return +(base * sevMult).toFixed(2);
}
function needsPaintFor(type: string, sev: number, part: string): boolean {
  if (["windshield", "headlight", "taillight", "mirror"].includes(part)) return false;
  if (type.includes("scratch") || type.includes("paint")) return true;
  return sev >= 2;
}
function estimateFromItems(items: DamageItem[]): Estimate {
  let labor = 0, paint = 0, parts = 0;
  const paintedZones = new Set<string>();
  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const hrs = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(d.part, sev);
    labor += hrs * LABOR;

    const paintNeeded =
      typeof d.needs_paint === "boolean" ? d.needs_paint : needsPaintFor(String(d.damage_type || ""), sev, d.part);
    if (paintNeeded && d.zone && !paintedZones.has(d.zone)) {
      paint += PAINT;
      paintedZones.add(d.zone);
    }

    const likelyParts = Array.isArray(d.likely_parts) ? d.likely_parts : [];
    if (sev >= 4 || likelyParts.length) {
      const perPart = 250;
      parts += perPart * Math.max(1, likelyParts.length || 1);
    }
  }
  const subtotal = labor + paint + parts;
  const variance = items.some(i => Number(i.severity ?? 1) >= 4) ? 0.25 : 0.15;
  return {
    currency: "USD",
    cost_low: Math.round(subtotal * (1 - variance)),
    cost_high: Math.round(subtotal * (1 + variance)),
    assumptions: [
      `Labor $${LABOR}/hr`,
      `Paint/material $${PAINT}/panel/zone`,
      `Parts allowance (midpoint)`,
      `Visual-only estimate; subject to teardown`,
    ],
  };
}
function aggregateConfidence(items: DamageItem[]): number {
  let num = 0, den = 0;
  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const conf = Number(d.confidence ?? 0.5);
    const w = 1 + 0.2 * (sev - 1);
    num += conf * w;
    den += w;
  }
  return den ? num / den : 0.5;
}
function routeDecision(items: DamageItem[], estimate: { cost_high: number }): Decision {
  const maxSev = items.reduce((m, d) => Math.max(m, Number(d.severity ?? 1)), 0);
  const agg = aggregateConfidence(items);
  const hi = Number(estimate?.cost_high ?? 0);

  if (maxSev <= AUTO_MAX_SEVERITY && hi <= AUTO_MAX_COST && agg >= AUTO_MIN_CONF) {
    return { label: "AUTO-APPROVE", reasons: [
      `severity ≤ ${AUTO_MAX_SEVERITY}`,
      `cost_high ≤ $${AUTO_MAX_COST}`,
      `agg_conf ≥ ${Math.round(AUTO_MIN_CONF * 100)}%`,
    ]};
  }
  if (maxSev >= SPEC_MIN_SEVERITY || hi >= SPEC_MIN_COST) {
    return { label: "SPECIALIST", reasons: [
      ...(maxSev >= SPEC_MIN_SEVERITY ? [`severity ≥ ${SPEC_MIN_SEVERITY}`] : []),
      ...(hi >= SPEC_MIN_COST ? [`cost_high ≥ $${SPEC_MIN_COST}`] : []),
    ]};
  }
  return { label: "INVESTIGATE", reasons: [
    `agg_conf ${Math.round(agg * 100)}%`,
    `max_severity ${maxSev}`,
    `cost_high $${hi}`,
  ]};
}

/* ---------- YOLO integration (optional) ---------- */
async function detectWithYolo(imageUrlOrDataUrl: string): Promise<YoloBox[]> {
  if (!YOLO_API_URL || !YOLO_API_KEY) return [];
  try {
    const url = new URL(YOLO_API_URL);
    url.searchParams.set("api_key", YOLO_API_KEY);
    const res = await fetch(url.toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageUrlOrDataUrl }),
    });
    if (!res.ok) return [];

    // Normalized attempt to read predictions
    const data = (await res.json()) as unknown;
    let preds: RawPrediction[] = [];
    if (data && typeof data === "object") {
      if ("predictions" in data) preds = (data as { predictions: RawPrediction[] }).predictions ?? [];
      else if ("outputs" in data) preds = (data as { outputs: RawPrediction[] }).outputs ?? [];
      else if ("result" in data && (data as { result?: { predictions?: RawPrediction[] } }).result?.predictions) {
        preds = (data as { result?: { predictions?: RawPrediction[] } }).result!.predictions!;
      }
    }

    const boxes: YoloBox[] = [];
    for (const p of preds) {
      const conf = typeof p.confidence === "number" ? p.confidence : (typeof p.conf === "number" ? p.conf : 0.5);

      // normalized corners
      if (
        typeof p.x_min === "number" && typeof p.x_max === "number" &&
        typeof p.y_min === "number" && typeof p.y_max === "number"
      ) {
        const nx = Math.max(0, Math.min(1, p.x_min));
        const ny = Math.max(0, Math.min(1, p.y_min));
        const nw = Math.max(0, Math.min(1, p.x_max - p.x_min));
        const nh = Math.max(0, Math.min(1, p.y_max - p.y_min));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf });
        continue;
      }

      // absolute center with image dims
      if (
        typeof p.x === "number" && typeof p.y === "number" &&
        typeof p.width === "number" && typeof p.height === "number" &&
        typeof p.image_width === "number" && typeof p.image_height === "number"
      ) {
        const nx = (p.x - p.width / 2) / p.image_width;
        const ny = (p.y - p.height / 2) / p.image_height;
        const nw = p.width / p.image_width;
        const nh = p.height / p.image_height;
        boxes.push({
          bbox_rel: [
            Math.max(0, Math.min(1, nx)),
            Math.max(0, Math.min(1, ny)),
            Math.max(0, Math.min(1, nw)),
            Math.max(0, Math.min(1, nh)),
          ],
          confidence: conf,
        });
        continue;
      }

      // normalized center + size
      if (typeof p.x === "number" && typeof p.y === "number" && typeof p.w === "number" && typeof p.h === "number") {
        const nx = Math.max(0, Math.min(1, p.x - p.w / 2));
        const ny = Math.max(0, Math.min(1, p.y - p.h / 2));
        const nw = Math.max(0, Math.min(1, p.w));
        const nh = Math.max(0, Math.min(1, p.h));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf });
        continue;
      }
    }
    return boxes;
  } catch {
    return [];
  }
}

/* ---------- Mock (for demos) ---------- */
function mockPayload(): AnalyzeResponse {
  const items: DamageItem[] = [
    {
      zone: "front-left",
      part: "bumper",
      damage_type: "dent",
      severity: 2,
      confidence: 0.87,
      est_labor_hours: 1.0,
      needs_paint: true,
      likely_parts: [],
      bbox_rel: [0.18, 0.62, 0.28, 0.18],
    },
    {
      zone: "front-left",
      part: "headlight",
      damage_type: "crack",
      severity: 3,
      confidence: 0.81,
      est_labor_hours: 0.6,
      needs_paint: false,
      likely_parts: ["headlight assembly"],
      polygon_rel: [
        [0.44, 0.50], [0.53, 0.50], [0.56, 0.58], [0.47, 0.60]
      ],
    },
  ];
  const estimate = estimateFromItems(items);
  const decision = routeDecision(items, estimate);
  return {
    schema_version: SCHEMA_VERSION,
    model: MODEL_ID,
    runId: crypto.randomUUID(),
    image_sha256: "mock",
    vehicle: { make: "Honda", model: "Civic", color: "Silver", confidence: 0.9 },
    damage_items: items,
    narrative: "Front-left impact; cosmetic bumper dent and cracked headlight. No obvious structural deformation.",
    normalization_notes: "Assumed sedan; adequate lighting; minor occlusion.",
    estimate,
    decision,
    damage_summary: items.map(d => `${d.zone} ${d.part} — ${d.damage_type}, sev ${d.severity}`).join("; "),
  };
}

/* ---------- Route ---------- */
export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    if (!file && !imageUrl) {
      return NextResponse.json({ error: "No file or imageUrl provided" }, { status: 400 });
    }

    // Mock?
    if (process.env.MOCK_API === "1") {
      return NextResponse.json(mockPayload());
    }

    // Build image source + audit hash
    let imageSourceUrl: string;
    let imageHash: string;

    if (file) {
      const arr = await file.arrayBuffer();
      const buf = Buffer.from(arr);
      const dataUrl = `data:${file.type || "image/jpeg"};base64,${buf.toString("base64")}`;
      imageSourceUrl = dataUrl;
      imageHash = sha256(buf);
    } else {
      try {
        const u = new URL(imageUrl!);
        if (!/^https?:$/.test(u.protocol)) {
          return NextResponse.json({ error: "imageUrl must be http(s)" }, { status: 400 });
        }
      } catch {
        return NextResponse.json({ error: "Invalid imageUrl" }, { status: 400 });
      }
      imageSourceUrl = imageUrl!;
      imageHash = sha256String(imageUrl!);
    }

    // Optional YOLO detection (fast geometry seeds)
    const yoloBoxes = await detectWithYolo(imageSourceUrl);
    const yoloContext = yoloBoxes.length
      ? `YOLO_SEEDS: ${JSON.stringify(yoloBoxes.slice(0, 12))}`
      : `YOLO_SEEDS: []`;

    // Strict schema; GPT fills metadata + labels (and geometry if YOLO absent)
    const system = `
Return ONLY valid JSON matching EXACTLY this shape (no prose, no markdown):

{
  "vehicle": { "make": string | null, "model": string | null, "color": string | null, "confidence": number },
  "damage_items": Array<{
    "zone": "front"|"front-left"|"left"|"rear-left"|"rear"|"rear-right"|"right"|"front-right"|"roof"|"unknown",
    "part": "bumper"|"fender"|"door"|"hood"|"trunk"|"quarter-panel"|"headlight"|"taillight"|"grille"|"mirror"|"windshield"|"wheel"|"unknown",
    "damage_type": "dent"|"scratch"|"crack"|"paint-chips"|"broken"|"bent"|"missing"|"glass-crack"|"unknown",
    "severity": 1|2|3|4|5,
    "confidence": number,
    "est_labor_hours": number,
    "needs_paint": boolean,
    "likely_parts": string[],
    "bbox_rel"?: [number, number, number, number],
    "polygon_rel"?: Array<[number, number]>
  }>,
  "narrative": string,
  "normalization_notes": string
}

Rules:
- Confidence ∈ [0,1].
- If unsure on vehicle fields, set to null but still provide a numeric confidence.
- Severity 1..5 (1=very minor, 5=severe/structural). Be conservative.
- est_labor_hours realistic per item.
- likely_parts may be empty.
- Geometry normalized to image width/height (0..1). Prefer polygon_rel for irregular scratches; else bbox_rel.
- If YOLO boxes are provided below, ALIGN your geometry to those boxes where applicable (do not invent far-away regions).
- No extra keys. No markdown. JSON only.
`.trim();

    const completion = await client.chat.completions.create({
      model: MODEL_ID,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: system },
        {
          role: "user",
          content: [
            { type: "text", text: "Analyze this car image and fill the schema. Be concise and conservative." },
            { type: "text", text: yoloContext },
            { type: "image_url", image_url: { url: imageSourceUrl } },
          ],
        },
      ],
    });

    const raw = completion.choices?.[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(raw) as {
      vehicle?: Vehicle;
      damage_items?: unknown;
      narrative?: string;
      normalization_notes?: string;
    };

    // Normalize items strictly
    const items: DamageItem[] = Array.isArray(parsed.damage_items)
      ? (parsed.damage_items as unknown[]).map((r) => {
          const d = r as Partial<DamageItem>;
          const severity = typeof d.severity === "number" ? d.severity : 2;
          const part = (d.part ?? "unknown") as DamageItem["part"];
          const damage_type = (d.damage_type ?? "unknown") as DamageItem["damage_type"];
          const zone = (d.zone ?? "unknown") as DamageItem["zone"];
          const est_labor_hours =
            typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(String(part), severity);
          const needs_paint =
            typeof d.needs_paint === "boolean"
              ? d.needs_paint
              : needsPaintFor(String(damage_type), severity, String(part));
          const likely_parts = Array.isArray(d.likely_parts) ? d.likely_parts.map(String) : [];
          const confidence = typeof d.confidence === "number" ? d.confidence : 0.5;

          const out: DamageItem = {
            zone,
            part,
            damage_type,
            severity,
            confidence,
            est_labor_hours,
            needs_paint,
            likely_parts,
          };

          if (
            Array.isArray(d.bbox_rel) &&
            d.bbox_rel.length === 4 &&
            d.bbox_rel.every((n) => typeof n === "number" && n >= 0 && n <= 1)
          ) {
            out.bbox_rel = d.bbox_rel as [number, number, number, number];
          }
          if (
            Array.isArray(d.polygon_rel) &&
            d.polygon_rel.length >= 3 &&
            d.polygon_rel.length <= 12 &&
            d.polygon_rel.every(
              (pt) => Array.isArray(pt) && pt.length === 2 && pt.every((n) => typeof n === "number" && n >= 0 && n <= 1),
            )
          ) {
            out.polygon_rel = d.polygon_rel as [number, number][];
          }

          return out;
        })
      : [];

    const estimate = estimateFromItems(items);
    const decision = routeDecision(items, estimate);

    const damage_summary = (items.length
      ? items.map((d) => `${d.zone} ${d.part} — ${d.damage_type}, sev ${d.severity}`).join("; ")
      : (parsed.narrative || "")).slice(0, 400);

    const payload: AnalyzeResponse = {
      schema_version: SCHEMA_VERSION,
      model: MODEL_ID,
      runId: crypto.randomUUID(),
      image_sha256: imageHash,
      vehicle: parsed.vehicle ?? { make: null, model: null, color: null, confidence: 0 },
      damage_items: items,
      narrative: parsed.narrative ?? "",
      normalization_notes: parsed.normalization_notes ?? "",
      estimate,
      decision,
      damage_summary,
    };

    return NextResponse.json(payload);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Server error";
    console.error(e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
