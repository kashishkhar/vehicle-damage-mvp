import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";

export const runtime = "nodejs";

/* ---------- Config ---------- */
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
const MODEL_ID = process.env.MODEL_VISION || "gpt-4o-mini";
const SCHEMA_VERSION = "1.2.0";

const LABOR = Number(process.env.LABOR_RATE ?? 95);
const PAINT = Number(process.env.PAINT_MAT_COST ?? 180);

const YOLO_API_URL = (process.env.YOLO_API_URL || "").trim();
const YOLO_API_KEY = (process.env.YOLO_API_KEY || "").trim();

/* ---------- Helpers ---------- */
function sha256(buf: Buffer) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}
function sha256String(s: string) {
  return crypto.createHash("sha256").update(s, "utf8").digest("hex");
}

function hoursFor(part: string, sev: number) {
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

function needsPaintFor(type: string, sev: number, part: string) {
  if (["windshield", "headlight", "taillight", "mirror"].includes(part)) return false;
  if (type?.includes?.("scratch") || type?.includes?.("paint")) return true;
  return sev >= 2;
}

function estimateFromItems(items: any[]) {
  let labor = 0, paint = 0, parts = 0;
  const paintedZones = new Set<string>();

  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const hrs = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(d.part, sev);
    labor += hrs * LABOR;

    const paintNeeded = typeof d.needs_paint === "boolean"
      ? d.needs_paint
      : needsPaintFor(String(d.damage_type || ""), sev, d.part);
    if (paintNeeded && d.zone && !paintedZones.has(d.zone)) {
      paint += PAINT;
      paintedZones.add(d.zone);
    }

    const likelyParts: string[] = Array.isArray(d.likely_parts) ? d.likely_parts : [];
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

function aggregateConfidence(items: any[]) {
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

const AUTO_MAX_COST       = Number(process.env.AUTO_MAX_COST ?? 1500);
const AUTO_MAX_SEVERITY   = Number(process.env.AUTO_MAX_SEVERITY ?? 2);
const AUTO_MIN_CONF       = Number(process.env.AUTO_MIN_CONF ?? 0.75);
const SPEC_MIN_COST       = Number(process.env.SPECIALIST_MIN_COST ?? 5000);
const SPEC_MIN_SEVERITY   = Number(process.env.SPECIALIST_MIN_SEVERITY ?? 4);

function routeDecision(items: any[], estimate: { cost_high: number }) {
  const maxSev = items.reduce((m, d) => Math.max(m, Number(d.severity ?? 1)), 0);
  const agg = aggregateConfidence(items);
  const hi = Number(estimate?.cost_high ?? 0);

  if (maxSev <= AUTO_MAX_SEVERITY && hi <= AUTO_MAX_COST && agg >= AUTO_MIN_CONF) {
    return {
      label: "AUTO-APPROVE",
      reasons: [
        `severity ≤ ${AUTO_MAX_SEVERITY}`,
        `cost_high ≤ $${AUTO_MAX_COST}`,
        `agg_conf ≥ ${Math.round(AUTO_MIN_CONF * 100)}%`,
      ],
    };
  }
  if (maxSev >= SPEC_MIN_SEVERITY || hi >= SPEC_MIN_COST) {
    return {
      label: "SPECIALIST",
      reasons: [
        ...(maxSev >= SPEC_MIN_SEVERITY ? [`severity ≥ ${SPEC_MIN_SEVERITY}`] : []),
        ...(hi >= SPEC_MIN_COST ? [`cost_high ≥ $${SPEC_MIN_COST}`] : []),
      ],
    };
  }
  return {
    label: "INVESTIGATE",
    reasons: [`agg_conf ${Math.round(agg * 100)}%`, `max_severity ${maxSev}`, `cost_high $${hi}`],
  };
}

/* ---------- YOLO (optional, tolerant) ---------- */
type RawPrediction = {
  x?: number; y?: number; width?: number; height?: number;
  w?: number; h?: number;
  confidence?: number; conf?: number;
  class?: string; label?: string;
  image_width?: number; image_height?: number;
  x_min?: number; y_min?: number; x_max?: number; y_max?: number;
};

async function detectWithYolo(imageUrlOrDataUrl: string): Promise<Array<{ bbox_rel: [number,number,number,number], confidence: number }>> {
  if (!YOLO_API_URL || !YOLO_API_KEY) return [];
  try {
    const url = new URL(YOLO_API_URL);
    url.searchParams.set("api_key", YOLO_API_KEY);
    const res = await fetch(url.toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageUrlOrDataUrl })
    });
    if (!res.ok) return [];
    const data: any = await res.json();
    const preds: RawPrediction[] = data?.predictions || data?.outputs || data?.result?.predictions || [];
    const boxes: Array<{ bbox_rel: [number,number,number,number], confidence: number }> = [];

    for (const p of preds) {
      const conf = typeof p.confidence === "number" ? p.confidence : (typeof p.conf === "number" ? p.conf : 0.5);

      if (typeof p.x_min === "number" && typeof p.x_max === "number" && typeof p.y_min === "number" && typeof p.y_max === "number") {
        const nx = Math.max(0, Math.min(1, p.x_min));
        const ny = Math.max(0, Math.min(1, p.y_min));
        const nw = Math.max(0, Math.min(1, p.x_max - p.x_min));
        const nh = Math.max(0, Math.min(1, p.y_max - p.y_min));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf });
        continue;
      }
      if (typeof p.x === "number" && typeof p.y === "number" && typeof p.width === "number" && typeof p.height === "number" && p.image_width && p.image_height) {
        const nx = (p.x - p.width / 2) / p.image_width;
        const ny = (p.y - p.height / 2) / p.image_height;
        const nw = p.width / p.image_width;
        const nh = p.height / p.image_height;
        boxes.push({ bbox_rel: [
          Math.max(0, Math.min(1, nx)),
          Math.max(0, Math.min(1, ny)),
          Math.max(0, Math.min(1, nw)),
          Math.max(0, Math.min(1, nh)),
        ], confidence: conf });
        continue;
      }
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

/* ---------- Mock ---------- */
function mockPayload() {
  const items = [
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
    warnings: [],
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

    // YOLO seeds (optional, tolerant)
    const yoloBoxes = await detectWithYolo(imageSourceUrl);

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
- Confidence ∈ [0,1]; be conservative.
- If unsure on vehicle fields, set them to null but still provide a numeric confidence.
- Severity 1..5 (1=very minor, 5=severe/structural).
- Geometry normalized to image width/height (0..1). Prefer polygon_rel for irregular scratches; else bbox_rel.
- If YOLO_SEEDS are provided, align your geometry to those where applicable (do not invent far-away regions).
- No extra keys. No markdown. JSON only.
`.trim();

    const yoloContext = yoloBoxes.length ? `YOLO_SEEDS: ${JSON.stringify(yoloBoxes.slice(0, 12))}` : `YOLO_SEEDS: []`;

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
    let parsed: any;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return NextResponse.json({ error: "Model returned invalid JSON" }, { status: 502 });
    }

    // Normalize/repair items
    const items: any[] = Array.isArray(parsed.damage_items) ? parsed.damage_items : [];
    for (const d of items) {
      d.severity = typeof d.severity === "number" ? d.severity : 2;
      d.est_labor_hours = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(d.part, d.severity);
      d.needs_paint = typeof d.needs_paint === "boolean" ? d.needs_paint : needsPaintFor(String(d.damage_type || ""), d.severity, d.part);
      if (!Array.isArray(d.likely_parts)) d.likely_parts = [];
      if (typeof d.confidence !== "number") d.confidence = 0.5;

      // Clamp geometry to [0..1]
      const bb = d.bbox_rel;
      if (bb) {
        const ok = Array.isArray(bb) && bb.length === 4 && bb.every((n: any) => typeof n === "number" && n >= 0 && n <= 1);
        if (!ok) delete d.bbox_rel;
      }
      const poly = d.polygon_rel;
      if (poly) {
        const ok = Array.isArray(poly) && poly.length >= 3 && poly.length <= 12 && poly.every(
          (pt: any) => Array.isArray(pt) && pt.length === 2 && pt.every((n: any) => typeof n === "number" && n >= 0 && n <= 1)
        );
        if (!ok) delete d.polygon_rel;
      }
    }

    const estimate = estimateFromItems(items);
    const decision = routeDecision(items, estimate);

    // Friendly non-blocking warnings to surface in UI
    const warnings: string[] = [];
    const vconf = Number(parsed?.vehicle?.confidence ?? 0);
    if (!parsed?.vehicle || vconf < 0.5) warnings.push("Low confidence identifying make/model/color.");
    if (!items.length) warnings.push("No clear damage regions detected.");
    if (items.length && items.every((d: any) => Number(d.confidence ?? 0) < 0.5)) warnings.push("Damage detections have low confidence.");

    const damage_summary = (items.length
      ? items.map((d) => `${d.zone} ${d.part} — ${d.damage_type}, sev ${d.severity}`).join("; ")
      : (parsed.narrative || "")).slice(0, 400);

    const payload = {
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
      warnings, // <-- UI reads this to show the gentle photo-quality note
    };

    return NextResponse.json(payload);
  } catch (e: any) {
    console.error(e);
    return NextResponse.json({ error: e?.message ?? "Server error" }, { status: 500 });
  }
}
