// app/api/analyze/route.ts
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";

export const runtime = "nodejs";

/** ---------- Types ---------- */
type Zone =
  | "front" | "front-left" | "left" | "rear-left" | "rear"
  | "rear-right" | "right" | "front-right" | "roof" | "unknown";
type Part =
  | "bumper" | "fender" | "door" | "hood" | "trunk"
  | "quarter-panel" | "headlight" | "taillight" | "grille"
  | "mirror" | "windshield" | "wheel" | "unknown";
type DamageType =
  | "dent" | "scratch" | "crack" | "paint-chips"
  | "broken" | "bent" | "missing" | "glass-crack" | "unknown";

export type DamageItem = {
  zone: Zone;
  part: Part;
  damage_type: DamageType;
  severity: 1 | 2 | 3 | 4 | 5;
  confidence: number;
  est_labor_hours: number;
  needs_paint: boolean;
  likely_parts: string[];
  bbox_rel?: [number, number, number, number];
  polygon_rel?: [number, number][];
};

export type Vehicle = {
  make: string | null;
  model: string | null;
  color: string | null;
  confidence: number;
};

export type AnalyzePayload = {
  schema_version: string;
  model: string;
  runId: string;
  image_sha256: string;
  vehicle: Vehicle;
  damage_items: DamageItem[];
  narrative: string;
  normalization_notes: string;
  estimate: {
    currency: "USD";
    cost_low: number;
    cost_high: number;
    assumptions: string[];
  };
  decision: { label: "AUTO-APPROVE" | "INVESTIGATE" | "SPECIALIST"; reasons: string[] };
  damage_summary: string;
};

/** ---------- Env ---------- */
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
const MODEL_ID = process.env.MODEL_VISION || "gpt-4o-mini";

const LABOR = Number(process.env.LABOR_RATE ?? 95);
const PAINT = Number(process.env.PAINT_MAT_COST ?? 180);

const AUTO_MAX_COST = Number(process.env.AUTO_MAX_COST ?? 1500);
const AUTO_MAX_SEVERITY = Number(process.env.AUTO_MAX_SEVERITY ?? 2);
const AUTO_MIN_CONF = Number(process.env.AUTO_MIN_CONF ?? 0.75);
const SPEC_MIN_COST = Number(process.env.SPECIALIST_MIN_COST ?? 5000);
const SPEC_MIN_SEVERITY = Number(process.env.SPECIALIST_MIN_SEVERITY ?? 4);

/** ---------- Utils ---------- */
function sha256(buf: Buffer) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}
function sha256String(s: string) {
  return crypto.createHash("sha256").update(s, "utf8").digest("hex");
}

// Bodyshop-ish heuristics
function hoursFor(part: Part, sev: number) {
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
    part === "trunk" ? 1.4 : 1.0; // unknown fallback
  const sevMult = sev <= 1 ? 0.5 : sev === 2 ? 0.8 : sev === 3 ? 1.0 : sev === 4 ? 1.4 : 1.8;
  return +(base * sevMult).toFixed(2);
}
function needsPaintFor(type: string, sev: number, part: Part) {
  if (["windshield", "headlight", "taillight", "mirror"].includes(part)) return false;
  if (type.includes("scratch") || type.includes("paint")) return true;
  return sev >= 2;
}
function estimateFromItems(items: DamageItem[]) {
  let labor = 0, paint = 0, parts = 0;
  const paintedZones = new Set<Zone>();

  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const hrs = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(d.part, sev);
    labor += hrs * LABOR;

    const needPaint = typeof d.needs_paint === "boolean" ? d.needs_paint : needsPaintFor(String(d.damage_type || ""), sev, d.part);
    if (needPaint && d.zone && !paintedZones.has(d.zone)) {
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
    currency: "USD" as const,
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
function aggregateConfidence(items: DamageItem[]) {
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
function routeDecision(items: DamageItem[], estimate: { cost_high: number }) {
  const maxSev = items.reduce((m, d) => Math.max(m, Number(d.severity ?? 1)), 0);
  const agg = aggregateConfidence(items);
  const hi = Number(estimate?.cost_high ?? 0);

  if (maxSev <= AUTO_MAX_SEVERITY && hi <= AUTO_MAX_COST && agg >= AUTO_MIN_CONF) {
    return { label: "AUTO-APPROVE" as const, reasons: [
      `severity ≤ ${AUTO_MAX_SEVERITY}`,
      `cost_high ≤ $${AUTO_MAX_COST}`,
      `agg_conf ≥ ${Math.round(AUTO_MIN_CONF * 100)}%`,
    ]};
  }
  if (maxSev >= SPEC_MIN_SEVERITY || hi >= SPEC_MIN_COST) {
    return { label: "SPECIALIST" as const, reasons: [
      ...(maxSev >= SPEC_MIN_SEVERITY ? [`severity ≥ ${SPEC_MIN_SEVERITY}`] : []),
      ...(hi >= SPEC_MIN_COST ? [`cost_high ≥ $${SPEC_MIN_COST}`] : []),
    ]};
  }
  return { label: "INVESTIGATE" as const, reasons: [
    `agg_conf ${Math.round(agg * 100)}%`, `max_severity ${maxSev}`, `cost_high $${hi}`
  ]};
}

/** ---------- Route ---------- */
export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    // Optional YOLO seeds passed from client as JSON string (array of {bbox_rel, confidence})
    const yoloSeedsRaw = (form.get("yolo") as string | null)?.toString() || "";
    let yoloSeeds: { bbox_rel: [number,number,number,number]; confidence: number }[] = [];
    if (yoloSeedsRaw) {
      try { const parsed = JSON.parse(yoloSeedsRaw) as unknown;
        if (Array.isArray(parsed)) {
          yoloSeeds = parsed.filter((b): b is { bbox_rel: [number,number,number,number]; confidence: number } =>
            Array.isArray((b as any).bbox_rel) && (b as any).bbox_rel.length === 4 && typeof (b as any).confidence === "number"
          );
        }
      } catch { /* ignore */ }
    }

    if (!file && !imageUrl) {
      return NextResponse.json({ error: "No file or imageUrl provided" }, { status: 400 });
    }

    // Build source url + audit hash
    let imageSourceUrl: string;
    let imageHash: string;

    if (file) {
      const arr = await file.arrayBuffer();
      const buf = Buffer.from(arr);
      imageSourceUrl = `data:${file.type || "image/jpeg"};base64,${buf.toString("base64")}`;
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

    // Strict schema prompt; GPT refines labels given YOLO seeds; JSON only
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
- Confidence ∈ [0,1]. Severity 1..5 (1=very minor, 5=severe/structural). Be conservative.
- est_labor_hours realistic per item; likely_parts may be empty.
- Geometry normalized in 0..1. Prefer polygon_rel for irregular scratches; else bbox_rel.
- If YOLO seeds are provided, ALIGN your geometry/labels to those regions when applicable; avoid inventing far-away areas.
- No extra keys. JSON only.
`.trim();

    const yoloContext = yoloSeeds.length ? `YOLO_SEEDS: ${JSON.stringify(yoloSeeds.slice(0, 12))}` : `YOLO_SEEDS: []`;

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
    let parsed: {
      vehicle?: Vehicle;
      damage_items?: DamageItem[];
      narrative?: string;
      normalization_notes?: string;
    };
    try { parsed = JSON.parse(raw); } catch {
      return NextResponse.json({ error: "Model returned invalid JSON" }, { status: 502 });
    }

    // Normalize/repair items
    const itemsIn = Array.isArray(parsed.damage_items) ? parsed.damage_items : [];
    const items: DamageItem[] = itemsIn.map((d) => {
      const sev = (typeof d.severity === "number" && d.severity >= 1 && d.severity <= 5 ? d.severity : 2) as 1|2|3|4|5;
      const p = (d.part ?? "unknown") as Part;
      const ht = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(p, sev);
      const paint = typeof d.needs_paint === "boolean" ? d.needs_paint : needsPaintFor(String(d.damage_type || ""), sev, p);
      const conf = typeof d.confidence === "number" ? Math.max(0, Math.min(1, d.confidence)) : 0.5;

      const out: DamageItem = {
        zone: (d.zone ?? "unknown") as Zone,
        part: p,
        damage_type: (d.damage_type ?? "unknown") as DamageType,
        severity: sev,
        confidence: conf,
        est_labor_hours: ht,
        needs_paint: paint,
        likely_parts: Array.isArray(d.likely_parts) ? d.likely_parts : [],
      };

      const bb = (d as any).bbox_rel;
      if (Array.isArray(bb) && bb.length === 4 && bb.every((n) => typeof n === "number" && n >= 0 && n <= 1)) {
        out.bbox_rel = [bb[0], bb[1], bb[2], bb[3]];
      }
      const poly = (d as any).polygon_rel;
      if (Array.isArray(poly) && poly.length >= 3 && poly.length <= 12 && poly.every(
        (pt) => Array.isArray(pt) && pt.length === 2 && pt.every((n) => typeof n === "number" && n >= 0 && n <= 1)
      )) {
        out.polygon_rel = poly.map((pt: [number, number]) => [pt[0], pt[1]]);
      }
      return out;
    });

    // If LLM returned zero items but YOLO had seeds, create placeholders so estimate & UI don’t collapse.
    const itemsFinal: DamageItem[] = items.length ? items : yoloSeeds.map((b) => ({
      zone: "unknown",
      part: "unknown",
      damage_type: "unknown",
      severity: 2,
      confidence: Math.max(0, Math.min(1, b.confidence)),
      est_labor_hours: hoursFor("unknown" as Part, 2),
      needs_paint: false,
      likely_parts: [],
      bbox_rel: b.bbox_rel,
    }));

    const estimate = estimateFromItems(itemsFinal);
    const decision = routeDecision(itemsFinal, estimate);

    const damage_summary = (itemsFinal.length
      ? itemsFinal.map((d) => `${d.zone} ${d.part} — ${d.damage_type}, sev ${d.severity}`).join("; ")
      : (parsed.narrative || "")
    ).slice(0, 400);

    const payload: AnalyzePayload = {
      schema_version: "1.3.0",
      model: MODEL_ID,
      runId: crypto.randomUUID(),
      image_sha256: imageHash,
      vehicle: parsed.vehicle ?? { make: null, model: null, color: null, confidence: 0 },
      damage_items: itemsFinal,
      narrative: parsed.narrative ?? "",
      normalization_notes: parsed.normalization_notes ?? "",
      estimate,
      decision,
      damage_summary,
    };

    return NextResponse.json(payload);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "Server error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
