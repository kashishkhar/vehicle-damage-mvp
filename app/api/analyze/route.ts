// app/api/analyze/route.ts
// --------------------------------------------------------------------------------------
// Car Damage Estimator — Analyze Route (server)
// - Accepts an uploaded image (file) or a public image URL
// - Calls OpenAI Vision to produce structured damage JSON
// - Normalizes/repairs the JSON, computes estimate + routing, and returns payload
// --------------------------------------------------------------------------------------

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";
import {
  AnalyzePayload,
  DamageItem,
  Vehicle,
  Part,
  DetectPayload,
} from "../../types";

export const runtime = "nodejs";

/** ---------- Error helper (adds error_code, unchanged UI behavior) ---------- */
const ERR = {
  NO_IMAGE: "E_NO_IMAGE",
  BAD_URL: "E_BAD_URL",
  MODEL_JSON: "E_MODEL_JSON",
  SERVER: "E_SERVER",
} as const;

function errJson(message: string, code: string, status = 400) {
  return NextResponse.json({ error: message, error_code: code }, { status });
}

/** ---------- Env & Constants ---------- */
const MODEL_ID = process.env.MODEL_VISION || "gpt-4o-mini";

const LABOR = Number(process.env.LABOR_RATE ?? 95);
const PAINT = Number(process.env.PAINT_MAT_COST ?? 180);

const AUTO_MAX_COST = Number(process.env.AUTO_MAX_COST ?? 1500);
const AUTO_MAX_SEVERITY = Number(process.env.AUTO_MAX_SEVERITY ?? 2);
const AUTO_MIN_CONF = Number(process.env.AUTO_MIN_CONF ?? 0.75);
const SPEC_MIN_COST = Number(process.env.SPECIALIST_MIN_COST ?? 5000);
const SPEC_MIN_SEVERITY = Number(process.env.SPECIALIST_MIN_SEVERITY ?? 4);

/** ---------- OpenAI client (lazy) ---------- */
let _openai: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (_openai) return _openai;
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY environment variable is missing or empty");
  _openai = new OpenAI({ apiKey: key });
  return _openai;
}

/** ---------- Utils ---------- */
function sha256(buf: Buffer) { return crypto.createHash("sha256").update(buf).digest("hex"); }
function sha256String(s: string) { return crypto.createHash("sha256").update(s, "utf8").digest("hex"); }
function isRecord(v: unknown): v is Record<string, unknown> { return typeof v === "object" && v !== null; }

/** ---- Local tuple guards (unchanged) ---- */
type BBoxRel = [number, number, number, number];
type PolyRel = [number, number][];

function isBBoxRel(v: unknown): v is BBoxRel {
  return Array.isArray(v) && v.length === 4 && v.every(n => typeof n === "number" && n >= 0 && n <= 1);
}
function isPolygonRel(v: unknown): v is PolyRel {
  return Array.isArray(v)
    && v.length >= 3 && v.length <= 12
    && v.every(pt =>
      Array.isArray(pt)
      && pt.length === 2
      && typeof pt[0] === "number" && pt[0] >= 0 && pt[0] <= 1
      && typeof pt[1] === "number" && pt[1] >= 0 && pt[1] <= 1
    );
}

type YoloSeed = { bbox_rel: BBoxRel; confidence: number };
function isYoloSeed(v: unknown): v is YoloSeed {
  if (!isRecord(v)) return false;
  return typeof v["confidence"] === "number" && isBBoxRel(v["bbox_rel"]);
}

function getNum(obj: Record<string, unknown>, key: string): number | undefined {
  return typeof obj[key] === "number" ? (obj[key] as number) : undefined;
}

// Approximate bodyshop labor hours per part × severity multiplier (unchanged)
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

// Paint heuristic (unchanged)
function needsPaintFor(type: string, sev: number, part: Part) {
  if (["windshield", "headlight", "taillight", "mirror"].includes(part)) return false;
  if (type.includes("scratch") || type.includes("paint")) return true;
  return sev >= 2;
}

// Roll up labor/paint/parts → cost band with variance (unchanged)
function estimateFromItems(items: DamageItem[]) {
  let labor = 0, paint = 0, parts = 0;
  const paintedZones = new Set<string>();

  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const hrs = typeof d.est_labor_hours === "number" ? d.est_labor_hours : hoursFor(d.part, sev);
    labor += hrs * LABOR;

    const needPaint = typeof d.needs_paint === "boolean" ? d.needs_paint : needsPaintFor(String(d.damage_type || ""), sev, d.part);
    if (needPaint && (d.zone as string) && !paintedZones.has(d.zone as string)) {
      paint += PAINT;
      paintedZones.add(d.zone as string);
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

// Weighted confidence (unchanged)
function aggregateConfidence(items: DamageItem[]) {
  let numr = 0, den = 0;
  for (const d of items) {
    const sev = Number(d.severity ?? 1);
    const conf = Number(d.confidence ?? 0.5);
    const w = 1 + 0.2 * (sev - 1);
    numr += conf * w;
    den += w;
  }
  return den ? numr / den : 0.5;
}

// Policy routing (unchanged)
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

/** ---------- Handler ---------- */
export async function POST(req: NextRequest) {
  console.time("analyze_total");
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    // Optional YOLO seeds passed from client
    const yoloSeedsRaw = (form.get("yolo") as string | null)?.toString() || "";
    let yoloSeeds: { bbox_rel: [number,number,number,number]; confidence: number }[] = [];
    if (yoloSeedsRaw) {
      try {
        const parsed = JSON.parse(yoloSeedsRaw) as unknown;
        if (Array.isArray(parsed)) {
          yoloSeeds = (parsed as unknown[]).filter(isYoloSeed) as YoloSeed[];
        }
      } catch { /* tolerate malformed seeds */ }
    }

    if (!file && !imageUrl) {
      return errJson("No file or imageUrl provided", ERR.NO_IMAGE, 400);
    }

    // Construct an OpenAI image source and an audit hash
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
          return errJson("imageUrl must be http(s)", ERR.BAD_URL, 400);
        }
      } catch {
        return errJson("Invalid imageUrl", ERR.BAD_URL, 400);
      }
      imageSourceUrl = imageUrl!;
      imageHash = sha256String(imageUrl!);
    }

    // Strict JSON schema; the model refines labels using YOLO seeds if provided
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

    const client = getOpenAI();
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
    let parsed: unknown;
    try { parsed = JSON.parse(raw); } catch {
      return errJson("Model returned invalid JSON", ERR.MODEL_JSON, 502);
    }

    // Normalize/repair items (unchanged)
    const itemsIn = isRecord(parsed) && Array.isArray(parsed["damage_items"])
      ? (parsed["damage_items"] as unknown[])
      : [];
    const items: DamageItem[] = itemsIn.map((dIn) => {
      const d = isRecord(dIn) ? dIn : {};
      const sevRaw = d["severity"];
      const sev = (typeof sevRaw === "number" && sevRaw >= 1 && sevRaw <= 5 ? sevRaw : 2) as 1|2|3|4|5;
      const p = (typeof d["part"] === "string" ? d["part"] : "unknown") as Part;
      const estRaw = d["est_labor_hours"];
      const ht = typeof estRaw === "number" ? estRaw : hoursFor(p, sev);
      const needsRaw = d["needs_paint"];
      const dmgType = (typeof d["damage_type"] === "string" ? d["damage_type"] : "unknown");
      const paint = typeof needsRaw === "boolean" ? needsRaw : needsPaintFor(String(dmgType || ""), sev, p);
      const confRaw = d["confidence"];
      const conf = typeof confRaw === "number" ? Math.max(0, Math.min(1, confRaw)) : 0.5;

      const out: DamageItem = {
        zone: (typeof d["zone"] === "string" ? d["zone"] : "unknown") as DamageItem["zone"],
        part: p,
        damage_type: dmgType as DamageItem["damage_type"],
        severity: sev,
        confidence: conf,
        est_labor_hours: ht,
        needs_paint: paint,
        likely_parts: Array.isArray(d["likely_parts"]) ? (d["likely_parts"] as unknown[]).map(String) : [],
      };

      const bb = d["bbox_rel"];
      if (isBBoxRel(bb)) out.bbox_rel = [bb[0], bb[1], bb[2], bb[3]];
      const poly = d["polygon_rel"];
      if (isPolygonRel(poly)) out.polygon_rel = (poly as PolyRel).map((pt) => [pt[0], pt[1]]);
      return out;
    });

    // Preserve UI behavior re: YOLO fallback
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

    const narrative = isRecord(parsed) && typeof parsed["narrative"] === "string" ? (parsed["narrative"] as string) : "";
    const normalization_notes =
      isRecord(parsed) && typeof parsed["normalization_notes"] === "string"
        ? (parsed["normalization_notes"] as string)
        : "";

    const vehicleRec = isRecord(parsed) && isRecord(parsed["vehicle"]) ? (parsed["vehicle"] as Record<string, unknown>) : {};
    const vehicle: Vehicle = {
      make: typeof vehicleRec["make"] === "string" ? (vehicleRec["make"] as string) : null,
      model: typeof vehicleRec["model"] === "string" ? (vehicleRec["model"] as string) : null,
      color: typeof vehicleRec["color"] === "string" ? (vehicleRec["color"] as string) : null,
      confidence: typeof vehicleRec["confidence"] === "number" ? (vehicleRec["confidence"] as number) : 0,
    };

    const damage_summary = (itemsFinal.length
      ? itemsFinal.map((d) => `${d.zone} ${d.part} — ${d.damage_type}, sev ${d.severity}`).join("; ")
      : narrative
    ).slice(0, 400);

    const payload: AnalyzePayload = {
      schema_version: "1.3.0",
      model: MODEL_ID,
      runId: crypto.randomUUID(),
      image_sha256: (typeof imageHash === "string" ? imageHash : String(imageHash)) as string,
      vehicle,
      damage_items: itemsFinal,
      narrative,
      normalization_notes,
      estimate,
      decision,
      damage_summary,
    };

    console.timeEnd("analyze_total");
    return NextResponse.json(payload);
  } catch (e: unknown) {
    console.timeEnd("analyze_total");
    const msg = e instanceof Error ? e.message : "Server error";
    return errJson(msg, ERR.SERVER, 500);
  }
}