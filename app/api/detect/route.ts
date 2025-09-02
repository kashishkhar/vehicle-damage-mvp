// app/api/detect/route.ts
// --------------------------------------------------------------------------------------
// Car Damage Estimator — Detect Route (server)
// - Accepts file or URL, produces lightweight validation + YOLO boxes via Roboflow
// - Runs a fast OpenAI classifier for usability (vehicle present? quality OK?)
// - Returns a compact payload used by the client and the analyze step
// --------------------------------------------------------------------------------------

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";
import { DetectPayload, Vehicle, YoloBoxRel } from "../../types";

export const runtime = "nodejs";

const MODEL_VEHICLE = process.env.MODEL_VEHICLE || "gpt-4o-mini"; // small/fast classifier

// Roboflow config (optional; route tolerates empty keys)
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY || "";
const ROBOFLOW_MODEL = process.env.ROBOFLOW_MODEL || "";
const ROBOFLOW_VERSION = process.env.ROBOFLOW_VERSION || "";

/** ---------- Error helper (adds error_code, unchanged behavior for UI) ---------- */
const ERR = {
  NO_IMAGE: "E_NO_IMAGE",
  BAD_URL: "E_BAD_URL",
  SERVER: "E_SERVER",
} as const;

function errJson(message: string, code: string, status = 400) {
  return NextResponse.json({ error: message, error_code: code }, { status });
}

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
function isRecord(v: unknown): v is Record<string, unknown> { return typeof v === "object" && v !== null; }
function num(v: unknown): number | undefined { return typeof v === "number" ? v : undefined; }
function getNum(obj: Record<string, unknown>, key: string): number | undefined {
  return typeof obj[key] === "number" ? (obj[key] as number) : undefined;
}
function sha256(buf: Buffer) { return crypto.createHash("sha256").update(buf).digest("hex"); }
function sha256String(s: string) { return crypto.createHash("sha256").update(s, "utf8").digest("hex"); }

/** Extract Roboflow predictions, tolerating multiple shapes */
function extractPredictions(obj: unknown): unknown[] {
  if (!isRecord(obj)) return [];
  if (Array.isArray((obj as Record<string, unknown>).predictions)) return (obj as Record<string, unknown>).predictions as unknown[];
  if (isRecord((obj as Record<string, unknown>).result) && Array.isArray(((obj as Record<string, unknown>).result as Record<string, unknown>).predictions)) {
    return (((obj as Record<string, unknown>).result as Record<string, unknown>).predictions as unknown[]);
  }
  if (Array.isArray((obj as Record<string, unknown>).outputs)) return (obj as Record<string, unknown>).outputs as unknown[];
  return [];
}

// Call Roboflow hosted inference and convert to normalized boxes
async function detectWithRoboflow(imageUrlOrDataUrl: string): Promise<YoloBoxRel[]> {
  if (!ROBOFLOW_API_KEY || !ROBOFLOW_MODEL || !ROBOFLOW_VERSION) return [];

  try {
    const url = `https://detect.roboflow.com/${ROBOFLOW_MODEL}/${ROBOFLOW_VERSION}?api_key=${ROBOFLOW_API_KEY}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ image: imageUrlOrDataUrl }).toString(),
    });

    if (!res.ok) return [];
    const data: unknown = await res.json();
    const rawPreds = extractPredictions(data);
    const preds = rawPreds.filter(isRecord) as Record<string, unknown>[];
    const boxes: YoloBoxRel[] = [];

    for (const p of preds) {
      const conf = getNum(p, "confidence") ?? getNum(p, "conf") ?? 0.5;

      // Prefer x,y,w,h normalized center format if present
      if (
        getNum(p, "x") !== undefined &&
        getNum(p, "y") !== undefined &&
        getNum(p, "width") !== undefined &&
        getNum(p, "height") !== undefined &&
        getNum(p, "image_width") !== undefined &&
        getNum(p, "image_height") !== undefined
      ) {
        const cx = getNum(p, "x") as number;
        const cy = getNum(p, "y") as number;
        const w = getNum(p, "width") as number;
        const h = getNum(p, "height") as number;
        const W = getNum(p, "image_width") as number;
        const H = getNum(p, "image_height") as number;

        const nx = Math.max(0, Math.min(1, (cx - w / 2) / W));
        const ny = Math.max(0, Math.min(1, (cy - h / 2) / H));
        const nw = Math.max(0, Math.min(1, w / W));
        const nh = Math.max(0, Math.min(1, h / H));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf! });
        continue;
      }

      // Or x_min,x_max,y_min,y_max (already normalized)
      if (
        getNum(p, "x_min") !== undefined &&
        getNum(p, "x_max") !== undefined &&
        getNum(p, "y_min") !== undefined &&
        getNum(p, "y_max") !== undefined
      ) {
        const xmin = getNum(p, "x_min") as number;
        const ymin = getNum(p, "y_min") as number;
        const xmax = getNum(p, "x_max") as number;
        const ymax = getNum(p, "y_max") as number;
        const nx = Math.max(0, Math.min(1, xmin));
        const ny = Math.max(0, Math.min(1, ymin));
        const nw = Math.max(0, Math.min(1, xmax - xmin));
        const nh = Math.max(0, Math.min(1, ymax - ymin));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf! });
      }
    }

    return boxes;
  } catch {
    return [];
  }
}

// Fast quality/vehicle gate via OpenAI (cost-efficient, deterministic)
async function quickQualityGate(imageUrlOrDataUrl: string): Promise<{
  is_vehicle: boolean;
  quality_ok: boolean;
  issues: string[];
  vehicle_guess: Vehicle;
}> {
  const system = `
Return ONLY JSON with this shape:

{
  "is_vehicle": boolean,
  "quality_ok": boolean,
  "issues": string[],
  "vehicle": { "make": string|null, "model": string|null, "color": string|null, "confidence": number }
}

Rules:
- "issues" can include: "not_vehicle", "blurry", "low_light", "heavy_occlusion", "cropped", "too_small".
- If unsure about make/model/color, set null but always provide numeric "confidence" for vehicle recognition [0..1].
- Be conservative; keep output terse and valid JSON only.
`.trim();

  const client = getOpenAI();
  const completion = await client.chat.completions.create({
    model: MODEL_VEHICLE,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: [
          { type: "text", text: "Classify whether this is a usable car image for damage assessment." },
          { type: "image_url", image_url: { url: imageUrlOrDataUrl } },
        ],
      },
    ],
  });

  let parsed: unknown = {};
  try {
    parsed = JSON.parse(completion.choices?.[0]?.message?.content ?? "{}");
  } catch {
    parsed = {
      is_vehicle: true,
      quality_ok: true,
      issues: [],
      vehicle: { make: null, model: null, color: null, confidence: 0.6 },
    };
  }

  const pv = (typeof parsed === "object" && parsed !== null ? parsed : {}) as Record<string, unknown>;
  const vehicleObj = (pv["vehicle"] && typeof pv["vehicle"] === "object" ? (pv["vehicle"] as Record<string, unknown>) : {}) as Record<string, unknown>;
  const vehicle_guess: Vehicle = {
    make: typeof vehicleObj["make"] === "string" ? (vehicleObj["make"] as string) : null,
    model: typeof vehicleObj["model"] === "string" ? (vehicleObj["model"] as string) : null,
    color: typeof vehicleObj["color"] === "string" ? (vehicleObj["color"] as string) : null,
    confidence: typeof vehicleObj["confidence"] === "number" ? (vehicleObj["confidence"] as number) : 0.6,
  };

  return {
    is_vehicle: typeof pv["is_vehicle"] === "boolean" ? (pv["is_vehicle"] as boolean) : true,
    quality_ok: typeof pv["quality_ok"] === "boolean" ? (pv["quality_ok"] as boolean) : true,
    issues: Array.isArray(pv["issues"]) ? (pv["issues"] as unknown[]).map(String) : [],
    vehicle_guess,
  };
}

export async function POST(req: NextRequest) {
  console.time("detect_total");
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    if (!file && !imageUrl) {
      return errJson("No file or imageUrl provided", ERR.NO_IMAGE, 400);
    }

    // Build source + audit hash
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

    // YOLO (Roboflow) boxes + quick gate
    const [yolo_boxes, q] = await Promise.all([
      detectWithRoboflow(imageSourceUrl),
      quickQualityGate(imageSourceUrl),
    ]);

    const payload: DetectPayload = {
      model: MODEL_VEHICLE,
      runId: crypto.randomUUID(),
      image_sha256: imageHash,
      yolo_boxes,
      vehicle: q.vehicle_guess,
      is_vehicle: q.is_vehicle,
      has_damage: yolo_boxes.length > 0,
      quality_ok: q.quality_ok,
      issues: q.issues,
    };

    console.timeEnd("detect_total");
    return NextResponse.json(payload);
  } catch (e: unknown) {
    console.timeEnd("detect_total");
    const message = e instanceof Error ? e.message : "detect error";
    return errJson(message, ERR.SERVER, 500);
  }
}