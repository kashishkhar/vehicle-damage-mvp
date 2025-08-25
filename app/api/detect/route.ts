// app/api/detect/route.ts
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import crypto from "crypto";

export const runtime = "nodejs";

type YoloBox = { bbox_rel: [number, number, number, number]; confidence: number };

type Vehicle = {
  make: string | null;
  model: string | null;
  color: string | null;
  confidence: number;
};

type DetectPayload = {
  model: string;              // labeler model string (for audit)
  runId: string;
  image_sha256: string;
  yolo_boxes: YoloBox[];      // normalized [0..1]
  vehicle: Vehicle;           // light vehicle guess (optional, we fill minimal)
  // Validation flags:
  is_vehicle: boolean;
  has_damage: boolean;
  quality_ok: boolean;
  issues: string[];           // machine-ish labels e.g., ["not_vehicle", "blurry", "low_light", "no_damage"]
};

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
const MODEL_VEHICLE = process.env.MODEL_VEHICLE || "gpt-4o-mini"; // small/fast classifier

// Roboflow config (put these in your .env)
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY || "";
const ROBOFLOW_MODEL = process.env.ROBOFLOW_MODEL || "";
const ROBOFLOW_VERSION = process.env.ROBOFLOW_VERSION || "";

// --- utils ---
function sha256(buf: Buffer) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}
function sha256String(s: string) {
  return crypto.createHash("sha256").update(s, "utf8").digest("hex");
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}
function num(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}

// Extract Roboflow predictions flexibly
function extractPredictions(obj: unknown): any[] {
  if (!isRecord(obj)) return [];
  if (Array.isArray(obj.predictions)) return obj.predictions;
  if (isRecord(obj.result) && Array.isArray((obj.result as any).predictions)) return (obj.result as any).predictions;
  if (Array.isArray(obj.outputs)) return obj.outputs;
  return [];
}

async function detectWithRoboflow(imageUrlOrDataUrl: string): Promise<YoloBox[]> {
  if (!ROBOFLOW_API_KEY || !ROBOFLOW_MODEL || !ROBOFLOW_VERSION) return [];

  try {
    const url = `https://detect.roboflow.com/${ROBOFLOW_MODEL}/${ROBOFLOW_VERSION}?api_key=${ROBOFLOW_API_KEY}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      // Roboflow expects "image=<URL or base64 data URL>"
      body: new URLSearchParams({ image: imageUrlOrDataUrl }).toString(),
    });

    if (!res.ok) return [];
    const data: unknown = await res.json();
    const preds = extractPredictions(data);
    const boxes: YoloBox[] = [];

    for (const p of preds) {
      const conf = num((p as any).confidence) ?? num((p as any).conf) ?? 0.5;

      // Prefer x,y,w,h normalized center format if present
      if (
        num((p as any).x) !== undefined &&
        num((p as any).y) !== undefined &&
        num((p as any).width) !== undefined &&
        num((p as any).height) !== undefined &&
        num((p as any).image_width) !== undefined &&
        num((p as any).image_height) !== undefined
      ) {
        const cx = (p as any).x as number;
        const cy = (p as any).y as number;
        const w = (p as any).width as number;
        const h = (p as any).height as number;
        const W = (p as any).image_width as number;
        const H = (p as any).image_height as number;

        const nx = Math.max(0, Math.min(1, (cx - w / 2) / W));
        const ny = Math.max(0, Math.min(1, (cy - h / 2) / H));
        const nw = Math.max(0, Math.min(1, w / W));
        const nh = Math.max(0, Math.min(1, h / H));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf });
        continue;
      }

      // Or x_min,x_max,y_min,y_max (already normalized)
      if (
        num((p as any).x_min) !== undefined &&
        num((p as any).x_max) !== undefined &&
        num((p as any).y_min) !== undefined &&
        num((p as any).y_max) !== undefined
      ) {
        const xmin = (p as any).x_min as number;
        const ymin = (p as any).y_min as number;
        const xmax = (p as any).x_max as number;
        const ymax = (p as any).y_max as number;
        const nx = Math.max(0, Math.min(1, xmin));
        const ny = Math.max(0, Math.min(1, ymin));
        const nw = Math.max(0, Math.min(1, xmax - xmin));
        const nh = Math.max(0, Math.min(1, ymax - ymin));
        boxes.push({ bbox_rel: [nx, ny, nw, nh], confidence: conf });
      }
    }

    return boxes;
  } catch {
    return [];
  }
}

// Very small, cheap quality gate (vehicle? usable quality?)
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

  let parsed: any = {};
  try {
    parsed = JSON.parse(completion.choices?.[0]?.message?.content ?? "{}");
  } catch {
    // fall back: unknown vehicle/quality
    parsed = {
      is_vehicle: true,
      quality_ok: true,
      issues: [],
      vehicle: { make: null, model: null, color: null, confidence: 0.6 },
    };
  }

  const vehicle_guess: Vehicle = {
    make: parsed?.vehicle?.make ?? null,
    model: parsed?.vehicle?.model ?? null,
    color: parsed?.vehicle?.color ?? null,
    confidence: typeof parsed?.vehicle?.confidence === "number" ? parsed.vehicle.confidence : 0.6,
  };

  return {
    is_vehicle: Boolean(parsed?.is_vehicle),
    quality_ok: Boolean(parsed?.quality_ok),
    issues: Array.isArray(parsed?.issues) ? parsed.issues : [],
    vehicle_guess,
  };
}

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    const imageUrl = (form.get("imageUrl") as string | null)?.toString().trim() || null;

    if (!file && !imageUrl) {
      return NextResponse.json({ error: "No file or imageUrl provided" }, { status: 400 });
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
          return NextResponse.json({ error: "imageUrl must be http(s)" }, { status: 400 });
        }
      } catch {
        return NextResponse.json({ error: "Invalid imageUrl" }, { status: 400 });
      }
      imageSourceUrl = imageUrl!;
      imageHash = sha256String(imageUrl!);
    }

    // YOLO (Roboflow) boxes
    const yolo_boxes = await detectWithRoboflow(imageSourceUrl);

    // Quick quality/vehicle gate
    const q = await quickQualityGate(imageSourceUrl);

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

    return NextResponse.json(payload);
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "detect error" }, { status: 500 });
  }
}
