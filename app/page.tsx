/* eslint-disable @next/next/no-img-element */
"use client";

/**
 * Car Damage Estimator – Client UI
 * ---------------------------------
 * ✔ Upload image or paste URL, then runs /api/detect → /api/analyze
 * ✔ Draws damage overlays on the preview (labels only show part names)
 * ✔ Human-friendly error messages (bad URL, fetch errors, no vehicle, etc.)
 * ✔ Minimal, useful report: Routing chip + confidence, vehicle meta, damage table, cost band
 * ✔ Damage table with search/reset, paint + confidence filters (right-aligned),
 *   and explicit ▲/▼ sort controls for Severity and Confidence
 * ✔ Print-to-PDF via browser (Export Report PDF)
 * ✔ Audit metadata gated behind a checkbox
 *
 * Note: This is an MVP. Code is kept simple, predictable, and easy to reason about.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalyzePayload, DamageItem, DetectPayload } from "./types";

/** ──────────────────────────────────────────────────────────────────────────
 *  UI Helpers
 *  - Confidence bands + formatting
 *  - Friendly error mapping
 *  - Narrative guardrails (no vehicle present)
 *  - Small cost estimation math helpers
 *  ------------------------------------------------------------------------- */

const CONF_HIGH = Number(process.env.NEXT_PUBLIC_CONF_HIGH ?? 0.85);
const CONF_MED = Number(process.env.NEXT_PUBLIC_CONF_MED ?? 0.6);

function confidenceBand(p?: number) {
  if (typeof p !== "number") return "Unknown";
  if (p >= CONF_HIGH) return "High";
  if (p >= CONF_MED) return "Medium";
  return "Low";
}

function fmtPct(p?: number) {
  return typeof p === "number" ? `${Math.round(p * 100)}%` : "—";
}

/**
 * Friendly mapping for raw API/edge errors coming back from /api/detect or /api/analyze
 */
function friendlyApiError(raw: string, status?: number) {
  const lower = (raw || "").toLowerCase();
  if (lower.includes("downloading") || lower.includes("fetch") || lower.includes("http")) {
    return "We couldn’t fetch that link. Make sure it’s a direct image URL (JPG/PNG/WebP) and publicly accessible, then try again.";
  }
  if (status && status >= 500) {
    return "Our service hit a hiccup while processing the image. Please try again in a moment.";
  }
  return "We couldn’t process that image or link. Try a different photo or a direct image URL.";
}

/**
 * Detect common LLM narratives that imply “no vehicle in image”.
 */
function narrativeIndicatesNoVehicle(text: string | undefined | null) {
  if (!text) return false;
  const t = String(text).toLowerCase();
  return (
    t.includes("no vehicle present") ||
    t.includes("no vehicle image") ||
    t.includes("no vehicle detected") ||
    t.includes("no vehicle found") ||
    t.includes("not a vehicle") ||
    t.includes("image does not contain a vehicle")
  );
}

/** Weighted aggregation (mirrors server) to show an overall decision confidence. */
function aggregateDecisionConfidence(items: DamageItem[]): number {
  if (!Array.isArray(items) || !items.length) return 0.5;
  let num = 0,
    den = 0;
  for (const d of items) {
    const sev = Number(d?.severity ?? 1);
    const conf = Number(d?.confidence ?? 0.5);
    const w = 1 + 0.2 * (sev - 1);
    num += conf * w;
    den += w;
  }
  return den ? num / den : 0.5;
}

/** ──────────────────────────────────────────────────────────────────────────
 *  Image utilities (client-only)
 *  ------------------------------------------------------------------------- */

/** Compress large images in-browser to keep uploads snappy. */
async function compressImage(file: File, maxW = 1600, quality = 0.72): Promise<File> {
  const img = document.createElement("img");
  const reader = new FileReader();
  const load = new Promise<void>((resolve, reject) => {
    reader.onload = () => {
      img.src = reader.result as string;
      img.onload = () => resolve();
      img.onerror = reject;
    };
    reader.onerror = reject;
  });
  reader.readAsDataURL(file);
  await load;
  const scale = Math.min(1, maxW / img.width);
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);
  const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b!), "image/jpeg", quality));
  return new File([blob], "upload.jpg", { type: "image/jpeg" });
}

/** Convert a File to a data URL — handy for audit/prompt pass-through. */
async function fileToDataUrl(file: File): Promise<string> {
  const reader = new FileReader();
  const p = new Promise<string>((resolve, reject) => {
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
  });
  reader.readAsDataURL(file);
  return p;
}

/** ──────────────────────────────────────────────────────────────────────────
 *  Overlay drawing
 *  ------------------------------------------------------------------------- */

function colorForSeverity(sev: number) {
  if (sev >= 5) return "#dc2626"; // red-600
  if (sev === 4) return "#f97316"; // orange-500
  if (sev === 3) return "#eab308"; // yellow-500
  if (sev === 2) return "#22c55e"; // green-500
  return "#10b981"; // emerald-500
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string
) {
  if (!text) return;
  ctx.save();
  ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  const pad = 3;
  const metrics = ctx.measureText(text);
  const w = metrics.width + pad * 2,
    h = 16;
  ctx.fillRect(x - w / 2, y - 2, w, h);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

/** Canvas overlay for normalized boxes/polygons. */
function CanvasOverlay({
  imgRef,
  items,
  show,
}: {
  imgRef: React.RefObject<HTMLImageElement>;
  items: DamageItem[];
  show: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const rect = img.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!show) return;

    ctx.scale(dpr, dpr);
    ctx.lineWidth = 2;

    items.forEach((d) => {
      const sev = Number(d.severity ?? 1);
      const color = colorForSeverity(sev);
      ctx.strokeStyle = color;
      ctx.fillStyle = color + "33"; // 20% alpha

      if (Array.isArray(d.polygon_rel) && d.polygon_rel.length >= 3) {
        const pts = d.polygon_rel as [number, number][];
        ctx.beginPath();
        pts.forEach(([nx, ny], i) => {
          const x = nx * rect.width;
          const y = ny * rect.height;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        const cx = (pts.reduce((s, p) => s + p[0], 0) / pts.length) * rect.width;
        const cy = (pts.reduce((s, p) => s + p[1], 0) / pts.length) * rect.height;
        drawLabel(ctx, String(d.part ?? ""), cx, cy, color);
      } else if (Array.isArray(d.bbox_rel) && d.bbox_rel.length === 4) {
        const [nx, ny, nw, nh] = d.bbox_rel as [number, number, number, number];
        const x = nx * rect.width,
          y = ny * rect.height,
          w = nw * rect.width,
          h = nh * rect.height;
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.fill();
        ctx.stroke();
        drawLabel(ctx, String(d.part ?? ""), x + w / 2, y + 14, color);
      }
    });

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [imgRef, items, show]);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" aria-hidden="true" />;
}

/** ──────────────────────────────────────────────────────────────────────────
 *  Small presentational primitives
 *  ------------------------------------------------------------------------- */

function Checkbox({
  checked,
  onChange,
  label,
  id,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  id: string;
}) {
  return (
    <label htmlFor={id} className="flex items-center gap-2 text-sm text-neutral-800 select-none">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 accent-black"
      />
      {label}
    </label>
  );
}

/** Build a concise, professional English damage summary. */
function buildDamageSummary(result: AnalyzePayload): string {
  const items: DamageItem[] = Array.isArray(result?.damage_items) ? result.damage_items : [];
  if (!items.length) return result?.narrative || "No visible damage detected.";

  const phrases: string[] = items.map((d) => {
    let sevText = "minor";
    if (d.severity >= 5) sevText = "severe";
    else if (d.severity === 4) sevText = "major";
    else if (d.severity === 3) sevText = "moderate";

    const typeMap: Record<string, string> = {
      dent: "dent",
      scratch: "surface scratch",
      crack: "structural crack",
      "paint-chips": "paint chipping",
      broken: "broken component",
      bent: "bent panel",
      missing: "missing component",
      "glass-crack": "glass fracture",
      unknown: "unspecified damage",
    };
    const typeText = typeMap[d.damage_type] || d.damage_type;

    const zonePart = [d.zone, d.part].filter(Boolean).join(" ");
    let phrase = `${sevText} ${typeText} on the ${zonePart}`;
    if (d.needs_paint) phrase += ` requiring repainting`;
    if (Array.isArray(d.likely_parts) && d.likely_parts.length) {
      phrase += ` with possible replacement of ${d.likely_parts.join(", ")}`;
    }
    return phrase;
  });

  const joined =
    phrases.length > 1 ? phrases.slice(0, -1).join("; ") + ", and " + phrases.slice(-1) : phrases[0];

  return `The inspection identified ${joined}. Based on the detected severity levels, professional repair work is recommended to restore the vehicle to safe operating condition.`;
}

/** ──────────────────────────────────────────────────────────────────────────
 *  Sorting controls (explicit ▲/▼)
 *  ------------------------------------------------------------------------- */

type SortKey = "severity" | "confidence";
type SortDir = "asc" | "desc";

type SortHeaderProps = {
  label: string;
  active: boolean;
  dir: SortDir;
  onAsc: () => void;
  onDesc: () => void;
  className?: string;
};

function SortHeader({ label, active, dir, onAsc, onDesc, className = "" }: SortHeaderProps) {
  const baseBtn = "px-1.5 py-0.5 rounded border text-[11px]";
  const activeCls = "border-neutral-800 text-neutral-900";
  const idleCls = "border-neutral-300 text-neutral-600 hover:text-neutral-800";
  return (
    <th className={`p-2 text-left select-none ${className}`}>
      <div className="flex items-center gap-2">
        <span className="text-sm text-neutral-700">{label}</span>
        <div className="flex items-center gap-1">
          {/* Order requested: down-facing then up-facing */}
          <button
            type="button"
            onClick={onDesc}
            aria-label={`Sort ${label} descending`}
            title="Sort descending"
            className={`${baseBtn} ${active && dir === "desc" ? activeCls : idleCls}`}
          >
            ▼
          </button>
          <button
            type="button"
            onClick={onAsc}
            aria-label={`Sort ${label} ascending`}
            title="Sort ascending"
            className={`${baseBtn} ${active && dir === "asc" ? activeCls : idleCls}`}
          >
            ▲
          </button>
        </div>
      </div>
    </th>
  );
}

function DamageTable(props: {
  rows: DamageItem[];
  sortKey: SortKey | null;
  sortDir: SortDir;
  onSetSort: (k: SortKey, dir: SortDir) => void;
  fPaint: "all" | "yes" | "no";
  setFPaint: (v: "all" | "yes" | "no") => void;
  fConfMin: string;
  setFConfMin: (v: string) => void;
  fSearch: string;
  setFSearch: (v: string) => void;
  onResetFilters: () => void;
}) {
  const {
    rows,
    sortKey,
    sortDir,
    onSetSort,
    fPaint,
    setFPaint,
    fConfMin,
    setFConfMin,
    fSearch,
    setFSearch,
    onResetFilters,
  } = props;

  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm">
      {/* Header: search + reset */}
      <div className="mb-1 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium">Detected Damage</div>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={fSearch}
            onChange={(e) => setFSearch(e.target.value)}
            placeholder="Search zone/part/type…"
            className="w-40 rounded border border-neutral-200 bg-white px-2 py-1 text-xs"
          />
          <button
            type="button"
            onClick={onResetFilters}
            className="rounded border border-neutral-300 bg-white px-2 py-1 text-xs"
            title="Reset filters"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Filters row: right side under search/reset */}
      <div className="mb-3 flex items-center justify-end gap-2">
        <select
          value={fPaint}
          onChange={(e) => setFPaint(e.target.value as "all" | "yes" | "no")}
          className="w-28 rounded border border-neutral-200 bg-white px-2 py-1 text-xs"
        >
          <option value="all">Paint: All</option>
          <option value="yes">Paint: Yes</option>
          <option value="no">Paint: No</option>
        </select>
        <input
          value={fConfMin}
          onChange={(e) => setFConfMin(e.target.value)}
          placeholder="Conf ≥ %"
          inputMode="numeric"
          className="w-20 rounded border border-neutral-200 bg-white px-2 py-1 text-xs"
        />
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-[13px] leading-5 border-collapse">
          <thead className="bg-neutral-50 text-neutral-700">
            <tr>
              {/* Narrower first three, wider Conf */}
              <th className="p-2 text-left w-24">Zone</th>
              <th className="p-2 text-left w-24">Part</th>
              <th className="p-2 text-left w-28">Type</th>
              <SortHeader
                label="Sev"
                active={sortKey === "severity"}
                dir={sortDir || "asc"}
                onAsc={() => onSetSort("severity", "asc")}
                onDesc={() => onSetSort("severity", "desc")}
                className="w-20"
              />
              <th className="p-2 text-left w-16">Paint</th>
              <SortHeader
                label="Conf"
                active={sortKey === "confidence"}
                dir={sortDir || "asc"}
                onAsc={() => onSetSort("confidence", "asc")}
                onDesc={() => onSetSort("confidence", "desc")}
                className="w-32"
              />
            </tr>
          </thead>
          <tbody>
            {rows.map((d, i) => (
              <tr key={i} className="border-b last:border-none align-top">
                <td className="p-2">{d.zone}</td>
                <td className="p-2">{d.part}</td>
                <td className="p-2">{d.damage_type}</td>
                <td className="p-2">{d.severity}</td>
                <td className="p-2">{d.needs_paint ? "Yes" : "No"}</td>
                <td className="p-2">{fmtPct(d.confidence)} ({confidenceBand(d.confidence)})</td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={6} className="p-3 text-center text-sm text-neutral-500">
                  No rows match your filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** ──────────────────────────────────────────────────────────────────────────
 *  Main Page Component
 *  ------------------------------------------------------------------------- */

export default function Home() {
  // Mode & Inputs
  const [mode, setMode] = useState<"upload" | "url">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [preview, setPreview] = useState<string>("");

  // Network & result state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzePayload | null>(null);
  const [error, setError] = useState<string>("");
  const [validationIssues, setValidationIssues] = useState<string[] | null>(null);

  // Toggles
  const [showOverlay, setShowOverlay] = useState(true);
  const [showAudit, setShowAudit] = useState(false);

  // Refs
  const imgRef = useRef<HTMLImageElement>(null);

  /** Switch input mode (Upload vs URL). */
  const switchMode = useCallback((next: "upload" | "url") => {
    setMode(next);
    setResult(null);
    setError("");
    setValidationIssues(null);
    setPreview("");
    if (next === "upload") setImageUrl("");
    else setFile(null);
  }, []);

  /** Local file selected. */
  const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setResult(null);
    setError("");
    setValidationIssues(null);
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : "");
  }, []);

  /** URL pasted/edited. */
  const onUrlChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setImageUrl(val);
    setResult(null);
    setError("");
    setValidationIssues(null);
    setPreview(val || "");
  }, []);

  /** Submit → Detect → Analyze (with no-vehicle guardrails) */
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setValidationIssues(null);
    setResult(null);

    try {
      const analyzeForm = new FormData();
      const detectForm = new FormData();

      if (mode === "upload") {
        if (!file) {
          setError("Please choose a photo to analyze.");
          return;
        }
        const compressed = await compressImage(file);
        const dataUrl = await fileToDataUrl(compressed);
        detectForm.append("file", compressed);
        analyzeForm.append("file", compressed);
        analyzeForm.append("image_data_url", dataUrl);
      } else {
        const raw = imageUrl.trim();
        if (!raw) {
          setError("Please paste an image link to analyze.");
          return;
        }
        detectForm.append("imageUrl", raw);
        analyzeForm.append("imageUrl", raw);
      }

      // 1) DETECT
      const dr = await fetch("/api/detect", { method: "POST", body: detectForm });
      if (!dr.ok) {
        const t = await dr.text();
        setError(friendlyApiError(t, dr.status));
        return;
      }
      const detectJson: DetectPayload = await dr.json();

      // Early out if quick gate says no vehicle.
      if (detectJson && detectJson.is_vehicle === false) {
        setError("We couldn’t detect a vehicle in that image. Please upload a photo that clearly shows a vehicle.");
        return;
      }

      // 2) ANALYZE – pass YOLO boxes as seeds if present
      const seeds = Array.isArray(detectJson?.yolo_boxes)
        ? detectJson.yolo_boxes
            .filter((b) => Array.isArray(b.bbox_rel) && b.bbox_rel.length === 4)
            .map((b) => ({ bbox_rel: b.bbox_rel, confidence: b.confidence ?? 0.5 }))
        : [];
      analyzeForm.append("yolo", JSON.stringify(seeds));

      const ar = await fetch("/api/analyze", { method: "POST", body: analyzeForm });
      if (!ar.ok) {
        const t = await ar.text();
        setError(friendlyApiError(t, ar.status));
        return;
      }
      const j: AnalyzePayload = await ar.json();

      // Guard against LLM saying "no vehicle" in narrative or extremely low vehicle confidence with no items.
      if (
        narrativeIndicatesNoVehicle(j?.narrative) ||
        ((!Array.isArray(j?.damage_items) || j.damage_items.length === 0) &&
          narrativeIndicatesNoVehicle(String(j?.narrative || "")))
      ) {
        setResult(null);
        setError("We couldn’t detect a vehicle in that image. Please upload a photo that clearly shows a vehicle.");
        return;
      }

      if ((!Array.isArray(j?.damage_items) || j.damage_items.length === 0) && Number(j?.vehicle?.confidence ?? 0) < 0.15) {
        setResult(null);
        setError("We couldn’t detect a vehicle in that image. Please upload a photo that clearly shows a vehicle.");
        return;
      }

      setResult(j);

      // Surface quality hints from detect (non-blocking)
      if (Array.isArray(detectJson?.issues) && detectJson.issues.length) {
        setValidationIssues(detectJson.issues);
      }
    } catch {
      setError("We couldn’t process that image or link. Try a different photo or a direct image URL.");
    } finally {
      setLoading(false);
    }
  }, [file, imageUrl, mode]);

  // Derived memoized display bits
  const make = result?.vehicle?.make ?? "—";
  const model = result?.vehicle?.model ?? "—";
  const color = result?.vehicle?.color ?? "—";

  // Sorting & minimal filters
  const [sortKey, setSortKey] = useState<SortKey | null>(null); // default: original order
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [fPaint, setFPaint] = useState<"all" | "yes" | "no">("all");
  const [fConfMin, setFConfMin] = useState<string>("");
  const [fSearch, setFSearch] = useState<string>("");

  const setSort = useCallback((k: SortKey, dir: SortDir) => {
    setSortKey(k);
    setSortDir(dir);
  }, []);

  const resetFilters = useCallback(() => {
    setFPaint("all");
    setFConfMin("");
    setFSearch("");
  }, []);

  const filteredSorted: DamageItem[] = useMemo(() => {
    const items: DamageItem[] = Array.isArray(result?.damage_items) ? result.damage_items : [];
    const q = fSearch.trim().toLowerCase();
    const confMin = fConfMin ? Number(fConfMin) / 100 : undefined;

    const filtered = items.filter((d) => {
      const paintOk = fPaint === "all" || (fPaint === "yes" && Boolean(d.needs_paint)) || (fPaint === "no" && !Boolean(d.needs_paint));
      const confOk = confMin === undefined || Number(d.confidence ?? 0) >= confMin;
      const hay = `${d.zone} ${d.part} ${d.damage_type}`.toLowerCase();
      const searchOk = !q || hay.includes(q);
      return paintOk && confOk && searchOk;
    });

    if (!sortKey) return filtered; // preserve original order by default

    const dirMul = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = Number(a?.[sortKey] ?? 0);
      const bv = Number(b?.[sortKey] ?? 0);
      return (av - bv) * dirMul;
    });
  }, [result, fPaint, fConfMin, fSearch, sortKey, sortDir]);

  const decisionConf = useMemo(
    () => aggregateDecisionConfidence(Array.isArray(result?.damage_items) ? result.damage_items : []),
    [result]
  );

  return (
    <main className="min-h-screen bg-white text-neutral-900">
      {/* Print styles */}
      <style jsx global>{`
        @media print {
          @page {
            size: A4 portrait;
            margin: 0.5in;
          }
          html,
          body {
            background: #ffffff !important;
          }
        }
      `}</style>

      {/* Header */}
      <header className="border-b bg-white print:hidden">
        <div className="mx-auto max-w-7xl px-6 py-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-xl sm:text-2xl font-semibold">Car Damage Estimator</h1>
            <p className="text-sm text-neutral-600">
              Upload a vehicle photo <em>or paste an image URL</em> to generate a structured damage report, cost band, and routing decision.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => window.print()}
              disabled={!result}
              className="inline-flex items-center justify-center rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm font-medium text-neutral-800 disabled:opacity-50"
              aria-label="Export Report PDF"
              title={result ? "Export Report PDF" : "Run an analysis first"}
            >
              Export Report PDF
            </button>
          </div>
        </div>
      </header>

      {/* Layout */
      }
      <div className="mx-auto max-w-7xl px-6 py-6 grid gap-6 lg:grid-cols-12">
        {/* Left: Image column */}
        <section className="lg:col-span-4 print:hidden">
          <div className="lg:sticky lg:top-6 space-y-4">
            {/* Input Card */}
            <form onSubmit={handleSubmit} className="rounded-2xl border bg-white p-4 shadow-sm space-y-3">
              {/* Mode switch */}
              <div className="flex rounded-lg overflow-hidden border">
                <button
                  type="button"
                  onClick={() => switchMode("upload")}
                  className={`flex-1 px-3 py-2 text-sm ${mode === "upload" ? "bg-neutral-900 text-white" : "bg-white text-neutral-700"}`}
                >
                  Upload
                </button>
                <button
                  type="button"
                  onClick={() => switchMode("url")}
                  className={`flex-1 px-3 py-2 text-sm ${mode === "url" ? "bg-neutral-900 text-white" : "bg-white text-neutral-700"}`}
                >
                  URL
                </button>
              </div>

              {/* Input */}
              {mode === "upload" ? (
                <label className="text-sm font-medium block">
                  <span className="sr-only">Upload image</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={onFile}
                    className="block w-full rounded border border-neutral-200 bg-white px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-neutral-100 file:px-3 file:py-2 file:text-sm hover:file:bg-neutral-200"
                  />
                </label>
              ) : (
                <input
                  type="url"
                  placeholder="https://example.com/damaged-car.jpg"
                  value={imageUrl}
                  onChange={onUrlChange}
                  className="w-full rounded border border-neutral-200 bg-white px-3 py-2 text-sm"
                />
              )}

              <button
                type="submit"
                disabled={loading || (mode === "upload" ? !file : !imageUrl)}
                className="inline-flex items-center justify-center rounded-lg bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>

              {/* Optional hints from detect */}
              {validationIssues && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                  <div className="font-medium mb-1">Note: accuracy of this report may be affected by:</div>
                  <ul className="list-disc pl-5">
                    {validationIssues.map((v, i) => (
                      <li key={i}>{String(v)}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Friendly errors */}
              {error && (
                <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{String(error)}</div>
              )}

              {/* View options */}
              <div className="pt-1 flex flex-col gap-2">
                <Checkbox id="overlay" checked={showOverlay} onChange={setShowOverlay} label="Show damage overlay" />
                <Checkbox id="audit" checked={showAudit} onChange={setShowAudit} label="Show audit metadata" />
              </div>
            </form>

            {/* Image Panel */}
            <div className="rounded-2xl border bg-white p-3 shadow-sm">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm font-medium">Image Preview</div>
                {/* Removed: severity legend text */}
              </div>

              <div className="relative">
                {preview ? (
                  <>
                    <img
                      ref={imgRef}
                      src={preview}
                      alt="preview"
                      className="w-full rounded-lg border max-h-[360px] object-contain bg-neutral-50"
                    />
                    {result?.damage_items && showOverlay && (
                      <CanvasOverlay imgRef={imgRef as React.RefObject<HTMLImageElement>} items={result.damage_items} show={showOverlay} />
                    )}
                  </>
                ) : (
                  <div className="flex h-48 items-center justify-center rounded-lg border border-dashed text-sm text-neutral-500">
                    {mode === "upload" ? "Upload a photo to begin" : "Paste an image URL to preview"}
                  </div>
                )}
              </div>
            </div>

            {/* Legend (left) */}
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Legend & Settings</div>
              <div className="text-xs text-neutral-700 space-y-2">
                <div>
                  <span className="font-medium">Routing:</span>{" "}
                  <span className="inline-flex gap-2 flex-wrap">
                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5">AUTO-APPROVE</span>
                    <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5">INVESTIGATE</span>
                    <span className="rounded-full border border-rose-200 bg-rose-50 px-2 py-0.5">SPECIALIST</span>
                  </span>
                  <div className="mt-1 text-neutral-600">Decision uses severity, confidence, and estimated cost thresholds.</div>
                </div>
                <div>
                  <span className="font-medium">Severity colors:</span> 1–2 (green) • 3 (yellow) • 4 (orange) • 5 (red)
                </div>
                <div>
                  <span className="font-medium">Confidence bands:</span> High ≥ {Math.round(CONF_HIGH * 100)}% • Medium ≥ {Math.round(CONF_MED * 100)}% • otherwise Low
                </div>
                <div className="font-medium">Cost assumptions:</div>
                <div className="text-neutral-600">
                  Labor rate & paint/materials from environment; parts allowance added when severity is high or part replacements are likely.
                </div>
                <div className="text-neutral-500">This report combines YOLO geometry for overlays and GPT vision for labeling and narrative.</div>
              </div>
            </div>
          </div>
        </section>

        {/* Right: Report column */}
        <section className="lg:col-span-8 space-y-4 print:col-span-12 print:w-full">
          {/* Print-only header */}
          <div className="hidden print:block">
            <h2 className="text-xl font-semibold">Car Damage Report</h2>
            <div className="text-xs text-neutral-600">Generated: {new Date().toLocaleString()}</div>
            <hr className="my-2" />
          </div>

          {/* Routing Decision (chip + overall confidence only) */}
          {result?.decision && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Routing Decision</div>
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs ${
                    result.decision.label === "AUTO-APPROVE"
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : result.decision.label === "SPECIALIST"
                      ? "border-rose-200 bg-rose-50 text-rose-700"
                      : "border-amber-200 bg-amber-50 text-amber-700"
                  }`}
                >
                  {result.decision.label}
                </span>
              </div>
              <div className="mt-2 text-xs text-neutral-600">
                Confidence: {fmtPct(decisionConf)} ({confidenceBand(decisionConf)})
              </div>
            </div>
          )}

          {/* Vehicle metadata */}
          {result && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-2 text-sm font-medium">Vehicle metadata</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
                <div>
                  <span className="text-neutral-500">Make:</span> {make}
                </div>
                <div>
                  <span className="text-neutral-500">Model:</span> {model}
                </div>
                <div>
                  <span className="text-neutral-500">Color:</span> {color}
                </div>
              </div>
              <div className="mt-1 text-xs text-neutral-500">
                Vehicle confidence: {fmtPct(result?.vehicle?.confidence)} ({confidenceBand(result?.vehicle?.confidence)})
              </div>
            </div>
          )}

          {/* Detected Damage — minimal, with triangle sort buttons on Sev/Conf only */}
          {Array.isArray(result?.damage_items) && result.damage_items.length > 0 && (
            <DamageTable
              rows={filteredSorted}
              sortKey={sortKey}
              sortDir={sortDir}
              onSetSort={setSort}
              fPaint={fPaint}
              setFPaint={setFPaint}
              fConfMin={fConfMin}
              setFConfMin={setFConfMin}
              fSearch={fSearch}
              setFSearch={setFSearch}
              onResetFilters={resetFilters}
            />
          )}

          {/* Damage summary */}
          {(result?.narrative || (Array.isArray(result?.damage_items) && result.damage_items.length > 0)) && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Damage summary</div>
              <div className="text-sm leading-relaxed text-neutral-800">{buildDamageSummary(result as AnalyzePayload)}</div>
            </div>
          )}

          {/* Estimate — bottom line only */}
          {result && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Estimated Repair Cost</div>
              <div className="text-xl font-semibold">
                {result?.estimate ? `$${result.estimate.cost_low} – $${result.estimate.cost_high}` : "—"}
              </div>
              {Array.isArray(result?.estimate?.assumptions) && result.estimate.assumptions.length > 0 && (
                <div className="mt-2 text-xs text-neutral-500">{result.estimate.assumptions.join(" • ")}</div>
              )}
            </div>
          )}

          {/* Audit (only if checkbox is checked) */}
          {result && showAudit && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Audit Metadata</div>
              <div className="grid grid-cols-1 gap-y-1 text-xs text-neutral-700 sm:grid-cols-2">
                <div>
                  <span className="text-neutral-500">Schema:</span> {result.schema_version ?? "—"}
                </div>
                <div>
                  <span className="text-neutral-500">Model:</span> {result.model ?? "—"}
                </div>
                <div>
                  <span className="text-neutral-500">runId:</span> {result.runId ?? "—"}
                </div>
                <div className="truncate">
                  <span className="text-neutral-500">image_sha256:</span> {result.image_sha256 ?? "—"}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
