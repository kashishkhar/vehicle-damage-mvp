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
 * ✔ Sample images (thumbnails), “Why this decision?” expander
 * ✔ Copy icons for summary & estimate
 * ✔ Print-only page with overlay snapshot + legend
 *
 * Note: This is an MVP. Code is kept simple, predictable, and easy to reason about.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalyzePayload, DamageItem, DetectPayload } from "./types";

/** ──────────────────────────────────────────────────────────────────────────
 *  UI Helpers
 *  ------------------------------------------------------------------------- */

const CONF_HIGH = Number(process.env.NEXT_PUBLIC_CONF_HIGH ?? 0.85);
const CONF_MED = Number(process.env.NEXT_PUBLIC_CONF_MED ?? 0.6);

// Public UI thresholds (mirror server defaults; safe to expose)
const AUTO_MAX_SEVERITY = Number(process.env.NEXT_PUBLIC_AUTO_MAX_SEVERITY ?? 2);
const SPEC_MIN_SEVERITY = Number(process.env.NEXT_PUBLIC_SPEC_MIN_SEVERITY ?? 4);
const AUTO_MAX_COST = Number(process.env.NEXT_PUBLIC_AUTO_MAX_COST ?? 1500);
const SPEC_MIN_COST = Number(process.env.NEXT_PUBLIC_SPEC_MIN_COST ?? 5000);
const AUTO_MIN_CONF = Number(process.env.NEXT_PUBLIC_AUTO_MIN_CONF ?? 0.75);

function confidenceBand(p?: number) {
  if (typeof p !== "number") return "Unknown";
  if (p >= CONF_HIGH) return "High";
  if (p >= CONF_MED) return "Medium";
  return "Low";
}

function fmtPct(p?: number) {
  return typeof p === "number" ? `${Math.round(p * 100)}%` : "—";
}

function fmtMoney(n?: number) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  return `$${n.toLocaleString()}`;
}

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

/** Weighted aggregation (mirrors server) */
function aggregateDecisionConfidence(items: DamageItem[]): number {
  if (!Array.isArray(items) || !items.length) return 0.5;
  let num = 0, den = 0;
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

async function fileToDataUrl(file: File): Promise<string> {
  const reader = new FileReader();
  const p = new Promise<string>((resolve, reject) => {
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
  });
  reader.readAsDataURL(file);
  return p;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
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
  const w = metrics.width + pad * 2, h = 16;
  ctx.fillRect(x - w / 2, y - 2, w, h);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

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
      ctx.fillStyle = color + "33";

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
        const x = nx * rect.width, y = ny * rect.height, w = nw * rect.width, h = nh * rect.height;
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

async function makeOverlaySnapshot(src: string, items: DamageItem[], targetW = 1200): Promise<string> {
  const img = await loadImage(src);
  const scale = Math.min(1, targetW / img.naturalWidth);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  ctx.lineWidth = 2;
  items.forEach((d) => {
    const sev = Number(d.severity ?? 1);
    const color = colorForSeverity(sev);
    ctx.strokeStyle = color;
    ctx.fillStyle = color + "33";
    if (Array.isArray(d.polygon_rel) && d.polygon_rel.length >= 3) {
      const pts = d.polygon_rel as [number, number][];
      ctx.beginPath();
      pts.forEach(([nx, ny], i) => {
        const x = nx * w, y = ny * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      const cx = (pts.reduce((s, p) => s + p[0], 0) / pts.length) * w;
      const cy = (pts.reduce((s, p) => s + p[1], 0) / pts.length) * h;
      drawLabel(ctx, String(d.part ?? ""), cx, cy, color);
    } else if (Array.isArray(d.bbox_rel) && d.bbox_rel.length === 4) {
      const [nx, ny, nw, nh] = d.bbox_rel as [number, number, number, number];
      const x = nx * w, y = ny * h, ww = nw * w, hh = nh * h;
      ctx.beginPath();
      ctx.rect(x, y, ww, hh);
      ctx.fill();
      ctx.stroke();
      drawLabel(ctx, String(d.part ?? ""), x + ww / 2, y + 14, color);
    }
  });

  return canvas.toDataURL("image/jpeg", 0.92);
}

/** ──────────────────────────────────────────────────────────────────────────
 *  Tiny Icon components (no new deps)
 *  ------------------------------------------------------------------------- */

function CopyIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
      <path d="M9 9.5A2.5 2.5 0 0 1 11.5 7H17a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-5.5a2.5 2.5 0 0 1-2.5-2.5v-7Z" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M7 15.5V6a2 2 0 0 1 2-2h6.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}
function CheckIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
      <path d="M20 6L9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
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
    <label htmlFor={id} className="flex items-center gap-2 text-sm text-slate-800 select-none">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 accent-indigo-600 transition-transform duration-150 ease-out focus-visible:ring-2 focus-visible:ring-indigo-500 rounded"
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
    const lp = Array.isArray(d.likely_parts) ? d.likely_parts : [];
    if (lp.length) {
      phrase += ` with possible replacement of ${lp.join(", ")}`;
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
  const baseBtn =
    "px-1.5 py-0.5 rounded border text-[11px] transition-colors duration-150 ease-out";
  const activeCls = "border-slate-800 text-slate-900 bg-white/70";
  const idleCls =
    "border-slate-300 text-slate-600 hover:text-slate-900 hover:bg-white/60";
  return (
    <th className={`p-2 text-left select-none ${className}`}>
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-700">{label}</span>
        <div className="flex items-center gap-1">
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
    <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)] transition-all duration-200 hover:shadow-[0_10px_40px_rgb(0,0,0,0.08)]">
      <div className="mb-1 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium text-slate-900">Detected Damage</div>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={fSearch}
            onChange={(e) => setFSearch(e.target.value)}
            placeholder="Search zone/part/type…"
            className="w-40 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 text-xs backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 transition-all"
          />
          <button
            type="button"
            onClick={onResetFilters}
            className="rounded-lg border border-slate-300 bg-white/70 px-2 py-1 text-xs transition-all hover:bg-white/90 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            title="Reset filters"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="mb-3 flex items-center justify-end gap-2">
        <select
          value={fPaint}
          onChange={(e) => setFPaint(e.target.value as "all" | "yes" | "no")}
          className="w-28 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 text-xs backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 transition-all"
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
          className="w-20 rounded-lg border border-slate-200 bg-white/70 px-2 py-1 text-xs backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 transition-all"
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[13px] leading-5 border-collapse">
          <thead className="bg-white/60 backdrop-blur text-slate-700">
            <tr>
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
              <tr
                key={i}
                className="border-b border-slate-200/60 last:border-none align-top transition-colors hover:bg-white/60"
              >
                <td className="p-2">{d.zone}</td>
                <td className="p-2">{d.part}</td>
                <td className="p-2">{d.damage_type}</td>
                <td className="p-2">{d.severity}</td>
                <td className="p-2">{d.needs_paint ? "Yes" : "No"}</td>
                <td className="p-2">
                  {fmtPct(d.confidence)} ({confidenceBand(d.confidence)})
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={6} className="p-3 text-center text-sm text-slate-500">
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
 *  Mini score helpers for the “Why this decision?” expander
 *  ------------------------------------------------------------------------- */

function sevClass(sev: number) {
  if (sev >= SPEC_MIN_SEVERITY) return "border-rose-200 bg-rose-50 text-rose-700";
  if (sev <= AUTO_MAX_SEVERITY) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  return "border-amber-200 bg-amber-50 text-amber-700";
}
function costClass(cost: number | undefined) {
  if (typeof cost !== "number") return "border-slate-200 bg-white/70 text-slate-700";
  if (cost >= SPEC_MIN_COST) return "border-rose-200 bg-rose-50 text-rose-700";
  if (cost <= AUTO_MAX_COST) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  return "border-amber-200 bg-amber-50 text-amber-700";
}
function confClass(conf: number) {
  // Confidence never escalates; red reserved for escalation on other dims.
  if (conf >= AUTO_MIN_CONF) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  return "border-amber-200 bg-amber-50 text-amber-700";
}

function whyBlurb(label: "AUTO-APPROVE" | "INVESTIGATE" | "SPECIALIST", m: {
  maxSev: number;
  costHigh?: number;
  aggConf: number;
}) {
  if (label === "SPECIALIST") {
    if (m.maxSev >= SPEC_MIN_SEVERITY && (m.costHigh ?? 0) >= SPEC_MIN_COST) {
      return "Escalated because severity and cost exceed specialist thresholds.";
    }
    if (m.maxSev >= SPEC_MIN_SEVERITY) return `Escalated because severity ≥ ${SPEC_MIN_SEVERITY}.`;
    if ((m.costHigh ?? 0) >= SPEC_MIN_COST) return `Escalated because cost ≥ ${fmtMoney(SPEC_MIN_COST)}.`;
    return "Escalated based on policy thresholds.";
  }
  if (label === "AUTO-APPROVE") {
    return "All checks are within auto-approve thresholds.";
  }
  // INVESTIGATE
  const blockers: string[] = [];
  if (m.maxSev > AUTO_MAX_SEVERITY) blockers.push(`severity > ${AUTO_MAX_SEVERITY}`);
  if ((m.costHigh ?? 0) > AUTO_MAX_COST) blockers.push(`cost > ${fmtMoney(AUTO_MAX_COST)}`);
  if (m.aggConf < AUTO_MIN_CONF) blockers.push(`confidence < ${fmtPct(AUTO_MIN_CONF)}`);
  return blockers.length ? `Needs review: ${blockers.join(", ")}.` : "Needs review.";
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
  const [showWhy, setShowWhy] = useState(false);

  // Copy state
  const [copiedSummary, setCopiedSummary] = useState(false);
  const [copiedEstimate, setCopiedEstimate] = useState(false);

  // Print snapshot
  const [snapshotUrl, setSnapshotUrl] = useState<string>("");

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

      // Guard: LLM says "no vehicle" or very low confidence with no items
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

  const copyToClipboard = useCallback(async (text: string, which: "summary" | "estimate") => {
    try {
      await navigator.clipboard.writeText(text);
      if (which === "summary") {
        setCopiedSummary(true);
        setTimeout(() => setCopiedSummary(false), 1200);
      } else {
        setCopiedEstimate(true);
        setTimeout(() => setCopiedEstimate(false), 1200);
      }
    } catch {
      // no-op
    }
  }, []);

  const handlePrint = useCallback(async () => {
    try {
      if (result?.damage_items?.length && preview) {
        const url = await makeOverlaySnapshot(preview, result.damage_items);
        setSnapshotUrl(url);
        setTimeout(() => window.print(), 50);
        return;
      }
    } catch {
      // fall through
    }
    window.print();
  }, [preview, result]);

  const samples = [
    {
      url: "https://images.pexels.com/photos/11985216/pexels-photo-11985216.jpeg",
    },
    {
      url: "https://images.pexels.com/photos/6442699/pexels-photo-6442699.jpeg",
    },
    {
      url: "https://i.redd.it/902pxt9a8r4c1.jpg",
    },
    {
      url: "https://preview.redd.it/bfcq81ek7pbf1.jpeg?auto=webp&s=4548c35ddfe6f371a1639df78528b5ea573ae64b",
    },
  ];

  /** Renamed from `useSample` → avoids hook rule false-positive. */
  const selectSample = useCallback((url: string) => {
    setMode("url");
    setImageUrl(url);
    setPreview(url);
    setResult(null);
    setError("");
    setValidationIssues(null);
  }, []);

  // Metrics used in "Why this decision?"
  const metrics = useMemo(() => {
    const maxSev = Math.max(
      0,
      ...((result?.damage_items ?? []).map((d) => Number(d.severity ?? 0)) as number[])
    );
    const costHigh = result?.estimate?.cost_high;
    const aggConf = decisionConf;
    return { maxSev, costHigh, aggConf };
  }, [result, decisionConf]);

  return (
    <main className="relative min-h-screen text-slate-900">
      {/* Background */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-br from-slate-50 via-indigo-50 to-sky-50 bg-[length:200%_200%]" />
      <div className="bg-animated fixed inset-0 -z-10" />
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,rgba(29,78,216,0.06),transparent_55%),radial-gradient(ellipse_at_bottom_left,rgba(2,132,199,0.06),transparent_55%)]" />

      <style jsx global>{`
        @media print {
          @page {
            size: A4 portrait;
            margin: 0.5in;
          }
          html, body { background: #ffffff !important; }
          .print-break { page-break-before: always; }
        }
        .bg-animated {
          animation: gradientShift 18s ease-in-out infinite;
          background: linear-gradient(
            120deg,
            rgba(99, 102, 241, 0.08),
            rgba(14, 165, 233, 0.08),
            rgba(99, 102, 241, 0.08)
          );
          background-size: 200% 200%;
        }
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>

      {/* Header */}
      <header className="border-b border-white/30 bg-white/60 backdrop-blur-xl print:hidden shadow-[0_4px_20px_rgba(0,0,0,0.05)]">
        <div className="mx-auto max-w-7xl px-6 py-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-slate-900">
              Car Damage Estimator
            </h1>
            <p className="text-sm text-slate-600">
              Upload a vehicle photo or paste an image URL to generate a structured damage report, cost band, and routing decision.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handlePrint}
              disabled={!result}
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white/70 px-3 py-2 text-sm font-medium text-slate-800 disabled:opacity-50 transition-all hover:shadow-sm active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              aria-label="Export Report PDF"
              title={result ? "Export Report PDF" : "Run an analysis first"}
            >
              Export Report PDF
            </button>
          </div>
        </div>
      </header>

      {/* Layout */}
      <div className="mx-auto max-w-7xl px-6 py-6 grid gap-6 lg:grid-cols-12">
        {/* Left: Image column */}
        <section className="lg:col-span-4 print:hidden">
          <div className="lg:sticky lg:top-6 space-y-4">
            {/* Input Card */}
            <form
              onSubmit={handleSubmit}
              className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-4 shadow-[0_8px_30px_rgb(0,0,0,0.06)] space-y-3 transition-all hover:shadow-[0_10px_40px_rgb(0,0,0,0.08)]"
            >
              {/* Mode switch */}
              <div className="flex rounded-lg overflow-hidden border border-slate-200/70">
                <button
                  type="button"
                  onClick={() => switchMode("upload")}
                  className={`flex-1 px-3 py-2 text-sm transition-all ${
                    mode === "upload" ? "bg-indigo-600 text-white" : "bg-white/70 text-slate-700 hover:bg-white/90"
                  }`}
                >
                  Upload
                </button>
                <button
                  type="button"
                  onClick={() => switchMode("url")}
                  className={`flex-1 px-3 py-2 text-sm transition-all ${
                    mode === "url" ? "bg-indigo-600 text-white" : "bg-white/70 text-slate-700 hover:bg-white/90"
                  }`}
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
                    className="block w-full rounded-lg border border-slate-200 bg-white/70 px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm hover:file:bg-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 transition-all"
                  />
                </label>
              ) : (
                <input
                  type="url"
                  placeholder="https://example.com/damaged-car.jpg"
                  value={imageUrl}
                  onChange={onUrlChange}
                  className="w-full rounded-lg border border-slate-200 bg-white/70 px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 transition-all"
                />
              )}

              <button
                type="submit"
                disabled={loading || (mode === "upload" ? !file : !imageUrl)}
                className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-60 transition-all hover:bg-indigo-700 hover:shadow-sm active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>

              {/* Optional hints from detect */}
              {validationIssues && (
                <div className="rounded-lg border border-amber-200/70 bg-amber-50/80 p-3 text-sm text-amber-800 backdrop-blur">
                  <div className="font-medium mb-1">Note: accuracy of this report may be affected by:</div>
                  <ul className="list-disc pl-5">
                    {validationIssues.map((v, i) => (
                      <li key={i}>{String(v)}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Friendly errors (A11y live region) */}
              <div aria-live="polite">
                {error && (
                  <div className="rounded-lg border border-rose-200/70 bg-rose-50/80 p-3 text-sm text-rose-700 backdrop-blur">
                    {String(error)}
                  </div>
                )}
              </div>

              {/* View options */}
              <div className="pt-1 flex flex-col gap-2">
                <Checkbox id="overlay" checked={showOverlay} onChange={setShowOverlay} label="Show damage overlay" />
                <Checkbox id="audit" checked={showAudit} onChange={setShowAudit} label="Show audit metadata" />
              </div>
            </form>

            {/* Image Panel */}
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-3 shadow-[0_8px_30px_rgb(0,0,0,0.06)] transition-all hover:shadow-[0_10px_40px_rgb(0,0,0,0.08)]">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm font-medium text-slate-900">Image Preview</div>
              </div>

              <div className="relative">
                {preview ? (
                  <>
                    <img
                      ref={imgRef}
                      src={preview}
                      alt="preview"
                      className="w-full rounded-lg border border-slate-200 max-h-[360px] object-contain bg-slate-50"
                    />
                    {result?.damage_items && showOverlay && (
                      <CanvasOverlay imgRef={imgRef as React.RefObject<HTMLImageElement>} items={result.damage_items} show={showOverlay} />
                    )}
                  </>
                ) : (
                  <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-300 text-sm text-slate-500">
                    {mode === "upload" ? "Upload a photo to begin" : "Paste an image URL to preview"}
                  </div>
                )}
              </div>
            </div>

            {/* Sample Images */}
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-4 shadow-[0_8px_30px_rgb(0,0,0,0.06)] transition-all hover:shadow-[0_10px_40px_rgb(0,0,0,0.08)]">
              <div className="mb-2 text-sm font-medium text-slate-900">Sample Images</div>
              <div className="grid grid-cols-4 gap-2">
                {samples.map((s) => (
                  <button
                    key={s.url}
                    type="button"
                    onClick={() => selectSample(s.url)}
                    className="group relative rounded-lg overflow-hidden border border-slate-200 bg-white/60 hover:shadow-sm active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                    title="Sample image"
                  >
                    <img src={s.url} alt="Sample image" className="h-16 w-full object-cover" />
                    <div className="absolute inset-x-0 bottom-0 bg-white/80 text-[10px] px-1 py-0.5 text-slate-700 truncate">
                    </div>
                  </button>
                ))}
              </div>
              <div className="mt-2 text-[11px] text-slate-500">
                Tip: Click a sample to auto-fill the URL and preview instantly.
              </div>
            </div>

            {/* Legend (left) */}
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-1 text-sm font-medium text-slate-900">Legend & Settings</div>
              <div className="text-xs text-slate-700 space-y-2">
                <div>
                  <span className="font-medium">Routing:</span>{" "}
                  <span className="inline-flex gap-2 flex-wrap">
                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5">AUTO-APPROVE</span>
                    <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5">INVESTIGATE</span>
                    <span className="rounded-full border border-rose-200 bg-rose-50 px-2 py-0.5">SPECIALIST</span>
                  </span>
                  <div className="mt-1 text-slate-600">Decision uses severity, confidence, and estimated cost thresholds.</div>
                </div>
                <div>
                  <span className="font-medium">Severity colors:</span> 1–2 (green) • 3 (yellow) • 4 (orange) • 5 (red)
                </div>
                <div>
                  <span className="font-medium">Confidence bands:</span> High ≥ {Math.round(CONF_HIGH * 100)}% • Medium ≥ {Math.round(CONF_MED * 100)}% • otherwise Low
                </div>
                <div>
                  <span className="font-medium">Cost assumptions:</span> Labor rate & paint/materials from environment; parts allowance added when severity is high or part replacements are likely.
                </div>
                <div className="text-slate-500">This report combines YOLO geometry for overlays and GPT vision for labeling and narrative.</div>
              </div>
            </div>
          </div>
        </section>

        {/* Right: Report column */}
        <section className="lg:col-span-8 space-y-4 print:col-span-12 print:w-full">
          {/* Print-only header */}
          <div className="hidden print:block">
            <h2 className="text-xl font-semibold">Car Damage Report</h2>
            <div className="text-xs text-slate-600">Generated: {new Date().toLocaleString()}</div>
            <hr className="my-2" />
          </div>

          {/* Routing Decision */}
          {result?.decision && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-1 text-sm font-medium text-slate-900">Routing Decision</div>
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
              <div className="mt-2 text-xs text-slate-600">
                Confidence: {fmtPct(decisionConf)} ({confidenceBand(decisionConf)})
              </div>

              {/* Why this decision? (simplified) */}
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => setShowWhy((v) => !v)}
                  className="text-xs text-indigo-700 hover:text-indigo-900 underline underline-offset-2"
                >
                  {showWhy ? "Hide details" : "Why this decision?"}
                </button>

                {showWhy && (
                  <div className="mt-2 space-y-3">
                    <div className="text-xs text-slate-700">{whyBlurb(result.decision.label, metrics)}</div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                      {/* Severity tile */}
                      <div className={`rounded-xl border px-3 py-2 ${sevClass(metrics.maxSev)}`}>
                        <div className="text-[11px] opacity-80">Severity (need ≤ {AUTO_MAX_SEVERITY})</div>
                        <div className="text-sm font-medium">Max {Math.max(0, metrics.maxSev)}</div>
                        {metrics.maxSev >= SPEC_MIN_SEVERITY && (
                          <div className="text-[11px] opacity-80 mt-0.5">Escalates at ≥ {SPEC_MIN_SEVERITY}</div>
                        )}
                      </div>

                      {/* Cost tile */}
                      <div className={`rounded-xl border px-3 py-2 ${costClass(metrics.costHigh)}`}>
                        <div className="text-[11px] opacity-80">Cost (need ≤ {fmtMoney(AUTO_MAX_COST)})</div>
                        <div className="text-sm font-medium">
                          High {typeof metrics.costHigh === "number" ? fmtMoney(metrics.costHigh) : "—"}
                        </div>
                        {typeof metrics.costHigh === "number" && metrics.costHigh >= SPEC_MIN_COST && (
                          <div className="text-[11px] opacity-80 mt-0.5">Escalates at ≥ {fmtMoney(SPEC_MIN_COST)}</div>
                        )}
                      </div>

                      {/* Confidence tile */}
                      <div className={`rounded-xl border px-3 py-2 ${confClass(metrics.aggConf)}`}>
                        <div className="text-[11px] opacity-80">Confidence (need ≥ {fmtPct(AUTO_MIN_CONF)})</div>
                        <div className="text-sm font-medium">Avg {fmtPct(metrics.aggConf)}</div>
                        <div className="text-[11px] opacity-80 mt-0.5">Doesn’t escalate on its own</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Vehicle metadata */}
          {result && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-2 text-sm font-medium text-slate-900">Vehicle metadata</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
                <div><span className="text-slate-500">Make:</span> {make}</div>
                <div><span className="text-slate-500">Model:</span> {model}</div>
                <div><span className="text-slate-500">Color:</span> {color}</div>
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Vehicle confidence: {fmtPct(result?.vehicle?.confidence)} ({confidenceBand(result?.vehicle?.confidence)})
              </div>
            </div>
          )}

          {/* Detected Damage */}
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

          {/* Damage summary + Copy icon */}
          {(result?.narrative || (Array.isArray(result?.damage_items) && result.damage_items.length > 0)) && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-1 flex items-center justify-between">
                <div className="text-sm font-medium text-slate-900">Damage summary</div>
                {result && (
                  <button
                    type="button"
                    onClick={() => copyToClipboard(buildDamageSummary(result as AnalyzePayload), "summary")}
                    className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-300 bg-white/70 hover:bg-white/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                    title="Copy summary"
                    aria-label="Copy summary"
                  >
                    {copiedSummary ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
                    <span className="sr-only">{copiedSummary ? "Copied summary" : "Copy summary"}</span>
                  </button>
                )}
              </div>
              <div className="text-sm leading-relaxed text-slate-800">{buildDamageSummary(result as AnalyzePayload)}</div>
            </div>
          )}

          {/* Estimate + Copy icon */}
          {result && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-1 flex items-center justify-between">
                <div className="text-sm font-medium text-slate-900">Estimated Repair Cost</div>
                <button
                  type="button"
                  onClick={() => {
                    const est = result?.estimate
                      ? `${fmtMoney(result.estimate.cost_low)} – ${fmtMoney(result.estimate.cost_high)}`
                      : "—";
                    copyToClipboard(est, "estimate");
                  }}
                  className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-300 bg-white/70 hover:bg-white/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                  title="Copy estimate"
                  aria-label="Copy estimate"
                >
                  {copiedEstimate ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
                  <span className="sr-only">{copiedEstimate ? "Copied estimate" : "Copy estimate"}</span>
                </button>
              </div>
              <div className="text-xl font-semibold tracking-tight">
                {result?.estimate ? `${fmtMoney(result.estimate.cost_low)} – ${fmtMoney(result.estimate.cost_high)}` : "—"}
              </div>
              {Array.isArray(result?.estimate?.assumptions) && result.estimate.assumptions.length > 0 && (
                <div className="mt-2 text-xs text-slate-500">{result.estimate.assumptions.join(" • ")}</div>
              )}
            </div>
          )}

          {/* Audit */}
          {result && showAudit && (
            <div className="rounded-2xl border border-white/30 bg-white/60 backdrop-blur-xl p-5 shadow-[0_8px_30px_rgb(0,0,0,0.06)]">
              <div className="mb-1 text-sm font-medium text-slate-900">Audit Metadata</div>
              <div className="grid grid-cols-1 gap-y-1 text-xs text-slate-700 sm:grid-cols-2">
                <div><span className="text-slate-500">Schema:</span> {result.schema_version ?? "—"}</div>
                <div><span className="text-slate-500">Model:</span> {result.model ?? "—"}</div>
                <div><span className="text-slate-500">runId:</span> {result.runId ?? "—"}</div>
                <div className="truncate"><span className="text-slate-500">image_sha256:</span> {result.image_sha256 ?? "—"}</div>
              </div>
            </div>
          )}

          {/* PRINT-ONLY: Overlay snapshot + legend */}
          <div className="hidden print:block print-break">
            <h3 className="text-base font-semibold text-slate-900 mb-2">Overlay Snapshot</h3>
            {snapshotUrl ? (
              <img src={snapshotUrl} alt="Overlay snapshot" className="w-full border rounded" />
            ) : (
              <div className="text-xs text-slate-600">No overlay available.</div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
