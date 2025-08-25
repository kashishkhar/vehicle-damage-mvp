"use client";

import { useEffect, useMemo, useRef, useState } from "react";

/** Confidence bands (config via env) */
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

/** Client-side compression for speed */
async function compressImage(file: File, maxW = 1600, quality = 0.72): Promise<File> {
  const img = document.createElement("img");
  const reader = new FileReader();
  const load = new Promise<void>((resolve, reject) => {
    reader.onload = () => { img.src = reader.result as string; img.onload = () => resolve(); img.onerror = reject; };
    reader.onerror = reject;
  });
  reader.readAsDataURL(file); await load;
  const scale = Math.min(1, maxW / img.width);
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  const canvas = document.createElement("canvas"); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b!), "image/jpeg", quality));
  return new File([blob], "upload.jpg", { type: "image/jpeg" });
}

/** Severity color mapping */
function colorForSeverity(sev: number) {
  if (sev >= 5) return "#dc2626";
  if (sev === 4) return "#f97316";
  if (sev === 3) return "#eab308";
  if (sev === 2) return "#22c55e";
  return "#10b981";
}

/** Canvas overlay for boxes/polygons (normalized coords) */
function CanvasOverlay({
  imgRef,
  items,
  show,
}: {
  imgRef: React.RefObject<HTMLImageElement>;
  items: any[];
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
        drawLabel(ctx, `${d.part} (${sev})`, cx, cy, color);
      } else if (Array.isArray(d.bbox_rel) && d.bbox_rel.length === 4) {
        const [nx, ny, nw, nh] = d.bbox_rel as [number, number, number, number];
        const x = nx * rect.width, y = ny * rect.height, w = nw * rect.width, h = nh * rect.height;
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.fill();
        ctx.stroke();
        drawLabel(ctx, `${d.part} (${sev})`, x + w / 2, y + 14, color);
      }
    });

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [imgRef, items, show]);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" aria-hidden="true" />;
}

function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
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

/** Pretty toggle */
function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="group inline-flex items-center gap-2 select-none"
      aria-pressed={checked}
    >
      <span
        className={`relative h-6 w-11 rounded-full transition-colors ${
          checked ? "bg-black/90" : "bg-neutral-300"
        }`}
      >
        <span
          className={`absolute top-[2px] h-5 w-5 rounded-full bg-white shadow transition-transform ${
            checked ? "translate-x-[22px]" : "translate-x-[2px]"
          }`}
        />
      </span>
      <span className="text-sm text-neutral-700">{label}</span>
    </button>
  );
}

/** Damage summary */
function buildDamageSummary(result: any) {
  const items: any[] = Array.isArray(result?.damage_items) ? result.damage_items : [];
  if (!items.length) return result?.narrative || "No visible damage detected.";

  const parts: string[] = items.map((d) => {
    let desc = `a ${d.damage_type} on the ${d.zone} ${d.part}`;
    if (d.severity >= 4) desc = `severe ${desc}`;
    else if (d.severity === 3) desc = `moderate ${desc}`;
    else if (d.severity <= 2) desc = `minor ${desc}`;

    if (d.needs_paint) desc += ` requiring paint work`;
    if (Array.isArray(d.likely_parts) && d.likely_parts.length) {
      desc += ` with possible replacement of ${d.likely_parts.join(", ")}`;
    }
    return desc;
  });

  const joined = parts.length > 1 ? parts.slice(0, -1).join(", ") + " and " + parts.slice(-1) : parts[0];
  const headline = `The vehicle shows ${joined}.`;
  const note = result?.narrative ? ` ${result.narrative}` : "";
  return headline + note;
}

/** Simple URL validator */
function isLikelyImageUrl(u: string) {
  try {
    const url = new URL(u);
    return /^https?:$/.test(url.protocol);
  } catch {
    return false;
  }
}

export default function Home() {
  const [mode, setMode] = useState<"upload" | "url">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string>("");
  const [warnings, setWarnings] = useState<string[] | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showAudit, setShowAudit] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const urlValid = useMemo(() => isLikelyImageUrl(imageUrl.trim()), [imageUrl]);

  function switchMode(next: "upload" | "url") {
    setMode(next);
    setResult(null);
    setError("");
    setWarnings(null);
    setPreview("");
    if (next === "upload") setImageUrl("");
    else setFile(null);
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setError("");
    setWarnings(null);
    if (f) setPreview(URL.createObjectURL(f));
  }

  function onUrlChange(e: React.ChangeEvent<HTMLInputElement>) {
    const val = e.target.value;
    setImageUrl(val);
    setResult(null);
    setError("");
    setWarnings(null);
    setPreview(val || "");
  }

  /** Single-call submit (no /api/validate-image) */
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setWarnings(null);
    setResult(null);

    try {
      const fd = new FormData();
      if (mode === "upload") {
        if (!file) { setError("Please choose a file."); return; }
        const compressed = await compressImage(file);
        fd.append("file", compressed);
      } else {
        if (!urlValid) { setError("Please enter a valid http(s) image URL."); return; }
        fd.append("imageUrl", imageUrl.trim());
      }

      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      const text = await r.text();
      if (!r.ok) {
        setError(text || "Analysis failed");
        return;
      }
      const j = JSON.parse(text);
      setResult(j);
      if (Array.isArray(j?.warnings) && j.warnings.length) setWarnings(j.warnings);
    } catch (err: any) {
      setError(err?.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-white text-neutral-900">
      {/* Header */}
      <header className="border-b bg-white">
        <div className="mx-auto max-w-7xl px-6 py-5">
          <h1 className="text-xl sm:text-2xl font-semibold">Car Damage Estimator</h1>
          <p className="text-sm text-neutral-600">
            Upload a vehicle photo <em>or paste an image URL</em> to generate a structured damage report, cost band, and routing decision.
          </p>
        </div>
      </header>

      {/* Layout: narrow left (image) / wide right (report) */}
      <div className="mx-auto max-w-7xl px-6 py-6 grid gap-6 lg:grid-cols-12">
        {/* Left: Image column */}
        <section className="lg:col-span-4">
          <div className="lg:sticky lg:top-6 space-y-4">
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
                disabled={loading || (mode === "upload" ? !file : !urlValid)}
                className="inline-flex items-center justify-center rounded-lg bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>

              {/* Photo quality note (non-blocking) */}
              {warnings && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                  <div className="font-medium mb-1">Note: accuracy of the report may be impacted by:</div>
                  <ul className="list-disc pl-5">
                    {warnings.map((w, i) => <li key={i}>{String(w)}</li>)}
                  </ul>
                  <div className="mt-2 text-[11px] text-amber-700">
                    Tip: a clear, well-lit 3/4 angle with the damaged area centered yields best results.
                  </div>
                </div>
              )}

              {/* Fatal errors */}
              {error && (
                <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                  {String(error)}
                </div>
              )}

              {/* Toggles */}
              <div className="pt-1 flex flex-col gap-2">
                <Toggle checked={showOverlay} onChange={setShowOverlay} label="Show damage overlay" />
                <Toggle checked={showAudit} onChange={setShowAudit} label="Show audit metadata" />
              </div>
            </form>

            {/* Image Panel */}
            <div className="rounded-2xl border bg-white p-3 shadow-sm">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm font-medium">Image Preview</div>
                <div className="text-[11px] text-neutral-500">Severity color: 1→5 (green→red)</div>
              </div>

              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {preview ? (
                  <>
                    <img
                      ref={imgRef}
                      src={preview}
                      alt="preview"
                      className="w-full rounded-lg border max-h-[360px] object-contain bg-neutral-50"
                    />
                    {result?.damage_items && (
                      <CanvasOverlay imgRef={imgRef} items={result.damage_items} show={showOverlay} />
                    )}
                  </>
                ) : (
                  <div className="flex h-48 items-center justify-center rounded-lg border border-dashed text-sm text-neutral-500">
                    {mode === "upload" ? "Upload a photo to begin" : "Paste an image URL to preview"}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Right: Report column */}
        <section className="lg:col-span-8 space-y-4">
          {/* Decision */}
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
                <span className="text-xs text-neutral-600">(policy-based)</span>
              </div>
              {Array.isArray(result.decision.reasons) && result.decision.reasons.length > 0 && (
                <ul className="mt-2 grid grid-cols-1 gap-1 pl-5 text-xs text-neutral-700 list-disc">
                  {result.decision.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}
                </ul>
              )}
            </div>
          )}

          {/* Vehicle metadata */}
          {result && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-2 text-sm font-medium">Vehicle metadata</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
                <div><span className="text-neutral-500">Make:</span> {result?.vehicle?.make ?? "—"}</div>
                <div><span className="text-neutral-500">Model:</span> {result?.vehicle?.model ?? "—"}</div>
                <div><span className="text-neutral-500">Color:</span> {result?.vehicle?.color ?? "—"}</div>
              </div>
              <div className="mt-1 text-xs text-neutral-500">
                Vehicle confidence: {fmtPct(result?.vehicle?.confidence)} ({confidenceBand(result?.vehicle?.confidence)})
              </div>
            </div>
          )}

          {/* Detected Damage */}
          {Array.isArray(result?.damage_items) && result.damage_items.length > 0 && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-2 text-sm font-medium">Detected Damage</div>
              <table className="w-full text-[13px] leading-5">
                <thead className="bg-neutral-50 text-neutral-700">
                  <tr>
                    <th className="p-2 text-left">Zone</th>
                    <th className="p-2 text-left">Part</th>
                    <th className="p-2 text-left">Type</th>
                    <th className="p-2 text-left">Sev</th>
                    <th className="p-2 text-left">Paint</th>
                    <th className="p-2 text-left">Hours</th>
                    <th className="p-2 text-left">Likely Parts</th>
                    <th className="p-2 text-left">Conf</th>
                  </tr>
                </thead>
                <tbody>
                  {result.damage_items.map((d: any, i: number) => (
                    <tr key={i} className="border-b last:border-none align-top">
                      <td className="p-2">{d.zone}</td>
                      <td className="p-2">{d.part}</td>
                      <td className="p-2">{d.damage_type}</td>
                      <td className="p-2">{d.severity}</td>
                      <td className="p-2">{d.needs_paint ? "Yes" : "No"}</td>
                      <td className="p-2">{d.est_labor_hours}</td>
                      <td className="p-2 break-words">
                        {Array.isArray(d.likely_parts) && d.likely_parts.length ? d.likely_parts.join(", ") : "—"}
                      </td>
                      <td className="p-2">{fmtPct(d.confidence)} ({confidenceBand(d.confidence)})</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Damage summary */}
          {(result?.narrative || (Array.isArray(result?.damage_items) && result.damage_items.length > 0)) && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Damage summary</div>
              <div className="text-sm leading-relaxed text-neutral-800">
                {buildDamageSummary(result)}
              </div>
            </div>
          )}

          {/* Estimate */}
          {result && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Estimated Repair Cost</div>
              <div className="text-xl font-semibold">
                {result?.estimate ? `$${result.estimate.cost_low} – $${result.estimate.cost_high}` : "—"}
              </div>
              {Array.isArray(result?.estimate?.assumptions) && result.estimate.assumptions.length > 0 && (
                <div className="mt-2 text-xs text-neutral-500">{result.estimate.assumptions.join(" • ")}</div>
              )}
              <div className="mt-2 text-[11px] text-neutral-500">
                Visual pre-estimate only; final cost subject to teardown.
              </div>
            </div>
          )}

          {/* Audit */}
          {result && showAudit && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Audit Metadata</div>
              <div className="grid grid-cols-1 gap-y-1 text-xs text-neutral-700 sm:grid-cols-2">
                <div><span className="text-neutral-500">Schema:</span> {result.schema_version ?? "—"}</div>
                <div><span className="text-neutral-500">Model:</span> {result.model ?? "—"}</div>
                <div><span className="text-neutral-500">runId:</span> {result.runId ?? "—"}</div>
                <div className="truncate"><span className="text-neutral-500">image_sha256:</span> {result.image_sha256 ?? "—"}</div>
              </div>
            </div>
          )}

          {/* Notes */}
          {result?.normalization_notes && (
            <div className="rounded-2xl border bg-white p-5 shadow-sm">
              <div className="mb-1 text-sm font-medium">Notes</div>
              <div className="text-[11px] text-neutral-600">{result.normalization_notes}</div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
