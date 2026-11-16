import { useEffect, useState } from "react";
import { useNavigate, useOutletContext } from "react-router-dom";
import { api } from "../api";
import type { ExtractOptions, Health, Upload } from "../types";

type Outlet = { health: Health | null };

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];

export default function Workspace() {
  const navigate = useNavigate();
  const { health } = useOutletContext<Outlet>();
  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [localUrl, setLocalUrl] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [options, setOptions] = useState<ExtractOptions>({
    parser: "auto",
    reconstruct: true,
    refine_sam2: false,
    clean_background: true,
  });

  useEffect(() => {
    if (!health) return;
    setOptions((current) => ({ ...current, refine_sam2: health.defaults.refine_sam2 }));
  }, [health]);

  function take(next: File | null) {
    setError("");
    if (!next) return;
    const extension = next.name.split(".").pop()?.toLowerCase();
    const allowed = ACCEPT.includes(next.type) || ["jpg", "jpeg", "png", "webp"].includes(extension || "");
    if (!allowed) {
      setError("Upload a JPG, JPEG, PNG, or WEBP image.");
      return;
    }
    if (next.size > 30 * 1024 * 1024) {
      setError("File exceeds the 30 MB limit.");
      return;
    }
    setFile(next);
    setUpload(null);
    setLocalUrl(URL.createObjectURL(next));
  }

  async function extract() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const saved = upload ?? (await api.upload(file));
      setUpload(saved);
      const job = await api.start(saved.image_id, options);
      navigate(`/jobs/${job.id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto grid max-w-5xl gap-6 lg:grid-cols-[1.3fr_0.8fr]">
      <section
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          take(event.dataTransfer.files?.[0] ?? null);
        }}
        className={`rounded-2xl border bg-white p-4 shadow-card ${dragging ? "border-pine" : "border-line"}`}
      >
        {!localUrl ? (
          <label className="flex min-h-[420px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-line bg-paper px-6 text-center">
            <span className="font-display text-3xl">Drop a fashion photograph</span>
            <span className="mt-3 max-w-md text-sm leading-6 text-muted">
              JPG, JPEG, PNG, or WEBP. Up to 30 MB. The person stays in this preview; extraction removes them.
            </span>
            <span className="mt-5 rounded-md bg-ink px-4 py-2 text-sm text-white">Browse files</span>
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(event) => take(event.target.files?.[0] ?? null)}
            />
          </label>
        ) : (
          <div>
            <img src={localUrl} alt="Selected fashion photograph" className="max-h-[70vh] w-full rounded-xl object-contain" />
            <div className="mt-3 flex items-center justify-between text-xs text-muted">
              <span>{file?.name}</span>
              <button type="button" className="underline" onClick={() => { setFile(null); setLocalUrl(""); setUpload(null); }}>
                Remove
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="h-fit rounded-2xl border border-line bg-white p-5 shadow-card">
        <h1 className="font-display text-2xl">Processing options</h1>
        <label className="mt-5 block text-sm">
          Parser
          <select
            value={options.parser}
            onChange={(event) => setOptions({ ...options, parser: event.target.value as ExtractOptions["parser"] })}
            className="mt-1 w-full rounded-md border border-line bg-paper px-3 py-2"
          >
            <option value="auto">Auto — SegFormer, then MediaPipe</option>
            <option value="segformer">SegFormer clothes — higher detail, research licence</option>
            <option value="mediapipe">MediaPipe multiclass — Apache-2.0, one clothes class</option>
          </select>
        </label>
        <p className="mt-2 text-xs leading-5 text-muted">
          {health?.models.segformer.license
            ? "SegFormer weights are for non-commercial research and evaluation. Choose MediaPipe for the Apache-2.0 model."
            : "Model licences load with the server status."}
        </p>
        <label className="mt-4 flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            checked={options.reconstruct}
            onChange={(event) => setOptions({ ...options, reconstruct: event.target.checked })}
            className="mt-1"
          />
          <span>Estimate cloth hidden by the body. Generated areas are not the original fabric.</span>
        </label>
        <label className="mt-3 flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            checked={options.refine_sam2}
            onChange={(event) => setOptions({ ...options, refine_sam2: event.target.checked })}
            className="mt-1"
          />
          <span>Refine the outline with SAM 2. Off by default on a 4GB GPU.</span>
        </label>
        <label className="mt-3 flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            checked={options.clean_background}
            onChange={(event) => setOptions({ ...options, clean_background: event.target.checked })}
            className="mt-1"
          />
          <span>Clean leftover background on the garment edge after the person is removed.</span>
        </label>
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
        <button
          type="button"
          disabled={!file || busy}
          onClick={extract}
          className="mt-6 w-full rounded-md bg-pine px-4 py-3 text-sm text-white"
        >
          {busy ? "Uploading…" : "Extract garment"}
        </button>
      </section>
    </div>
  );
}
