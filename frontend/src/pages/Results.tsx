import { useEffect, useState } from "react";
import { Link, useNavigate, useOutletContext, useParams } from "react-router-dom";
import { api } from "../api";
import CompareSlider from "../components/CompareSlider";
import MaskEditor from "../components/MaskEditor";
import type { Health, Job, ResultRecord } from "../types";

type Outlet = { health: Health | null };

export default function Results() {
  const { resultId = "" } = useParams();
  const navigate = useNavigate();
  const { health } = useOutletContext<Outlet>();
  const [record, setRecord] = useState<ResultRecord | null>(null);
  const [background, setBackground] = useState<"transparent" | "white">("transparent");
  const [error, setError] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [editJob, setEditJob] = useState<Job | null>(null);
  const [version, setVersion] = useState(0);

  async function load() {
    const next = await api.result(resultId);
    setRecord(next);
  }

  useEffect(() => {
    load().catch((reason: Error) => setError(reason.message));
  }, [resultId]);

  useEffect(() => {
    if (!editJob || (editJob.status !== "queued" && editJob.status !== "running")) return;
    const timer = window.setTimeout(async () => {
      try {
        const next = await api.job(editJob.id);
        setEditJob(next);
        if (next.status === "completed") {
          setVersion((value) => value + 1);
          await load();
        }
        if (next.status === "failed") setError(next.error || "The edit failed");
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Could not update the edit");
      }
    }, 800);
    return () => window.clearTimeout(timer);
  }, [editJob]);

  if (!record && !error) return <p className="text-sm text-muted">Loading the garment…</p>;
  if (!record) return <p className="text-sm text-red-700">{error}</p>;

  const after = `${background === "transparent" ? record.transparent_preview_url : record.white_preview_url}&v=${version}`;
  const before = `${record.original_preview_url}&v=${version}`;
  const editing = editJob?.status === "queued" || editJob?.status === "running";

  return (
    <div className="mx-auto max-w-6xl">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.16em] text-muted">{record.parser} · {record.garment_label}</p>
          <h1 className="mt-1 font-display text-4xl capitalize">{record.product_tag}</h1>
        </div>
        <div className="flex flex-wrap gap-2">
          <a className="rounded-md bg-ink px-3 py-2 text-sm text-white" href={record.download_png_url} download>Download PNG</a>
          <a className="rounded-md border border-line bg-white px-3 py-2 text-sm" href={record.download_jpeg_url} download>Download JPEG</a>
          <Link to="/new" className="rounded-md border border-line bg-white px-3 py-2 text-sm">New extraction</Link>
          {confirmDelete ? (
            <>
              <button
                type="button"
                disabled={deleting}
                onClick={async () => {
                  setDeleting(true);
                  setError("");
                  try {
                    await api.remove(record.id);
                    navigate("/");
                  } catch (reason) {
                    setDeleting(false);
                    setError(reason instanceof Error ? reason.message : "Could not delete this extraction");
                  }
                }}
                className="rounded-md bg-ink px-3 py-2 text-sm text-white"
              >
                {deleting ? "Deleting…" : "Delete extraction"}
              </button>
              <button type="button" onClick={() => setConfirmDelete(false)} className="rounded-md border border-line bg-white px-3 py-2 text-sm">
                Keep
              </button>
            </>
          ) : (
            <button type="button" onClick={() => setConfirmDelete(true)} className="rounded-md border border-line bg-white px-3 py-2 text-sm">
              Delete
            </button>
          )}
        </div>
      </div>

      {record.low_confidence && (
        <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-950">
          Segmentation confidence is {Math.round(record.confidence * 100)}%. Correct the mask before treating this export as final.
          Automatic reconstruction is skipped at this confidence unless you regenerate it yourself.
        </p>
      )}
      <p className="mt-4 text-sm leading-6 text-ink">{record.reconstruction_message}</p>
      {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
      {editing && <p className="mt-3 text-sm text-muted">{editJob?.message || "Updating the garment…"}</p>}

      <div className="mt-5 flex gap-2">
        <button type="button" onClick={() => setBackground("transparent")} className={`rounded-full px-3 py-1.5 text-sm ${background === "transparent" ? "bg-pine text-white" : "bg-white border border-line"}`}>Transparent</button>
        <button type="button" onClick={() => setBackground("white")} className={`rounded-full px-3 py-1.5 text-sm ${background === "white" ? "bg-pine text-white" : "bg-white border border-line"}`}>White</button>
        <label className="ml-auto flex items-center gap-2 text-sm">
          Tag
          <select
            value={record.product_tag}
            onChange={async (event) => {
              const next = await api.setTag(record.id, event.target.value);
              setRecord(next);
            }}
            className="rounded-md border border-line bg-white px-2 py-1"
          >
            {(health?.product_tags || [record.product_tag]).map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-4">
        <CompareSlider before={before} after={after} checker={background === "transparent"} />
      </div>

      <section className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-2xl border border-line bg-white p-5 shadow-card">
          <h2 className="text-sm font-medium">Mask editing</h2>
          <div className="mt-4">
            <MaskEditor
              originalUrl={`/api/uploads/${record.image_id}/file?kind=original`}
              maskUrl={`${record.mask_url}&v=${version}`}
              disabled={editing}
              onApply={async (blob, regenerate) => {
                setError("");
                try {
                  const job = await api.editMask(record.id, blob, regenerate);
                  setEditJob(job);
                } catch (reason) {
                  setError(reason instanceof Error ? reason.message : "Could not apply the mask");
                }
              }}
            />
          </div>
        </div>
        <div className="rounded-2xl border border-line bg-white p-5 shadow-card">
          <h2 className="text-sm font-medium">Notes from this extraction</h2>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-muted">
            {record.notes.map((note, index) => (
              <li key={`${index}-${note.slice(0, 24)}`}>{note}</li>
            ))}
          </ul>
          <p className="mt-4 text-xs leading-5 text-muted">{record.parser_license}</p>
          <button
            type="button"
            disabled={editing}
            onClick={async () => {
              setError("");
              try {
                setEditJob(await api.regenerate(record.id));
              } catch (reason) {
                setError(reason instanceof Error ? reason.message : "Could not regenerate");
              }
            }}
            className="mt-5 rounded-md border border-line px-3 py-2 text-sm"
          >
            Regenerate hidden areas
          </button>
          <Link to={`/new`} className="mt-3 block text-sm text-pine underline">Retry with another parser</Link>
        </div>
      </section>
    </div>
  );
}
