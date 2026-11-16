import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import StageList from "../components/StageList";
import type { Health, Job } from "../types";
import { useOutletContext } from "react-router-dom";

type Outlet = { health: Health | null };

export default function JobPage() {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const { health } = useOutletContext<Outlet>();
  const [job, setJob] = useState<Job | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [tag, setTag] = useState("dress");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let stop = false;
    async function poll() {
      try {
        const next = await api.job(jobId);
        if (stop) return;
        setJob(next);
        if (next.status === "completed" && next.result_id) {
          navigate(`/results/${next.result_id}`, { replace: true });
          return;
        }
        if (next.status === "queued" || next.status === "running") {
          window.setTimeout(poll, 900);
        }
      } catch (reason) {
        if (!stop) setError(reason instanceof Error ? reason.message : "Could not read the job");
      }
    }
    poll();
    return () => {
      stop = true;
    };
  }, [jobId, navigate]);

  useEffect(() => {
    if (!job || job.status !== "awaiting_selection") return;
    const primary = job.garments.filter((item) => !["Hat", "Belt", "Bag", "Left-shoe", "Right-shoe", "Sunglasses", "Others"].includes(item.label));
    const initial = (primary[0] || job.garments[0])?.id;
    setSelected(initial ? [initial] : []);
    setTag((primary[0] || job.garments[0])?.product_tag || "other");
  }, [job?.id, job?.status]);

  async function confirm() {
    if (!job || selected.length === 0) return;
    setBusy(true);
    setError("");
    try {
      await api.continueJob(job.id, selected, tag, {
        reconstruct: job.options.reconstruct !== false,
        refine_sam2: Boolean(job.options.refine_sam2),
      });
      const next = await api.job(job.id);
      setJob(next);
      const watch = async () => {
        const current = await api.job(job.id);
        setJob(current);
        if (current.status === "completed" && current.result_id) {
          navigate(`/results/${current.result_id}`, { replace: true });
          return;
        }
        if (current.status === "queued" || current.status === "running") {
          window.setTimeout(watch, 900);
        }
      };
      window.setTimeout(watch, 600);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not continue");
    } finally {
      setBusy(false);
    }
  }

  if (!job && !error) {
    return <p className="text-sm text-muted">Reading the extraction job…</p>;
  }

  return (
    <div className="mx-auto grid max-w-5xl gap-6 lg:grid-cols-[0.9fr_1.1fr]">
      <section className="rounded-2xl border border-line bg-white p-5 shadow-card">
        <p className="text-xs uppercase tracking-[0.16em] text-muted">Processing</p>
        <h1 className="mt-2 font-display text-3xl">Extracting the garment</h1>
        <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-paper">
          <div className="h-full bg-pine" style={{ width: `${job?.progress ?? 0}%` }} />
        </div>
        <p className="mt-3 min-h-10 text-sm leading-6 text-ink">{job?.message}</p>
        {job && <div className="mt-5"><StageList stages={job.stages} /></div>}
        {job?.status === "failed" && (
          <div className="mt-5 rounded-lg border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-800">
            {job.error}
            <Link to="/new" className="mt-2 block underline">Start again</Link>
          </div>
        )}
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
      </section>

      <section className="rounded-2xl border border-line bg-white p-5 shadow-card">
        {job?.preview_url && (
          <img src={job.preview_url} alt="Uploaded photograph" className="max-h-64 w-full rounded-lg object-contain" />
        )}
        {job?.status === "awaiting_selection" ? (
          <div className="mt-4">
            <h2 className="text-sm font-medium">Choose the garment</h2>
            <p className="mt-1 text-xs leading-5 text-muted">
              The overlay is the detected clothing, not a background cutout. Confirm it before the person is removed.
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {job.garments.map((garment) => {
                const checked = selected.includes(garment.id);
                return (
                  <label key={garment.id} className={`overflow-hidden rounded-lg border ${checked ? "border-pine" : "border-line"}`}>
                    <img
                      src={`/api/extractions/${job.id}/garments/${garment.id}/overlay`}
                      alt={`${garment.label} mask overlay`}
                      className="aspect-[3/4] w-full object-cover"
                    />
                    <span className="flex items-start gap-2 px-3 py-2 text-sm">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          setSelected((current) =>
                            current.includes(garment.id)
                              ? current.filter((id) => id !== garment.id)
                              : [...current, garment.id],
                          );
                          if (!checked) setTag(garment.product_tag);
                        }}
                      />
                      <span>
                        <span className="block">{garment.label}</span>
                        <span className="text-xs text-muted">
                          {Math.round(garment.confidence * 100)}% · suggested {garment.product_tag}
                        </span>
                      </span>
                    </span>
                  </label>
                );
              })}
            </div>
            <label className="mt-4 block text-sm">
              Product tag
              <select
                value={tag}
                onChange={(event) => setTag(event.target.value)}
                className="mt-1 w-full rounded-md border border-line bg-paper px-3 py-2"
              >
                {(health?.product_tags || ["dress", "top", "trousers", "skirt", "other"]).map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
            <p className="mt-2 text-xs leading-5 text-muted">
              Saree, kurti, lehenga, jacket, coat, and jeans are tags. The parser does not have a separate trained class for each of them.
            </p>
            <button
              type="button"
              disabled={busy || selected.length === 0}
              onClick={confirm}
              className="mt-4 rounded-md bg-pine px-4 py-2.5 text-sm text-white"
            >
              {busy ? "Starting…" : "Extract selected garment"}
            </button>
          </div>
        ) : (
          <p className="mt-4 text-sm leading-6 text-muted">
            The stages above change only when the backend enters them.
          </p>
        )}
      </section>
    </div>
  );
}
