import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { api } from "../api";
import type { HistoryResponse } from "../types";

export default function Dashboard() {
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [error, setError] = useState("");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  function load() {
    return api.history().then(setHistory);
  }

  useEffect(() => {
    load().catch((reason: Error) => setError(reason.message));
  }, []);

  async function remove(resultId: string) {
    setDeletingId(resultId);
    setError("");
    try {
      await api.remove(resultId);
      setPendingId(null);
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete that extraction");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.16em] text-muted">Studio</p>
          <h1 className="mt-2 font-display text-4xl text-ink">Garment extraction</h1>
          <p className="mt-3 max-w-xl text-sm leading-6 text-muted">
            Upload a fashion photograph. The parser separates the clothing from the person,
            then you can export the garment on transparency or white.
          </p>
        </div>
        <Link to="/new" className="rounded-md bg-pine px-4 py-2.5 text-sm text-white">
          Upload new image
        </Link>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        <article className="rounded-xl border border-line bg-white p-5 shadow-card">
          <p className="text-xs text-muted">Processed images</p>
          <p className="mt-2 font-display text-4xl">{history ? history.count : "—"}</p>
        </article>
        <article className="rounded-xl border border-line bg-white p-5 shadow-card sm:col-span-2">
          <p className="text-xs text-muted">How a result is made</p>
          <p className="mt-2 text-sm leading-6 text-ink">
            Clothing is segmented apart from skin, hair, and limbs. Original fabric pixels are kept.
            Only regions hidden by the body can be estimated, and that estimate is labelled as generated.
          </p>
        </article>
      </div>

      <section className="mt-8">
        <h2 className="text-sm font-medium">Recent extractions</h2>
        {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
        {history && history.items.length === 0 && (
          <p className="mt-4 rounded-xl border border-dashed border-line bg-white px-4 py-8 text-sm text-muted">
            No garments extracted yet.
          </p>
        )}
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {history?.items.map((item) => {
            const label = item.product_tag || item.garment_label || "extraction";
            const confirming = pendingId === item.id;
            return (
              <article
                key={item.id}
                className="relative overflow-hidden rounded-xl border border-line bg-white shadow-card"
              >
                {confirming ? (
                  <div className="absolute inset-x-2 top-2 z-10 flex items-center justify-between gap-2 rounded-md bg-white/95 px-2 py-1.5 text-xs shadow-card">
                    <span>Delete this {label}?</span>
                    <span className="flex gap-1">
                      <button
                        type="button"
                        disabled={deletingId === item.id}
                        onClick={() => remove(item.id)}
                        className="rounded bg-ink px-2 py-1 text-white"
                      >
                        {deletingId === item.id ? "Deleting…" : "Delete"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setPendingId(null)}
                        className="rounded border border-line px-2 py-1"
                      >
                        Keep
                      </button>
                    </span>
                  </div>
                ) : (
                  <button
                    type="button"
                    aria-label={`Delete ${label}`}
                    onClick={() => setPendingId(item.id)}
                    className="absolute right-2 top-2 z-10 grid h-8 w-8 place-items-center rounded-md bg-white/95 text-ink shadow-card"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
                <Link to={`/results/${item.id}`}>
                  <img src={item.thumbnail_url} alt="" className="aspect-[3/4] w-full object-cover" />
                  <div className="px-3 py-3 text-sm">
                    <p className="capitalize">{label}</p>
                    <p className="text-xs text-muted">{Math.round((item.confidence || 0) * 100)}% confidence</p>
                  </div>
                </Link>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
