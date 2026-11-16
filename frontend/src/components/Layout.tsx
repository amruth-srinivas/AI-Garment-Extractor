import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { LayoutDashboard, Shirt } from "lucide-react";
import { api } from "../api";
import type { Health } from "../types";

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const gpu = health == null
    ? "Checking runtime…"
    : health.gpu.available
      ? `${health.gpu.name} · ${health.gpu.vram_mb} MB`
      : "CPU fallback";

  return (
    <div className="min-h-screen md:grid md:grid-cols-[240px_1fr]">
      <aside className="border-b border-line bg-white md:min-h-screen md:border-b-0 md:border-r">
        <div className="flex items-center gap-3 px-5 py-5 md:px-6 md:py-7">
          <div className="grid h-9 w-9 place-items-center rounded-md bg-pine text-white">
            <Shirt size={18} />
          </div>
          <div>
            <p className="font-display text-xl leading-none text-ink">GarmentAI</p>
            <p className="mt-1 text-xs text-muted">Clothing extraction</p>
          </div>
        </div>
        <nav className="flex gap-2 px-4 pb-4 md:block md:px-3 md:pb-0">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `block rounded-md px-3 py-2 text-sm ${isActive ? "bg-moss text-pine" : "text-ink hover:bg-paper"}`
            }
          >
            <span className="inline-flex items-center gap-2">
              <LayoutDashboard size={16} /> Dashboard
            </span>
          </NavLink>
          <NavLink
            to="/new"
            className={({ isActive }) =>
              `block rounded-md px-3 py-2 text-sm ${isActive ? "bg-moss text-pine" : "text-ink hover:bg-paper"}`
            }
          >
            <span className="inline-flex items-center gap-2">
              <Shirt size={16} /> New extraction
            </span>
          </NavLink>
        </nav>
        <div className="hidden px-6 pb-6 pt-10 text-xs leading-5 text-muted md:block">
          <p className="font-medium text-ink">Runtime</p>
          <p className="mt-1">{gpu}</p>
          <p className="mt-3">
            Hidden cloth filled by the model is an estimate, not the fabric that was covered.
          </p>
        </div>
      </aside>
      <main className="px-4 py-6 md:px-10 md:py-8">
        <Outlet context={{ health }} />
      </main>
    </div>
  );
}
