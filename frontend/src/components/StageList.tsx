import { Check, Minus } from "lucide-react";
import type { Stage } from "../types";

export default function StageList({ stages }: { stages: Stage[] }) {
  return (
    <ol className="space-y-3">
      {stages.map((stage) => (
        <li key={stage.id} className="flex items-center gap-3 text-sm">
          <span
            className={`grid h-6 w-6 place-items-center rounded-full border ${
              stage.status === "done"
                ? "border-pine bg-pine text-white"
                : stage.status === "active"
                  ? "border-pine bg-white text-pine"
                  : stage.status === "skipped"
                    ? "border-line bg-paper text-muted"
                    : "border-line bg-white text-transparent"
            }`}
          >
            {stage.status === "skipped" ? <Minus size={12} /> : <Check size={12} />}
          </span>
          <span className={stage.status === "pending" ? "text-muted" : "text-ink"}>
            {stage.label}
            {stage.status === "skipped" ? " · skipped" : ""}
            {stage.status === "active" ? " · in progress" : ""}
          </span>
        </li>
      ))}
    </ol>
  );
}
