import { useRef, useState } from "react";

type Props = {
  before: string;
  after: string;
  checker: boolean;
};

export default function CompareSlider({ before, after, checker }: Props) {
  const [position, setPosition] = useState(52);
  const frame = useRef<HTMLDivElement>(null);

  function move(clientX: number) {
    const bounds = frame.current?.getBoundingClientRect();
    if (!bounds) return;
    const next = ((clientX - bounds.left) / bounds.width) * 100;
    setPosition(Math.min(96, Math.max(4, next)));
  }

  return (
    <div
      ref={frame}
      className={`relative overflow-hidden rounded-xl border border-line ${checker ? "checker" : "bg-white"}`}
      onPointerDown={(event) => {
        (event.currentTarget as HTMLDivElement).setPointerCapture(event.pointerId);
        move(event.clientX);
      }}
      onPointerMove={(event) => {
        if (event.buttons !== 1 && event.pointerType === "mouse") return;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) move(event.clientX);
      }}
    >
      <img src={after} alt="Extracted garment" className="block max-h-[72vh] w-full object-contain" />
      <img
        src={before}
        alt="Original photograph"
        className="absolute inset-0 h-full w-full object-contain"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      />
      <div className="absolute inset-y-0" style={{ left: `${position}%` }}>
        <div className="h-full w-px -translate-x-1/2 bg-ink/80" />
        <div className="absolute top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-[10px] font-medium shadow-card">
          drag
        </div>
      </div>
      <div className="pointer-events-none absolute left-3 top-3 rounded-full bg-white/90 px-2 py-1 text-[11px] text-ink">
        Original
      </div>
      <div className="pointer-events-none absolute right-3 top-3 rounded-full bg-white/90 px-2 py-1 text-[11px] text-ink">
        Garment
      </div>
    </div>
  );
}
