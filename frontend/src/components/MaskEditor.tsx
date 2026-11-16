import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Eraser, Paintbrush } from "lucide-react";

type Props = {
  originalUrl: string;
  maskUrl: string;
  disabled: boolean;
  onApply: (blob: Blob, regenerate: boolean) => void;
};

export default function MaskEditor({ originalUrl, maskUrl, disabled, onApply }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const photoRef = useRef<HTMLImageElement | null>(null);
  const drawing = useRef(false);
  const [tool, setTool] = useState<"erase" | "restore">("erase");
  const [size, setSize] = useState(36);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    const photo = new Image();
    const maskImage = new Image();
    photo.onload = () => {
      maskImage.onload = () => {
        if (cancelled) return;
        const mask = document.createElement("canvas");
        mask.width = photo.naturalWidth;
        mask.height = photo.naturalHeight;
        const context = mask.getContext("2d");
        if (!context) return;
        context.drawImage(maskImage, 0, 0, mask.width, mask.height);
        // The stored mask is grayscale. Canvas compositing reads alpha, and
        // the API reads brightness, so both channels carry the same keep/drop value.
        const pixels = context.getImageData(0, 0, mask.width, mask.height);
        const data = pixels.data;
        for (let index = 0; index < data.length; index += 4) {
          const keep = data[index];
          data[index] = keep;
          data[index + 1] = keep;
          data[index + 2] = keep;
          data[index + 3] = keep;
        }
        context.putImageData(pixels, 0, 0);
        maskRef.current = mask;
        photoRef.current = photo;
        paint();
        setReady(true);
      };
      maskImage.src = maskUrl;
    };
    photo.src = originalUrl;
    return () => {
      cancelled = true;
    };
  }, [originalUrl, maskUrl]);

  function paint() {
    const canvas = canvasRef.current;
    const mask = maskRef.current;
    const photo = photoRef.current;
    if (!canvas || !mask || !photo) return;
    canvas.width = mask.width;
    canvas.height = mask.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(photo, 0, 0);
    context.globalCompositeOperation = "destination-in";
    context.drawImage(mask, 0, 0);
    context.globalCompositeOperation = "source-over";
  }

  function point(event: PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    const mask = maskRef.current;
    if (!canvas || !mask) return null;
    const bounds = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * mask.width,
      y: ((event.clientY - bounds.top) / bounds.height) * mask.height,
    };
  }

  function stamp(event: PointerEvent<HTMLCanvasElement>) {
    const mask = maskRef.current;
    const location = point(event);
    if (!mask || !location) return;
    const context = mask.getContext("2d");
    if (!context) return;
    if (tool === "erase") {
      context.globalCompositeOperation = "destination-out";
      context.fillStyle = "#000";
    } else {
      context.globalCompositeOperation = "source-over";
      context.fillStyle = "#fff";
    }
    context.beginPath();
    context.arc(location.x, location.y, size, 0, Math.PI * 2);
    context.fill();
    context.globalCompositeOperation = "source-over";
    paint();
  }

  async function exportMask(regenerate: boolean) {
    const mask = maskRef.current;
    if (!mask) return;
    const blob = await new Promise<Blob | null>((resolve) => mask.toBlob(resolve, "image/png"));
    if (blob) onApply(blob, regenerate);
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setTool("erase")}
          className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm ${tool === "erase" ? "border-pine bg-moss text-pine" : "border-line bg-white"}`}
        >
          <Eraser size={14} /> Erase
        </button>
        <button
          type="button"
          onClick={() => setTool("restore")}
          className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm ${tool === "restore" ? "border-pine bg-moss text-pine" : "border-line bg-white"}`}
        >
          <Paintbrush size={14} /> Restore
        </button>
        <label className="ml-auto flex items-center gap-2 text-xs text-muted">
          Brush
          <input
            type="range"
            min={8}
            max={120}
            value={size}
            onChange={(event) => setSize(Number(event.target.value))}
          />
        </label>
      </div>
      <canvas
        ref={canvasRef}
        className="checker max-h-[520px] w-full rounded-xl border border-line object-contain"
        onPointerDown={(event) => {
          if (!ready || disabled) return;
          drawing.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          stamp(event);
        }}
        onPointerMove={(event) => {
          if (!drawing.current) return;
          stamp(event);
        }}
        onPointerUp={() => {
          drawing.current = false;
        }}
      />
      <p className="text-xs leading-5 text-muted">
        Erase drops pixels. Restore paints the original photograph back. It does not invent hidden cloth.
        Regenerate estimates only the areas still covered by the body.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          disabled={!ready || disabled}
          onClick={() => exportMask(false)}
          className="rounded-md bg-ink px-3 py-2 text-sm text-white"
        >
          Apply mask edits
        </button>
        <button
          type="button"
          disabled={!ready || disabled}
          onClick={() => exportMask(true)}
          className="rounded-md border border-line bg-white px-3 py-2 text-sm"
        >
          Apply and regenerate
        </button>
      </div>
    </div>
  );
}
