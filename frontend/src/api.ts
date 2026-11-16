import type { ExtractOptions, Health, HistoryResponse, Job, ResultRecord, Upload } from "./types";

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item: { msg?: string }) => item.msg || "Invalid request").join(" ");
    }
  } catch {
    return response.statusText || "Request failed";
  }
  return "Request failed";
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  history: () => request<HistoryResponse>("/api/results?limit=8"),
  upload: async (file: File): Promise<Upload> => {
    const body = new FormData();
    body.append("file", file);
    return request<Upload>("/api/uploads", { method: "POST", body });
  },
  start: (imageId: string, options: ExtractOptions) =>
    request<Job>("/api/extractions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId, ...options }),
    }),
  job: (jobId: string) => request<Job>(`/api/extractions/${jobId}`),
  continueJob: (
    jobId: string,
    garmentIds: string[],
    productTag: string,
    options: Pick<ExtractOptions, "reconstruct" | "refine_sam2">,
  ) =>
    request<Job>(`/api/extractions/${jobId}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        garment_ids: garmentIds,
        product_tag: productTag,
        reconstruct: options.reconstruct,
        refine_sam2: options.refine_sam2,
      }),
    }),
  result: (resultId: string) => request<ResultRecord>(`/api/results/${resultId}`),
  setTag: (resultId: string, productTag: string) =>
    request<ResultRecord>(`/api/results/${resultId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_tag: productTag }),
    }),
  editMask: async (resultId: string, mask: Blob, regenerate: boolean) => {
    const body = new FormData();
    body.append("mask", mask, "mask.png");
    return request<Job>(`/api/results/${resultId}/mask?regenerate=${regenerate ? "true" : "false"}`, {
      method: "POST",
      body,
    });
  },
  regenerate: (resultId: string) => request<Job>(`/api/results/${resultId}/reconstruct`, { method: "POST" }),
  remove: async (resultId: string) => {
    const response = await fetch(`/api/results/${resultId}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await readError(response));
  },
};
