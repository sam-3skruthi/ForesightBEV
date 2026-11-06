export class ApiError extends Error {
  constructor(message: string, readonly code: string, readonly status: number) { super(message); }
}

async function parse<T>(response: Response, path: string): Promise<T> {
  if (response.ok) return response.json() as Promise<T>;
  let code = "http_error", message = `${path} returned ${response.status}`;
  try { const body = await response.json(); code = body.error?.code ?? code; message = body.error?.message ?? message; } catch { /* The body was not JSON; the status line stands. */ }
  throw new ApiError(message, code, response.status);
}

export const get = <T,>(path: string, signal?: AbortSignal) => fetch(`/api/v1${path}`, { signal }).then((r) => parse<T>(r, path));
export const post = <T,>(path: string, body: unknown) => fetch(`/api/v1${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) => parse<T>(r, path));
export const camera = (run: string, frame: number, name: string) => `/api/v1/runs/${run}/frames/${frame}/image/${name}`;
export const layer = (run: string, frame: number, name: string, cam?: string) => `/api/v1/runs/${run}/frames/${frame}/layer/${name}${cam ? `/${cam}` : ""}`;
