import { v4 as uuidv4 } from "uuid";

export function getSessionId(): string {
  const KEY = "ai-data-vis-session-id";
  let sid = typeof window !== "undefined" ? localStorage.getItem(KEY) : null;
  if (!sid) {
    sid = uuidv4();
    if (typeof window !== "undefined") localStorage.setItem(KEY, sid);
  }
  return sid;
}

function base(): string {
  const b = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  return b.endsWith("/") ? b.slice(0, -1) : b;
}

function joinUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${base()}${path.startsWith("/") ? "" : "/"}${path}`;
}

// Accept any HeadersInit and convert to a plain object
type HeadersInitLoose = HeadersInit | Record<string, string>;

function toRecord(h?: HeadersInitLoose): Record<string, string> {
  if (!h) return {};
  if (h instanceof Headers) {
    const obj: Record<string, string> = {};
    h.forEach((v, k) => {
      obj[k] = v;
    });
    return obj;
  }
  if (Array.isArray(h)) {
    const obj: Record<string, string> = {};
    for (const [k, v] of h) obj[String(k)] = String(v);
    return obj;
  }
  return { ...(h as Record<string, string>) };
}

type Opts = Omit<RequestInit, "headers"> & {
  headers?: HeadersInitLoose;
  /** Abort if running longer than this many ms (creates an internal AbortController). */
  timeoutMs?: number;
};

/** Core fetch with X-Session-Id header, AbortSignal and optional timeout. */
export async function apiFetch(path: string, opts: Opts = {}) {
  const url = joinUrl(path);

  const mergedHeaders: Record<string, string> = {
    ...toRecord(opts.headers),
    "X-Session-Id": getSessionId(),
  };

  // Handle timeout by racing an internal AbortController with any provided signal.
  const hasTimeout = typeof opts.timeoutMs === "number" && opts.timeoutMs! > 0;
  const outerSignal = opts.signal;
  const ctrl = hasTimeout ? new AbortController() : null;
  const signal = ctrl ? ctrl.signal : outerSignal;

  let timeoutId: any;
  if (ctrl) {
    timeoutId = setTimeout(
      () => ctrl.abort(new DOMException("Request timed out", "AbortError")),
      opts.timeoutMs
    );
    // If caller aborts, propagate to our controller too:
    if (outerSignal) {
      const onAbort = () =>
        ctrl.abort(
          outerSignal.reason ?? new DOMException("Aborted", "AbortError")
        );
      if (outerSignal.aborted) onAbort();
      else outerSignal.addEventListener("abort", onAbort, { once: true });
    }
  }

  try {
    const res = await fetch(url, { ...opts, headers: mergedHeaders, signal });
    if (!res.ok) {
      // Try to get helpful error text
      const ct = res.headers.get("content-type") || "";
      let detail = "";
      try {
        if (ct.includes("application/json")) {
          const j = await res.json();
          detail =
            typeof j === "string"
              ? j
              : j.detail || j.error || JSON.stringify(j);
        } else {
          detail = await res.text();
        }
      } catch {
        /* ignore */
      }
      throw new Error(`HTTP ${res.status}: ${detail || res.statusText}`);
    }
    return res;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function apiGetJSON<T = any>(
  path: string,
  opts: Omit<Opts, "method" | "body"> = {}
): Promise<T> {
  const res = await apiFetch(path, { ...opts, method: "GET" });
  return res.json();
}

export async function apiPostJSON<T = any>(
  path: string,
  body: any,
  opts: Omit<Opts, "method" | "body"> = {}
): Promise<T> {
  const res = await apiFetch(path, {
    ...opts,
    method: "POST",
    headers: { "Content-Type": "application/json", ...toRecord(opts.headers) },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function apiPostForm<T = any>(
  path: string,
  form: FormData,
  opts: Omit<Opts, "method" | "body" | "headers"> = {}
): Promise<T> {
  // browser adds the multipart boundary Content-Type
  const res = await apiFetch(path, { ...opts, method: "POST", body: form });
  return res.json();
}

export async function apiFetchRaw(path: string, opts: Opts = {}) {
  const url = joinUrl(path);

  const mergedHeaders: Record<string, string> = {
    ...toRecord(opts.headers),
    "X-Session-Id": getSessionId(),
  };

  const hasTimeout = typeof opts.timeoutMs === "number" && opts.timeoutMs! > 0;
  const outerSignal = opts.signal;
  const ctrl = hasTimeout ? new AbortController() : null;
  const signal = ctrl ? ctrl.signal : outerSignal;

  let timeoutId: any;
  if (ctrl) {
    timeoutId = setTimeout(
      () => ctrl.abort(new DOMException("Request timed out", "AbortError")),
      opts.timeoutMs
    );
    if (outerSignal) {
      const onAbort = () =>
        ctrl.abort(
          outerSignal.reason ?? new DOMException("Aborted", "AbortError")
        );
      if (outerSignal.aborted) onAbort();
      else outerSignal.addEventListener("abort", onAbort, { once: true });
    }
  }

  try {
    return await fetch(url, { ...opts, headers: mergedHeaders, signal });
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function apiPostFormRaw(
  path: string,
  form: FormData,
  opts: Omit<Opts, "method" | "body" | "headers"> = {}
) {
  const res = await apiFetchRaw(path, { ...opts, method: "POST", body: form });
  let data: any = null;
  try {
    data = await res.clone().json();
  } catch {
    /* maybe text or empty */
  }
  return { res, data };
}

export async function apiPostJSONRaw(
  path: string,
  body: any,
  opts: Omit<Opts, "method" | "body"> = {}
) {
  const res = await apiFetchRaw(path, {
    ...opts,
    method: "POST",
    headers: { "Content-Type": "application/json", ...toRecord(opts.headers) },
    body: JSON.stringify(body),
  });
  let data: any = null;
  try {
    data = await res.clone().json();
  } catch {}
  return { res, data };
}
