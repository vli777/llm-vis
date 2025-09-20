import { v4 as uuidv4 } from "uuid";

export function getSessionId(): string {
  const KEY = "ai-data-vis-session-id";
  let sid = localStorage.getItem(KEY);
  if (!sid) { sid = uuidv4(); localStorage.setItem(KEY, sid); }
  return sid;
}

function base() {
  return process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
}

// Accept any HeadersInit and convert to a plain object
type HeadersInitLoose = HeadersInit | Record<string, string>;

function toRecord(h?: HeadersInitLoose): Record<string, string> {
  if (!h) return {};
  if (h instanceof Headers) {
    const obj: Record<string, string> = {};
    h.forEach((v, k) => { obj[k] = v; });
    return obj;
  }
  if (Array.isArray(h)) {
    const obj: Record<string, string> = {};
    for (const [k, v] of h) obj[String(k)] = String(v);
    return obj;
  }
  return { ...(h as Record<string, string>) };
}

type Opts = Omit<RequestInit, "headers"> & { headers?: HeadersInitLoose };

export async function apiFetch(path: string, opts: Opts = {}) {
  const url = path.startsWith("http") ? path : `${base()}${path}`;

  const mergedHeaders: Record<string, string> = {
    ...toRecord(opts.headers),
    "X-Session-Id": getSessionId(),
  };

  const res = await fetch(url, { ...opts, headers: mergedHeaders });
  if (!res.ok) {
    let detail = "";
    try { detail = await res.text(); } catch {}
    throw new Error(`HTTP ${res.status}: ${detail || res.statusText}`);
  }
  return res;
}

export async function apiGetJSON<T = any>(path: string): Promise<T> {
  const res = await apiFetch(path, { method: "GET" });
  return res.json();
}

export async function apiPostJSON<T = any>(path: string, body: any): Promise<T> {
  const res = await apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function apiPostForm<T = any>(path: string, form: FormData): Promise<T> {
  // browser will set the multipart boundary Content-Type
  const res = await apiFetch(path, { method: "POST", body: form });
  return res.json();
}
