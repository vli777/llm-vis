"use client";
import { useRef, useState } from "react";
import { apiPostForm } from "../lib/api";

export function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const upload = async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    try {
      await apiPostForm("/upload", form);
      onUploaded();
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950 p-4">
      <div>
        <div className="font-medium">Upload CSV</div>
        <div className="text-sm text-slate-400">Drag & drop or choose a file. Stored in memory for this session.</div>
      </div>
      <div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => e.target.files && upload(e.target.files[0])}
        />
        <button
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 hover:bg-slate-800 disabled:opacity-60"
        >
          {busy ? "Uploading…" : "Choose file"}
        </button>
      </div>
    </div>
  );
}
