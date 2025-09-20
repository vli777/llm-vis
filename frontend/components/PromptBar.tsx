"use client";
import { useState } from "react";

export function PromptBar({ onSubmit, placeholder }: { onSubmit: (p: string) => void; placeholder?: string }) {
  const [text, setText] = useState("");

  return (
    <form onSubmit={(e) => { e.preventDefault(); if (text.trim()) { onSubmit(text.trim()); setText(""); } }}>
      <div style={{ display: "flex", gap: 8 }}>
        <input value={text} onChange={(e)=>setText(e.target.value)} placeholder={placeholder || "Describe a chart"} 
          style={{ flex: 1, padding: 12, borderRadius: 10, border: "1px solid #374151", background: "#0b1220", color: "#e5e7eb" }} />
        <button type="submit" style={{ padding: "12px 16px", borderRadius: 10, border: "1px solid #374151", background: "#0b1220", color: "#e5e7eb" }}>Generate</button>
      </div>
    </form>
  );
}
