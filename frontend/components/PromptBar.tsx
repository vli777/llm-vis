"use client";
import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

type Props = {
  onSubmit: (prompt: string) => void | Promise<void>;
  placeholder?: string;
  disabled?: boolean;
};

export function PromptBar({ onSubmit, placeholder, disabled = false }: Props) {
  const [text, setText] = useState("");
  const composingRef = useRef(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { if (!disabled) inputRef.current?.focus(); }, [disabled]);

  const submit = async () => {
    const value = text.trim();
    if (!value || disabled) return;
    try {
      await onSubmit(value);
      setText("");
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); if (!composingRef.current) submit(); }}
      className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2"
      aria-busy={disabled || undefined}
    >
      {disabled ? (
        <Loader2 className="h-5 w-5 animate-spin text-slate-400" aria-hidden="true" />
      ) : (
        <span className="h-5 w-5" aria-hidden="true" />
      )}

      <input
        ref={inputRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={placeholder || "Describe a chart"}
        disabled={disabled}
        onCompositionStart={() => { composingRef.current = true; }}
        onCompositionEnd={() => { composingRef.current = false; }}
        className="flex-1 bg-transparent py-2 text-slate-200 outline-none placeholder:text-slate-500 disabled:opacity-60"
        aria-label="Prompt"
        autoComplete="off"
        spellCheck={false}
      />

      <button
        type="submit"
        disabled={disabled || !text.trim()}
        className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 hover:bg-slate-800 disabled:opacity-60"
      >
        Generate
      </button>
    </form>
  );
}
