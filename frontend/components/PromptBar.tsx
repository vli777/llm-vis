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
  const [focused, setFocused] = useState(false);
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
      className={`theme-card flex items-center gap-2 px-3 py-2 transition-opacity ${
        focused ? "opacity-100" : "opacity-60"
      }`}
      aria-busy={disabled || undefined}
    >
      {disabled ? (
        <Loader2 className="h-5 w-5 animate-spin theme-muted" aria-hidden="true" />
      ) : (
        <span className="h-5 w-5" aria-hidden="true" />
      )}

      <input
        ref={inputRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={placeholder || "Describe a chart"}
        disabled={disabled}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onCompositionStart={() => { composingRef.current = true; }}
        onCompositionEnd={() => { composingRef.current = false; }}
        className="theme-input flex-1 bg-transparent py-2 outline-none disabled:opacity-60"
        aria-label="Prompt"
        autoComplete="off"
        spellCheck={false}
      />

      <button
        type="submit"
        disabled={disabled || !text.trim()}
        className="theme-button rounded-md px-3 py-2 text-sm"
      >
        Submit
      </button>
    </form>
  );
}
