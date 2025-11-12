"use client";

import * as React from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

interface CodeBlockProps {
  code: string;
  language?: string;
  className?: string;
}

export function CodeBlock({ code, language = "python", className }: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div className={cn("relative group w-full max-w-full", className)}>
      <div className="absolute right-2 top-2 z-10">
        <Button
          size="sm"
          variant="ghost"
          className="h-7 w-7 p-0 opacity-0 group-hover:opacity-100 transition-opacity bg-muted hover:bg-muted/80"
          onClick={copyToClipboard}
          title={copied ? "Copied!" : "Copy code"}
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-green-600" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>
      
      <div className="rounded-lg bg-muted/50 border border-border overflow-hidden w-full">
        {language && (
          <div className="px-4 py-2 bg-muted/80 border-b border-border text-xs text-muted-foreground font-mono">
            {language}
          </div>
        )}
        <pre className="p-4 overflow-x-auto max-w-full">
          <code className="text-sm font-mono leading-relaxed block">{code}</code>
        </pre>
      </div>
    </div>
  );
}

