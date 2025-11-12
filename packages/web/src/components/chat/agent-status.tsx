"use client";

import { Button } from "@/components/ui/button";
import { X, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface AgentStatusProps {
  status: {
    type: "idle" | "thinking" | "tool_execution" | "responding" | "error";
    message?: string;
    tool?: string;
  };
  onCancel?: () => void;
  className?: string;
}

export function AgentStatus({ status, onCancel, className }: AgentStatusProps) {
  const getStatusDisplay = () => {
    switch (status.type) {
      case "thinking":
        return { text: "Thinking...", color: "text-blue-600 dark:text-blue-400" };
      case "tool_execution":
        return {
          text: status.tool 
            ? `${status.tool === "web_search" || status.tool === "tavily_search_results_json" || status.tool === "tavily_search_results" || status.tool === "tavily" 
                ? "Searching web" 
                : status.tool === "hybrid_doc_search" 
                  ? "Searching documentation"
                  : status.tool === "execute_python_code"
                    ? "Executing code"
                    : status.tool}` 
            : "Executing tool...",
          color: "text-purple-600 dark:text-purple-400",
        };
      case "responding":
        return { text: "Responding...", color: "text-green-600 dark:text-green-400" };
      default:
        return { text: "Processing...", color: "text-muted-foreground" };
    }
  };

  const display = getStatusDisplay();

  return (
    <div className={cn("px-6 py-3 bg-muted/30", className)}>
      <div className="flex items-center justify-between max-w-4xl mx-auto">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          <span className={cn("text-sm", display.color)}>{display.text}</span>
        </div>
        {onCancel && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onCancel}
            className="h-7 px-3 text-xs"
          >
            Cancel
          </Button>
        )}
      </div>
    </div>
  );
}

