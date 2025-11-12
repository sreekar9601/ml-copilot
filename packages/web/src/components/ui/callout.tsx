"use client";

import * as React from "react";
import { AlertCircle, AlertTriangle, Info, Lightbulb, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface CalloutProps {
  type?: "info" | "warning" | "tip" | "success" | "danger";
  title?: string;
  children: React.ReactNode;
  className?: string;
}

const calloutConfig = {
  info: {
    icon: Info,
    styles: "bg-blue-50 border-blue-200 text-blue-900 dark:bg-blue-950/30 dark:border-blue-900 dark:text-blue-100",
    iconStyles: "text-blue-600 dark:text-blue-400",
  },
  warning: {
    icon: AlertTriangle,
    styles: "bg-yellow-50 border-yellow-200 text-yellow-900 dark:bg-yellow-950/30 dark:border-yellow-900 dark:text-yellow-100",
    iconStyles: "text-yellow-600 dark:text-yellow-400",
  },
  tip: {
    icon: Lightbulb,
    styles: "bg-purple-50 border-purple-200 text-purple-900 dark:bg-purple-950/30 dark:border-purple-900 dark:text-purple-100",
    iconStyles: "text-purple-600 dark:text-purple-400",
  },
  success: {
    icon: CheckCircle2,
    styles: "bg-green-50 border-green-200 text-green-900 dark:bg-green-950/30 dark:border-green-900 dark:text-green-100",
    iconStyles: "text-green-600 dark:text-green-400",
  },
  danger: {
    icon: AlertCircle,
    styles: "bg-red-50 border-red-200 text-red-900 dark:bg-red-950/30 dark:border-red-900 dark:text-red-100",
    iconStyles: "text-red-600 dark:text-red-400",
  },
};

export function Callout({ type = "info", title, children, className }: CalloutProps) {
  const config = calloutConfig[type];
  const Icon = config.icon;

  return (
    <div className={cn("rounded-lg border p-4 w-full max-w-full overflow-hidden", config.styles, className)}>
      <div className="flex items-start gap-3">
        <Icon className={cn("h-5 w-5 mt-0.5 flex-shrink-0", config.iconStyles)} />
        <div className="flex-1 space-y-1 min-w-0">
          {title && <div className="font-semibold text-sm break-words">{title}</div>}
          <div className="text-sm leading-relaxed break-words">{children}</div>
        </div>
      </div>
    </div>
  );
}

