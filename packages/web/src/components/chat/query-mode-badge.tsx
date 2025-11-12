"use client";

import { Badge } from "@/components/ui/badge";
import { QueryMode, getModeName, getActiveFeatures } from "@/lib/query-analyzer";
import { 
  Sparkles, 
  Network, 
  GraduationCap, 
  Search,
  Info
} from "lucide-react";
import * as Tooltip from "@radix-ui/react-tooltip";

interface QueryModeBadgeProps {
  mode: QueryMode;
  detectedVendors?: string[];
  reasoning?: string;
  className?: string;
}

const MODE_ICONS = {
  tutorial: GraduationCap,
  'multi-source': Network,
  advanced: Sparkles,
  standard: Search,
};

const MODE_COLORS = {
  tutorial: "bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/20",
  'multi-source': "bg-purple-500/10 text-purple-700 dark:text-purple-400 border-purple-500/20",
  advanced: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20",
  standard: "bg-slate-500/10 text-slate-700 dark:text-slate-400 border-slate-500/20",
};

export function QueryModeBadge({ 
  mode, 
  detectedVendors = [], 
  reasoning,
  className 
}: QueryModeBadgeProps) {
  const Icon = MODE_ICONS[mode];
  const modeName = getModeName(mode);
  const features = getActiveFeatures(mode);
  
  return (
    <Tooltip.Provider delayDuration={200}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <Badge 
            variant="outline" 
            className={`${MODE_COLORS[mode]} cursor-help ${className}`}
          >
            <Icon className="h-3 w-3 mr-1.5" />
            {modeName}
            {detectedVendors.length > 0 && (
              <span className="ml-1.5 opacity-70">
                • {detectedVendors.length} {detectedVendors.length === 1 ? 'vendor' : 'vendors'}
              </span>
            )}
          </Badge>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content 
            className="z-50 max-w-xs rounded-lg border bg-popover px-3 py-2 text-sm text-popover-foreground shadow-md"
            sideOffset={5}
          >
            <div className="space-y-2">
              <div className="font-semibold flex items-center gap-2">
                <Icon className="h-4 w-4" />
                {modeName}
              </div>
              
              {reasoning && (
                <p className="text-xs text-muted-foreground">
                  {reasoning}
                </p>
              )}
              
              {detectedVendors.length > 0 && (
                <div className="pt-1 border-t">
                  <p className="text-xs font-medium mb-1">Detected Technologies:</p>
                  <div className="flex flex-wrap gap-1">
                    {detectedVendors.map((vendor) => (
                      <Badge 
                        key={vendor} 
                        variant="secondary" 
                        className="text-[10px] px-1.5 py-0"
                      >
                        {vendor}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="pt-1 border-t">
                <p className="text-xs font-medium mb-1.5 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Active RAG Features:
                </p>
                <ul className="space-y-0.5 text-xs text-muted-foreground">
                  {features.map((feature, idx) => (
                    <li key={idx} className="flex items-start gap-1.5">
                      <span className="text-emerald-600 dark:text-emerald-400">✓</span>
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <Tooltip.Arrow className="fill-popover" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

/**
 * Compact version for inline display
 */
export function QueryModeBadgeCompact({ 
  mode, 
  className 
}: { 
  mode: QueryMode; 
  className?: string 
}) {
  const Icon = MODE_ICONS[mode];
  
  return (
    <Badge 
      variant="outline" 
      className={`${MODE_COLORS[mode]} ${className}`}
    >
      <Icon className="h-3 w-3" />
    </Badge>
  );
}




