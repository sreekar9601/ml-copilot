"use client";

import { Source } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Code, BookOpen, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface CitationsProps {
  sources: Source[];
  className?: string;
}

export function Citations({ sources, className }: CitationsProps) {

  if (!sources || sources.length === 0) {
    return null;
  }

  const getVendorColor = (vendor?: string) => {
    switch (vendor?.toLowerCase()) {
      case 'aws': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
      case 'kubernetes': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'mlflow': return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      case 'pytorch': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'docker': return 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getDocTypeIcon = (docType?: string) => {
    switch (docType) {
      case 'tutorial': return <BookOpen className="h-3 w-3" />;
      case 'api_reference': return <Code className="h-3 w-3" />;
      case 'deployment_guide': return <Zap className="h-3 w-3" />;
      default: return <BookOpen className="h-3 w-3" />;
    }
  };


  // Filter and show only relevant sources
  // Only show sources with >40% relevance to avoid showing completely irrelevant results
  const relevantSources = sources
    .filter(source => source.relevance_score > 0.4)
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .slice(0, 5);  // Top 5 at most
  
  // If we have no relevant sources after filtering, just show the top 3
  const topSources = relevantSources.length > 0 ? relevantSources : sources
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .slice(0, 3);

  return (
    <div className={cn("space-y-2", className)}>
      <h3 className="text-sm font-semibold text-muted-foreground">
        Sources ({topSources.length} most relevant)
      </h3>
      <div className="space-y-1">
        {topSources.map((source, index) => {
          // Create a more specific reference URL
          let referenceUrl = source.url;
          
          // If we have an anchor_link, use it
          if (source.anchor_link && source.anchor_link !== source.url) {
            referenceUrl = source.anchor_link;
          } else if (source.heading_path) {
            // Generate anchor from heading path
            const anchor = source.heading_path
              .toLowerCase()
              .replace(/[^a-z0-9\s-]/g, '')
              .replace(/\s+/g, '-')
              .replace(/-+/g, '-')
              .replace(/^-+|-+$/g, '');
            referenceUrl = `${source.url}#${anchor}`;
          }
          
          // Create a more descriptive reference title
          const referenceTitle = source.heading_path 
            ? `${source.title} - ${source.heading_path}`
            : source.title;
          
          // Determine if this is a highly relevant source (top 40% relevance)
          const isHighlyRelevant = source.relevance_score > 0.6;
          
          return (
            <div 
              key={source.chunk_id} 
              className={cn(
                "flex items-center gap-3 p-2 rounded-lg transition-colors group cursor-pointer",
                isHighlyRelevant 
                  ? "bg-primary/10 hover:bg-primary/20 border border-primary/30" 
                  : "bg-muted/20 hover:bg-muted/40"
              )}
              onClick={() => window.open(referenceUrl, '_blank')}
              title={`Click to view: ${referenceTitle}`}
            >
              <div className={cn(
                "flex-shrink-0 w-2 h-2 rounded-full",
                isHighlyRelevant ? "bg-primary animate-pulse" : "bg-primary/50"
              )}></div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className={cn(
                    "text-sm hover:text-primary transition-colors font-medium",
                    isHighlyRelevant ? "text-foreground" : "text-foreground/80"
                  )}>
                    {source.title}
                  </span>
                  {source.vendor && (
                    <Badge 
                      variant="secondary" 
                      className={cn("text-xs px-2 py-0.5", getVendorColor(source.vendor))}
                    >
                      {source.vendor}
                    </Badge>
                  )}
                  <Badge 
                    variant="outline"
                    className={cn(
                      "text-xs px-2 py-0.5",
                      isHighlyRelevant 
                        ? "bg-primary/20 text-primary border-primary/40" 
                        : "bg-muted/50 text-muted-foreground"
                    )}
                  >
                    Highly relevant
                  </Badge>
                </div>
                {source.heading_path && (
                  <div className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                    <span>📍</span>
                    <span>{source.heading_path}</span>
                  </div>
                )}
                {source.topics && source.topics.length > 0 && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Topics: {source.topics.slice(0, 3).join(', ')}
                    {source.topics.length > 3 && ` +${source.topics.length - 3} more`}
                  </div>
                )}
              </div>
              <ExternalLink className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
          );
        })}
      </div>
    </div>
  );
}
