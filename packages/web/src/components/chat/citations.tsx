"use client";

import { useState } from "react";
import { Source } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Code, Star, BookOpen, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface CitationsProps {
  sources: Source[];
  className?: string;
}

export function Citations({ sources, className }: CitationsProps) {
  const [expandedSource, setExpandedSource] = useState<string | null>(null);

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

  const getQualityStars = (score?: number) => {
    if (!score) return null;
    const stars = Math.round(score * 5);
    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <Star
            key={i}
            className={cn(
              "h-3 w-3",
              i < stars ? "fill-yellow-400 text-yellow-400" : "text-gray-300"
            )}
          />
        ))}
        <span className="text-xs text-muted-foreground ml-1">
          {score.toFixed(1)}
        </span>
      </div>
    );
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-muted-foreground">
          Sources ({sources.length})
        </h3>
      </div>
      
      <div className="space-y-2">
        {sources.map((source, index) => (
          <Card 
            key={source.chunk_id} 
            className="hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => setExpandedSource(
              expandedSource === source.chunk_id ? null : source.chunk_id
            )}
          >
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <CardTitle className="text-sm font-medium line-clamp-2">
                    {source.title}
                  </CardTitle>
                  <div className="flex items-center gap-2 mt-1">
                    {source.vendor && (
                      <Badge 
                        variant="secondary" 
                        className={cn("text-xs", getVendorColor(source.vendor))}
                      >
                        {source.vendor}
                      </Badge>
                    )}
                    {source.doc_type && (
                      <Badge variant="outline" className="text-xs">
                        {getDocTypeIcon(source.doc_type)}
                        <span className="ml-1 capitalize">
                          {source.doc_type.replace('_', ' ')}
                        </span>
                      </Badge>
                    )}
                    {source.has_code_examples && (
                      <Badge variant="outline" className="text-xs">
                        <Code className="h-3 w-3 mr-1" />
                        Code
                      </Badge>
                    )}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(source.url, '_blank');
                  }}
                >
                  <ExternalLink className="h-3 w-3" />
                </Button>
              </div>
            </CardHeader>
            
            {expandedSource === source.chunk_id && (
              <CardContent className="pt-0">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Relevance: {(source.relevance_score * 100).toFixed(0)}%</span>
                    {getQualityStars(source.quality_score)}
                  </div>
                  
                  {source.topics.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {source.topics.slice(0, 3).map((topic) => (
                        <Badge key={topic} variant="outline" className="text-xs">
                          {topic}
                        </Badge>
                      ))}
                      {source.topics.length > 3 && (
                        <Badge variant="outline" className="text-xs">
                          +{source.topics.length - 3} more
                        </Badge>
                      )}
                    </div>
                  )}
                  
                  {source.technical_depth && (
                    <div className="text-xs text-muted-foreground">
                      Level: <span className="capitalize">{source.technical_depth}</span>
                    </div>
                  )}
                  
                  <div className="pt-2 border-t">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full"
                      onClick={(e) => {
                        e.stopPropagation();
                        window.open(source.url, '_blank');
                      }}
                    >
                      <ExternalLink className="h-3 w-3 mr-2" />
                      View Full Document
                    </Button>
                  </div>
                </div>
              </CardContent>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}
