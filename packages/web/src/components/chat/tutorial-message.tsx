"use client";

import * as React from "react";
import { TutorialResponse, Citation } from "@/types/chat";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CodeBlock } from "@/components/ui/code-block";
import { Callout } from "@/components/ui/callout";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { 
  CheckCircle2, 
  Terminal, 
  FileCode, 
  AlertCircle, 
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Circle
} from "lucide-react";
import { cn } from "@/lib/utils";

interface TutorialMessageProps {
  tutorial: TutorialResponse;
}

export function TutorialMessage({ tutorial }: TutorialMessageProps) {
  const { tutorial: content, total_citations, processing_time } = tutorial;
  const [expandedSteps, setExpandedSteps] = React.useState<Set<number>>(new Set([0])); // First step open by default
  const [prereqsOpen, setPrereqsOpen] = React.useState(true);
  const [checkedPrereqs, setCheckedPrereqs] = React.useState<Set<number>>(new Set());
  const [referencesOpen, setReferencesOpen] = React.useState(false);
  const [activeStep, setActiveStep] = React.useState<number | null>(null);

  const stepRefs = React.useRef<(HTMLDivElement | null)[]>([]);

  // Collect all unique citations
  const allCitations = React.useMemo(() => {
    const citationMap = new Map<string, Citation>();
    content.steps.forEach(step => {
      step.citations.forEach(citation => {
        if (!citationMap.has(citation.url)) {
          citationMap.set(citation.url, citation);
        }
      });
    });
    return Array.from(citationMap.values());
  }, [content.steps]);

  const toggleStep = (idx: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const togglePrereq = (idx: number) => {
    setCheckedPrereqs(prev => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const scrollToStep = (idx: number) => {
    stepRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setExpandedSteps(prev => new Set(prev).add(idx));
  };

  // Detect which step is in viewport
  React.useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const idx = stepRefs.current.findIndex(ref => ref === entry.target);
            if (idx !== -1) {
              setActiveStep(idx);
            }
          }
        });
      },
      { threshold: 0.5 }
    );

    stepRefs.current.forEach(ref => {
      if (ref) observer.observe(ref);
    });

    return () => observer.disconnect();
  }, []);

  // Parse explanation for tips and warnings
  const parseExplanation = (text: string) => {
    const tipRegex = /\*\*Tip:\*\*\s*(.*?)(?=\*\*|$)/gi;
    const warningRegex = /\*\*Warning:\*\*\s*(.*?)(?=\*\*|$)/gi;
    
    const tips = [];
    const warnings = [];
    let match;
    
    while ((match = tipRegex.exec(text)) !== null) {
      tips.push(match[1].trim());
    }
    
    while ((match = warningRegex.exec(text)) !== null) {
      warnings.push(match[1].trim());
    }
    
    // Remove tips and warnings from main text
    let cleanText = text
      .replace(/\*\*Tip:\*\*\s*.*?(?=\*\*|$)/gi, '')
      .replace(/\*\*Warning:\*\*\s*.*?(?=\*\*|$)/gi, '')
      .trim();
    
    return { cleanText, tips, warnings };
  };

  // Check if tutorial mentions costs or GPU
  const hasCostWarning = React.useMemo(() => {
    const contentText = JSON.stringify(content).toLowerCase();
    return contentText.includes('gpu') || 
           contentText.includes('a100') || 
           contentText.includes('h100') ||
           contentText.includes('cloud') ||
           contentText.includes('instance');
  }, [content]);

  return (
    <div className="flex flex-col lg:flex-row gap-4 w-full max-w-full overflow-hidden">
      {/* Mobile Step Indicator */}
      <div className="lg:hidden w-full">
        <Card className="p-3">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium truncate">
              Progress: {expandedSteps.size} of {content.steps.length} steps viewed
            </div>
            <Badge variant="outline" className="text-xs flex-shrink-0 ml-2">
              {Math.round((expandedSteps.size / content.steps.length) * 100)}%
            </Badge>
          </div>
        </Card>
      </div>

      {/* Step Navigator Sidebar - Desktop Only */}
      <div className="hidden lg:block w-64 flex-shrink-0 sticky top-4 h-fit">
        <Card className="p-4">
          <div className="space-y-1 mb-4">
            <h4 className="text-sm font-semibold text-muted-foreground">Tutorial Progress</h4>
            <div className="text-xs text-muted-foreground">
              {content.steps.length} steps • {(processing_time / 1000).toFixed(1)}s
            </div>
          </div>
          
          <div className="space-y-2">
            {/* Prerequisites in navigator */}
            <button
              onClick={() => {
                const prereqSection = document.getElementById('prerequisites');
                prereqSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }}
              className="w-full text-left flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 transition-colors text-sm"
            >
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
              <span>Prerequisites</span>
            </button>

            {/* Step list */}
            {content.steps.map((step, idx) => {
              const isActive = activeStep === idx;
              const isExpanded = expandedSteps.has(idx);
              
              return (
                <button
                  key={idx}
                  onClick={() => scrollToStep(idx)}
                  className={cn(
                    "w-full text-left flex items-start gap-2 p-2 rounded-md transition-colors text-sm",
                    isActive 
                      ? "bg-primary/10 text-primary font-medium" 
                      : "hover:bg-muted/50"
                  )}
                >
                  <div className={cn(
                    "flex items-center justify-center w-5 h-5 rounded-full text-xs flex-shrink-0 mt-0.5",
                    isActive 
                      ? "bg-primary text-primary-foreground font-bold" 
                      : "bg-muted text-muted-foreground"
                  )}>
                    {idx + 1}
                  </div>
                  <span className="line-clamp-2 break-words">{step.title}</span>
                </button>
              );
            })}
          </div>
        </Card>
      </div>

      {/* Main Content */}
      <div className="flex-1 space-y-4 min-w-0 max-w-full overflow-hidden">
        {/* Tutorial Header */}
        <div className="space-y-2 w-full">
          <div className="flex items-center gap-2">
            <FileCode className="h-5 w-5 text-primary flex-shrink-0" />
            <h3 className="text-lg font-semibold break-words">{content.title}</h3>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
            <Badge variant="outline" className="font-mono">
              {content.steps.length} steps
            </Badge>
            <Badge variant="outline" className="font-mono">
              {total_citations} citations
            </Badge>
            <Badge variant="outline" className="font-mono">
              {(processing_time / 1000).toFixed(1)}s
            </Badge>
          </div>
        </div>

        {/* Cost Warning Callout - Show if tutorial mentions GPU/Cloud costs */}
        {hasCostWarning && (
          <Callout type="warning" title="Cost Warning">
            This tutorial involves cloud GPU instances which may incur significant costs. 
            Remember to shut down your instances after use to avoid unexpected charges. 
            Monitor your cloud provider's billing dashboard regularly.
          </Callout>
        )}

        {/* Prerequisites - Collapsible with Interactive Checklist */}
        {content.prereqs && content.prereqs.length > 0 && (
          <Collapsible open={prereqsOpen} onOpenChange={setPrereqsOpen}>
            <Card className="p-4 bg-muted/30" id="prerequisites">
              <CollapsibleTrigger className="w-full flex items-center justify-between hover:opacity-80 transition-opacity">
                <h4 className="text-sm font-semibold flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  Prerequisites
                </h4>
                {prereqsOpen ? (
                  <ChevronUp className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                )}
              </CollapsibleTrigger>
              
              <CollapsibleContent className="mt-3">
                <div className="space-y-2">
                  {content.prereqs.map((prereq, idx) => (
                    <div 
                      key={idx} 
                      className="flex items-start gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors"
                    >
                      <Checkbox
                        checked={checkedPrereqs.has(idx)}
                        onCheckedChange={() => togglePrereq(idx)}
                        className="mt-0.5 flex-shrink-0"
                      />
                      <span className={cn(
                        "text-sm flex-1 break-words",
                        checkedPrereqs.has(idx) && "line-through text-muted-foreground"
                      )}>
                        {prereq}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  {checkedPrereqs.size} of {content.prereqs.length} completed
                </div>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        )}

        {/* Tutorial Steps - Collapsible */}
        <div className="space-y-3">
          {content.steps.map((step, idx) => {
            const isExpanded = expandedSteps.has(idx);
            const { cleanText, tips, warnings } = parseExplanation(step.explanation);
            
            return (
              <Collapsible 
                key={idx} 
                open={isExpanded} 
                onOpenChange={() => toggleStep(idx)}
              >
                <Card 
                  ref={(el) => { stepRefs.current[idx] = el; }}
                  className={cn(
                    "transition-all",
                    isExpanded && "ring-2 ring-primary/20"
                  )}
                >
                  {/* Step Header - Always Visible */}
                  <CollapsibleTrigger className="w-full p-4 hover:bg-muted/30 transition-colors">
                    <div className="flex items-start gap-3 w-full">
                      <div className="flex-shrink-0 mt-0.5">
                        <div className="flex items-center justify-center w-7 h-7 rounded-full bg-primary text-primary-foreground text-sm font-bold">
                          {idx + 1}
                        </div>
                      </div>
                      <div className="flex-1 text-left min-w-0">
                        <h4 className="font-semibold text-base break-words">{step.title}</h4>
                        {!isExpanded && (
                          <p className="text-sm text-muted-foreground mt-1 line-clamp-1 break-words">
                            {cleanText}
                          </p>
                        )}
                      </div>
                      <div className="flex-shrink-0 ml-2">
                        {isExpanded ? (
                          <ChevronUp className="h-5 w-5 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="h-5 w-5 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </CollapsibleTrigger>

                  {/* Step Content - Collapsible */}
                  <CollapsibleContent>
                    <div className="px-4 pb-4 space-y-4 w-full overflow-hidden">
                      {/* Step Explanation */}
                      <p className="text-sm leading-relaxed break-words">
                        {cleanText}
                      </p>

                      {/* Tips */}
                      {tips.map((tip, tipIdx) => (
                        <Callout key={`tip-${tipIdx}`} type="tip" title="Pro Tip">
                          {tip}
                        </Callout>
                      ))}

                      {/* Warnings */}
                      {warnings.map((warning, warnIdx) => (
                        <Callout key={`warn-${warnIdx}`} type="warning" title="Important Warning">
                          {warning}
                        </Callout>
                      ))}

                      {/* Commands */}
                      {step.commands && step.commands.length > 0 && (
                        <div className="space-y-2 w-full">
                          {step.commands.map((command, cmdIdx) => (
                            <div key={cmdIdx} className="relative w-full overflow-hidden">
                              <div className="flex items-start gap-2 rounded-md bg-muted/50 border border-border p-3 font-mono text-sm group overflow-x-auto">
                                <Terminal className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                                <code className="flex-1 break-all whitespace-pre-wrap">{command}</code>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Code Block with Syntax Highlighting */}
                      {step.code && (
                        <CodeBlock 
                          code={step.code} 
                          language={detectLanguage(step.code)}
                        />
                      )}

                      {/* Notes */}
                      {step.notes && step.notes.length > 0 && (
                        <div className="space-y-2">
                          {step.notes.map((note, noteIdx) => (
                            <Callout key={noteIdx} type="info">
                              {note}
                            </Callout>
                          ))}
                        </div>
                      )}

                      {/* Step Citations - Inline with tooltips */}
                      {step.citations && step.citations.length > 0 && (
                        <div className="pt-3 border-t w-full overflow-hidden">
                          <div className="text-xs font-semibold text-muted-foreground mb-2">
                            References for this step:
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <TooltipProvider>
                              {step.citations.map((citation, citIdx) => (
                                <Tooltip key={citIdx}>
                                  <TooltipTrigger asChild>
                                    <a
                                      href={citation.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-primary/10 hover:bg-primary/20 text-primary transition-colors max-w-full"
                                    >
                                      <span className="flex-shrink-0">[{citIdx + 1}]</span>
                                      <span className="max-w-[200px] truncate">{citation.title}</span>
                                      <ExternalLink className="h-3 w-3 flex-shrink-0" />
                                    </a>
                                  </TooltipTrigger>
                                  <TooltipContent side="bottom" className="max-w-sm break-words">
                                    <div className="space-y-1">
                                      <div className="font-semibold break-words">{citation.title}</div>
                                      {citation.source_vendor && (
                                        <div className="text-xs opacity-80">
                                          Source: {citation.source_vendor}
                                        </div>
                                      )}
                                      {citation.quote && (
                                        <div className="text-xs italic mt-2 break-words">
                                          "{citation.quote.substring(0, 150)}{citation.quote.length > 150 ? '...' : ''}"
                                        </div>
                                      )}
                                    </div>
                                  </TooltipContent>
                                </Tooltip>
                              ))}
                            </TooltipProvider>
                          </div>
                        </div>
                      )}
                    </div>
                  </CollapsibleContent>
                </Card>
              </Collapsible>
            );
          })}
        </div>

        {/* All References Section - Collapsible */}
        {allCitations.length > 0 && (
          <Collapsible open={referencesOpen} onOpenChange={setReferencesOpen}>
            <Card className="p-4 bg-muted/20">
              <CollapsibleTrigger className="w-full flex items-center justify-between hover:opacity-80 transition-opacity">
                <h4 className="text-sm font-semibold flex items-center gap-2">
                  <FileCode className="h-4 w-4" />
                  All References ({allCitations.length})
                </h4>
                {referencesOpen ? (
                  <ChevronUp className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                )}
              </CollapsibleTrigger>
              
              <CollapsibleContent className="mt-3">
                <div className="space-y-2 w-full overflow-hidden">
                  {allCitations.map((citation, idx) => (
                    <div 
                      key={idx}
                      className="flex items-start gap-2 p-2 rounded-md bg-background hover:bg-muted/50 transition-colors"
                    >
                      <Circle className="h-3 w-3 mt-1 text-primary flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <a
                          href={citation.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-primary hover:underline font-medium inline-flex items-center gap-1 break-words"
                        >
                          <span className="break-words">{citation.title}</span>
                          <ExternalLink className="h-3 w-3 flex-shrink-0" />
                        </a>
                        {citation.source_vendor && (
                          <Badge variant="secondary" className="ml-2 text-[10px] px-1.5 py-0">
                            {citation.source_vendor}
                          </Badge>
                        )}
                        {citation.quote && (
                          <p className="text-xs text-muted-foreground mt-1 italic break-words">
                            "{citation.quote.substring(0, 150)}{citation.quote.length > 150 ? '...' : ''}"
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        )}
      </div>
    </div>
  );
}

// Helper function to detect programming language
function detectLanguage(code: string): string {
  if (code.includes('import torch') || code.includes('import tensorflow')) {
    return 'python';
  }
  if (code.includes('const ') || code.includes('let ') || code.includes('function')) {
    return 'javascript';
  }
  if (code.includes('class ') && code.includes('public')) {
    return 'java';
  }
  if (code.includes('#!/bin/bash') || code.includes('apt-get')) {
    return 'bash';
  }
  if (code.includes('SELECT') || code.includes('FROM')) {
    return 'sql';
  }
  return 'python'; // default
}
