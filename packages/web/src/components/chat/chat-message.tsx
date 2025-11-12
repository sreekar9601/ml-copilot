"use client";

import React, { useState } from "react";
import { Message } from "@/types/chat";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bot, User, Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Citations } from "./citations";
import { TutorialMessage } from "./tutorial-message";
import { QueryModeBadge } from "./query-mode-badge";
import { cn } from "@/lib/utils";
import { processMarkdownContent } from "@/lib/markdown-utils";

interface ChatMessageProps {
  message: Message;
  className?: string;
}

export function ChatMessage({ message, className }: ChatMessageProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={cn(
        "flex gap-4 w-full",
        isUser ? "justify-end" : "justify-start",
        className
      )}
    >
      {!isUser && (
        <Avatar className="h-8 w-8 mt-1 flex-shrink-0 bg-gradient-to-br from-blue-500 to-purple-600">
          <AvatarFallback>
            <Bot className="h-4 w-4 text-white" />
          </AvatarFallback>
        </Avatar>
      )}
      
      <div
        className={cn(
          "relative group flex-1 min-w-0",
          isUser && "max-w-[75%]"
        )}
      >
        {!isUser && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10 h-7 w-7 p-0 rounded-lg bg-background border shadow-sm"
          >
            {copied ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        )}
        <div className={cn(
          "rounded-2xl px-5 py-3",
          isUser
            ? "bg-blue-600 text-white ml-auto"
            : "bg-muted/50"
        )}>
          {isUser ? (
            <p className="text-[15px] leading-relaxed break-words whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="space-y-4 w-full max-w-full overflow-hidden">
              {/* Query Mode Badge */}
              {message.queryMode && message.queryAnalysis && (
                <div className="pb-2 border-b">
                  <QueryModeBadge 
                    mode={message.queryMode}
                    detectedVendors={message.queryAnalysis.detectedVendors}
                    reasoning={message.queryAnalysis.reasoning}
                  />
                </div>
              )}

              {/* Tutorial Mode - Special Rendering */}
              {message.tutorial ? (
                <TutorialMessage tutorial={message.tutorial} />
              ) : (
                <div className="prose prose-sm md:prose-base max-w-none dark:prose-invert prose-headings:font-semibold prose-p:leading-relaxed prose-p:text-foreground/90 prose-strong:text-foreground prose-strong:font-medium prose-code:text-sm prose-pre:bg-muted prose-pre:text-foreground w-full overflow-x-auto">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => {
                      // Process text to convert [1], [2] citations to superscripts
                      const processChildren = (node: any): any => {
                        if (typeof node === 'string') {
                          const parts = node.split(/(\[\d+\])/g);
                          return parts.map((part, i) => {
                            const match = part.match(/\[(\d+)\]/);
                            if (match) {
                              return (
                                <sup key={i} className="text-blue-600 dark:text-blue-400 font-semibold cursor-help mx-0.5 hover:underline" title="See reference below">
                                  {part}
                                </sup>
                              );
                            }
                            return part;
                          });
                        }
                        return node;
                      };
                      
                      const processedChildren = React.Children.map(children, processChildren);
                      
                      return (
                        <p className="mb-3 last:mb-0 leading-relaxed break-words whitespace-pre-wrap">
                          {processedChildren}
                        </p>
                      );
                    },
                    ul: ({ children }) => (
                      <ul className="list-disc ml-4 mb-3 space-y-1">
                        {children}
                      </ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="list-decimal ml-4 mb-3 space-y-1">
                        {children}
                      </ol>
                    ),
                    li: ({ children }) => (
                      <li className="leading-relaxed break-words">
                        {children}
                      </li>
                    ),
                    strong: ({ children }) => (
                      <strong className="font-semibold text-foreground">
                        {children}
                      </strong>
                    ),
                    code: ({ children, className }) => {
                      const isBlock = className?.includes("language-");
                      if (isBlock) {
                        const language = className?.replace('language-', '') || 'code';
                        return (
                          <div className="relative my-4 rounded-lg overflow-hidden border border-border/50">
                            <div className="flex items-center justify-between bg-muted/50 px-4 py-2 border-b border-border/50">
                              <span className="text-xs font-medium text-muted-foreground">{language}</span>
                            </div>
                            <pre className="bg-muted/30 p-4 overflow-x-auto max-w-full">
                              <code className={`${className} text-sm leading-relaxed block whitespace-pre break-words`}>{children}</code>
                            </pre>
                          </div>
                        );
                      }
                      
                      // Inline code
                      return (
                        <code className="px-1.5 py-0.5 rounded bg-muted text-foreground font-mono text-sm border border-border/50">
                          {children}
                        </code>
                      );
                    },
                    h1: ({ children }) => (
                      <h1 className="text-2xl font-bold mb-4 mt-6 first:mt-0 text-foreground">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-xl font-semibold mb-3 mt-5 first:mt-0 text-foreground">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-lg font-semibold mb-2 mt-4 first:mt-0 text-foreground">
                        {children}
                      </h3>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-2 border-border pl-4 my-3 text-muted-foreground italic">
                        {children}
                      </blockquote>
                    ),
                    // Add support for custom sections
                    div: ({ children, className }) => {
                      if (className?.includes('consideration') || className?.includes('note')) {
                        return (
                          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 my-4">
                            <div className="flex items-start gap-2">
                              <span className="text-blue-400 font-semibold text-sm">💡 Note:</span>
                              <div className="text-blue-100">{children}</div>
                            </div>
                          </div>
                        );
                      }
                      if (className?.includes('warning') || className?.includes('important')) {
                        return (
                          <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 my-4">
                            <div className="flex items-start gap-2">
                              <span className="text-yellow-400 font-semibold text-sm">⚠️ Important:</span>
                              <div className="text-yellow-100">{children}</div>
                            </div>
                          </div>
                        );
                      }
                      return <div className={className}>{children}</div>;
                    },
                    table: ({ children }) => (
                      <div className="overflow-x-auto my-4">
                        <table className="min-w-full border-collapse border border-border rounded-lg overflow-hidden">
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children }) => (
                      <th className="border border-border px-4 py-2 bg-muted font-semibold text-left text-sm">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-border px-4 py-2 text-sm">
                        {children}
                      </td>
                    ),
                    a: ({ href, children }) => (
                      <a 
                        href={href} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                      >
                        {children}
                      </a>
                    ),
                  }}
                >
                  {processMarkdownContent(message.content)}
                </ReactMarkdown>
              </div>
              )}
              
              {/* Show citations from RAG metadata - Clean References section */}
              {!message.tutorial && message.sources && message.sources.length > 0 && (
                <div className="mt-6 pt-4 border-t border-border/30">
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">References</h4>
                  <ol className="space-y-2">
                    {message.sources.map((source, idx) => (
                      <li key={idx} className="flex gap-2.5 text-sm group">
                        <span className="flex-shrink-0 text-muted-foreground font-medium text-xs">[{idx + 1}]</span>
                        <div className="flex-1 min-w-0">
                          <a 
                            href={source.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 dark:text-blue-400 hover:underline break-words inline text-sm"
                          >
                            {source.heading_path || source.title || 'Documentation'}
                          </a>
                          {source.vendor && source.vendor !== 'unknown' && (
                            <span className="text-muted-foreground ml-1.5 text-xs">
                              · {source.vendor}
                            </span>
                          )}
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {isUser && (
        <Avatar className="h-8 w-8 mt-1 flex-shrink-0 bg-gradient-to-br from-green-500 to-teal-600">
          <AvatarFallback>
            <User className="h-4 w-4 text-white" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}
