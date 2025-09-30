"use client";

import { Message } from "@/types/chat";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bot, User, Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Citations } from "./citations";
import { cn } from "@/lib/utils";
import { processMarkdownContent } from "@/lib/markdown-utils";
import { useState } from "react";

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
        "flex gap-3 py-4 animate-in slide-in-from-bottom-4 fade-in duration-500",
        isUser ? "justify-end" : "justify-start",
        className
      )}
    >
      {!isUser && (
        <Avatar className="h-8 w-8 bg-primary">
          <AvatarFallback>
            <Bot className="h-4 w-4 text-primary-foreground" />
          </AvatarFallback>
        </Avatar>
      )}
      
      <Card
        className={cn(
          "max-w-[85%] overflow-hidden relative group break-words",
          isUser
            ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-lg"
            : "bg-card/80 backdrop-blur-sm border border-border/50"
        )}
      >
        {!isUser && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10 h-8 w-8 p-0"
          >
            {copied ? (
              <Check className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </Button>
        )}
        <CardContent className="px-4 py-3">
          {isUser ? (
            <div className="space-y-2">
              <p className="text-sm leading-relaxed break-words">{message.content}</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-semibold prose-h1:text-lg prose-h2:text-base prose-h3:text-sm prose-p:leading-relaxed prose-ul:my-2 prose-ol:my-2 prose-li:my-1 prose-strong:text-foreground prose-strong:font-semibold break-words overflow-wrap-anywhere">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => (
                      <p className="mb-4 last:mb-0 leading-relaxed text-foreground/90">
                        {children}
                      </p>
                    ),
                    ul: ({ children }) => (
                      <ul className="list-none mb-4 space-y-2 pl-0">
                        {children}
                      </ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="list-none mb-4 space-y-2 pl-0">
                        {children}
                      </ol>
                    ),
                    li: ({ children }) => (
                      <li className="flex items-start gap-3 leading-relaxed text-foreground/90">
                        <span className="flex-shrink-0 w-2 h-2 rounded-full bg-primary mt-2"></span>
                        <span className="flex-1">{children}</span>
                      </li>
                    ),
                    strong: ({ children }) => (
                      <strong className="font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded-md border border-primary/20">
                        {children}
                      </strong>
                    ),
                    code: ({ children, className }) => {
                      const isBlock = className?.includes("language-");
                      if (isBlock) {
                        return (
                          <div className="relative mb-4">
                            <div className="absolute top-2 right-2 text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded">
                              {className?.replace('language-', '') || 'code'}
                            </div>
                            <pre className="bg-gradient-to-br from-muted/90 to-muted/70 rounded-lg p-4 pt-8 overflow-x-auto border shadow-sm">
                              <code className={className}>{children}</code>
                            </pre>
                          </div>
                        );
                      }
                      return (
                        <code className="bg-gradient-to-r from-primary/10 to-primary/5 px-2 py-1 rounded text-sm font-mono border border-primary/20 text-primary">
                          {children}
                        </code>
                      );
                    },
                    h1: ({ children }) => (
                      <h1 className="text-2xl font-bold mb-5 mt-7 first:mt-0 text-foreground border-l-4 border-primary pl-4 bg-primary/5 py-3 rounded-r-lg">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-xl font-semibold mb-4 mt-6 first:mt-0 text-foreground border-l-3 border-primary/70 pl-3 bg-primary/3 py-2 rounded-r-md">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-lg font-semibold mb-3 mt-5 first:mt-0 text-primary border-l-2 border-primary/50 pl-2">
                        {children}
                      </h3>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-primary pl-4 my-4 italic text-muted-foreground bg-gradient-to-r from-primary/5 to-transparent py-2 rounded-r-lg">
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
                      <div className="overflow-x-auto my-6">
                        <table className="min-w-full border-collapse border border-border/50 rounded-lg bg-card/50">
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children }) => (
                      <th className="border border-border/50 px-4 py-3 bg-muted/50 font-semibold text-left text-foreground">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-border/50 px-4 py-3 text-foreground/90">
                        {children}
                      </td>
                    ),
                  }}
                >
                  {processMarkdownContent(message.content)}
                </ReactMarkdown>
              </div>
              
              {message.sources && message.sources.length > 0 && (
                <div className="border-t pt-4">
                  <Citations sources={message.sources} />
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {isUser && (
        <Avatar className="h-8 w-8 bg-secondary">
          <AvatarFallback>
            <User className="h-4 w-4 text-secondary-foreground" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}
