"use client";

import { Message } from "@/types/chat";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Citations } from "./citations";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: Message;
  className?: string;
}

export function ChatMessage({ message, className }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 py-4",
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
          "max-w-[85%] overflow-hidden",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted/50"
        )}
      >
        <CardContent className="px-4 py-3">
          {isUser ? (
            <p className="text-sm">{message.content}</p>
          ) : (
            <div className="space-y-4">
              <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-semibold prose-h1:text-lg prose-h2:text-base prose-h3:text-sm prose-p:leading-relaxed prose-ul:my-2 prose-ol:my-2 prose-li:my-1">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
                    ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                    li: ({ children }) => <li className="mb-1 leading-relaxed">{children}</li>,
                    code: ({ children, className }) => {
                      const isBlock = className?.includes("language-");
                      if (isBlock) {
                        return (
                          <pre className="bg-muted/80 rounded-lg p-4 overflow-x-auto mb-4 border">
                            <code className={className}>{children}</code>
                          </pre>
                        );
                      }
                      return (
                        <code className="bg-muted/80 px-2 py-1 rounded text-sm font-mono">
                          {children}
                        </code>
                      );
                    },
                    h1: ({ children }) => <h1 className="text-lg font-bold mb-3 mt-4 first:mt-0 text-foreground">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-base font-semibold mb-2 mt-3 first:mt-0 text-foreground">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-sm font-semibold mb-2 mt-2 first:mt-0 text-foreground">{children}</h3>,
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-primary pl-4 my-4 italic text-muted-foreground">
                        {children}
                      </blockquote>
                    ),
                    table: ({ children }) => (
                      <div className="overflow-x-auto my-4">
                        <table className="min-w-full border-collapse border border-border rounded-lg">
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children }) => (
                      <th className="border border-border px-3 py-2 bg-muted font-semibold text-left">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-border px-3 py-2">
                        {children}
                      </td>
                    ),
                  }}
                >
                  {message.content}
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
