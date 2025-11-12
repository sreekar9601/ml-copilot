"use client";

import { useEffect, useRef } from "react";
import { Message } from "@/types/chat";
import { ChatMessage } from "./chat-message";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface ChatMessagesProps {
  messages: Message[];
  className?: string;
}

export function ChatMessages({ messages, className }: ChatMessagesProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className={cn("h-full flex items-center justify-center p-8", className)}>
        <div className="text-center max-w-2xl">
          <h1 className="text-4xl font-bold mb-4 text-foreground">
            ML Documentation Copilot
          </h1>
          <p className="text-lg text-muted-foreground mb-8">
            Get started by asking a question about ML frameworks, deployment, or best practices
          </p>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl mx-auto">
            <button className="p-4 text-left rounded-xl border-2 border-border hover:border-primary/50 hover:bg-accent/50 transition-all group">
              <div className="text-2xl mb-2">📝</div>
              <div className="font-medium text-sm mb-1">PyTorch Examples</div>
              <div className="text-xs text-muted-foreground">DataLoader, training loops, custom layers</div>
            </button>
            
            <button className="p-4 text-left rounded-xl border-2 border-border hover:border-primary/50 hover:bg-accent/50 transition-all group">
              <div className="text-2xl mb-2">🚀</div>
              <div className="font-medium text-sm mb-1">Deployment</div>
              <div className="text-xs text-muted-foreground">AWS SageMaker, Docker, Kubernetes</div>
            </button>
            
            <button className="p-4 text-left rounded-xl border-2 border-border hover:border-primary/50 hover:bg-accent/50 transition-all group">
              <div className="text-2xl mb-2">🔍</div>
              <div className="font-medium text-sm mb-1">Debug Issues</div>
              <div className="text-xs text-muted-foreground">Memory errors, CUDA problems</div>
            </button>
            
            <button className="p-4 text-left rounded-xl border-2 border-border hover:border-primary/50 hover:bg-accent/50 transition-all group">
              <div className="text-2xl mb-2">📊</div>
              <div className="font-medium text-sm mb-1">MLflow</div>
              <div className="text-xs text-muted-foreground">Tracking, registry, deployment</div>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className={cn("h-full w-full", className)} ref={scrollAreaRef}>
      <div className="py-8 space-y-6 max-w-4xl mx-auto px-6">
        {messages.map((message, index) => (
          <div
            key={message.id}
            className="animate-in fade-in duration-300"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <ChatMessage message={message} />
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
