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
      <div className={cn("flex-1 flex items-center justify-center", className)}>
        <div className="text-center text-muted-foreground max-w-md">
          <h3 className="text-lg font-semibold mb-2">
            Welcome to ML Documentation Copilot
          </h3>
          <p className="text-sm">
            Ask me anything about PyTorch, MLflow, KServe, or Ray Serve. 
            I&apos;ll provide detailed answers with citations from the official documentation.
          </p>
          <div className="mt-4 text-xs space-y-1">
            <p>Try asking about:</p>
            <ul className="list-disc list-inside space-y-1 text-left">
              <li>&ldquo;How to set up distributed training with PyTorch?&rdquo;</li>
              <li>&ldquo;What are the key components of MLflow?&rdquo;</li>
              <li>&ldquo;How to deploy models with KServe?&rdquo;</li>
              <li>&ldquo;Best practices for Ray Serve scaling?&rdquo;</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className={cn("flex-1", className)} ref={scrollAreaRef}>
      <div className="px-4 py-2">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
