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
      <div className={cn("flex-1 flex items-center justify-center p-8", className)}>
        <div className="text-center text-muted-foreground max-w-2xl">
          <div className="mb-6">
            <div className="h-16 w-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-2xl">🤖</span>
            </div>
            <h3 className="text-2xl font-bold mb-2 text-foreground">
              Welcome to ML Documentation Copilot
            </h3>
            <p className="text-base leading-relaxed">
              Your intelligent assistant for machine learning infrastructure. 
              Get comprehensive answers with detailed citations from official documentation.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
            <div className="bg-muted/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2 text-foreground">🚀 Try asking about:</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <span>&ldquo;How to deploy models with AWS SageMaker?&rdquo;</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <span>&ldquo;Kubernetes service networking best practices&rdquo;</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <span>&ldquo;MLflow model registry and tracking&rdquo;</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <span>&ldquo;PyTorch DataLoader with MLflow integration&rdquo;</span>
                </li>
              </ul>
            </div>
            
            <div className="bg-muted/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2 text-foreground">✨ Features:</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Multi-source documentation</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Code examples & best practices</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Source citations with quality scores</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Production-grade answers</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className={cn("flex-1", className)} ref={scrollAreaRef}>
      <div className="px-4 py-2 space-y-2 min-w-0">
        {messages.map((message, index) => (
          <div
            key={message.id}
            className="animate-in slide-in-from-bottom-2 fade-in duration-300 min-w-0"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <ChatMessage message={message} />
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
