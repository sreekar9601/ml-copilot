"use client";

import { useState } from "react";
import { Message } from "@/types/chat";
import { ChatMessages } from "./chat-messages";
import { ChatInput } from "./chat-input";
import { useChatApi } from "@/hooks/use-chat-api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInterfaceProps {
  className?: string;
  onBackToLanding?: () => void;
}

export function ChatInterface({ className, onBackToLanding }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const chatApi = useChatApi();

  const handleSendMessage = async (content: string) => {
    // Create user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    };

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);

    try {
      // Call API
      const response = await chatApi.mutateAsync({
        q: content,
        top_k: 5,
        include_sources: true,
      });

      // Create assistant message
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources,
      };

      // Add assistant response
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      // Create error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `I encountered an error while processing your question: ${
          error instanceof Error ? error.message : "Unknown error"
        }. Please try again or check if the backend service is running.`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <Card className={cn("flex flex-col h-full", className)}>
      <div className="p-4 border-b bg-muted/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Chat with ML Documentation Copilot</h2>
            <p className="text-sm text-muted-foreground">
              Ask questions about ML frameworks, deployment, and best practices
            </p>
          </div>
          {onBackToLanding && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onBackToLanding}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
          )}
        </div>
      </div>
      <ChatMessages messages={messages} className="flex-1" />
      <ChatInput
        onSubmit={handleSendMessage}
        isLoading={chatApi.isPending}
        className="border-0 border-t rounded-none bg-muted/20"
      />
    </Card>
  );
}
