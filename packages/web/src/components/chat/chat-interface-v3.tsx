"use client";

import { useState, useEffect } from "react";
import { Message } from "@/types/chat";
import { ChatMessages } from "./chat-messages";
import { ChatInput } from "./chat-input";
import { AgentStatus } from "./agent-status";
import { useAgentStream } from "@/hooks/use-agent-stream";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { v4 as uuidv4 } from 'uuid';

interface ChatInterfaceV3Props {
  className?: string;
  onBackToLanding?: () => void;
}

export function ChatInterfaceV3({ className, onBackToLanding }: ChatInterfaceV3Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId] = useState(() => uuidv4());
  
  const {
    streamQuery,
    cancelStream,
    isStreaming,
    status,
    response,
    metadata,
    error,
  } = useAgentStream({
    onComplete: (fullResponse, meta) => {
      // When streaming completes, update the assistant message with final metadata
      console.log('onComplete called with sources:', meta.sources);
      
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.role === 'assistant') {
          const updatedMessage = {
            ...lastMessage,
            content: fullResponse,
            sources: meta.sources || [],  // Add sources for citations
            metadata: {
              cost: meta.cost,
              iterations: meta.iterations,
              tools_used: meta.tools_used,
              frameworks_detected: meta.frameworks_detected,
            }
          };
          
          console.log('Updated message with sources:', updatedMessage.sources?.length || 0);
          
          return [
            ...prev.slice(0, -1),
            updatedMessage
          ];
        }
        return prev;
      });
    },
    onError: (errorMsg) => {
      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ **Error**: ${errorMsg}\n\nPlease try again or start a new conversation.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  });

  const handleSendMessage = async (content: string) => {
    // Create user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    };

    // Create placeholder assistant message
    const assistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
      timestamp: new Date(),
    };

    // Add both messages
    setMessages(prev => [...prev, userMessage, assistantMessage]);

    // Start streaming
    await streamQuery(content, conversationId);
  };

  // Update the last assistant message as tokens arrive
  useEffect(() => {
    if (response && isStreaming) {
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.role === 'assistant') {
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              content: response
            }
          ];
        }
        return prev;
      });
    }
  }, [response, isStreaming]);

  return (
    <div className={cn("flex flex-col h-screen w-full bg-background", className)}>
      {/* Minimal Header */}
      <div className="flex-shrink-0 px-6 py-3 border-b">
        <div className="flex items-center justify-between max-w-5xl mx-auto">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">ML Documentation Copilot</h1>
            {metadata && (
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>💰 ${metadata.cost.toFixed(4)}</span>
                <span>🔄 {metadata.iterations}</span>
              </div>
            )}
          </div>
          {onBackToLanding && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onBackToLanding}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Messages - Full screen */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <ChatMessages messages={messages} className="h-full" />
      </div>

      {/* Agent Status - Minimal */}
      {isStreaming && (
        <div className="flex-shrink-0 border-t border-b bg-muted/30">
          <AgentStatus
            status={status}
            onCancel={cancelStream}
          />
        </div>
      )}

      {/* Error - Minimal */}
      {error && !isStreaming && (
        <div className="flex-shrink-0 px-6 py-2 bg-destructive/10 text-destructive text-sm border-t max-w-5xl mx-auto">
          {error}
        </div>
      )}

      {/* Input - Clean bottom */}
      <div className="flex-shrink-0 bg-background">
        <ChatInput
          onSubmit={handleSendMessage}
          isLoading={isStreaming}
          placeholder={
            isStreaming
              ? "Agent is working..."
              : "Ask about ML frameworks, deployment, best practices..."
          }
        />
      </div>
    </div>
  );
}

