"use client";

import { useMutation } from "@tanstack/react-query";
import { ApiResponse } from "@/types/chat";

interface ChatRequest {
  q: string;
  top_k?: number;
  include_sources?: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function askQuestion(request: ChatRequest): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(
      errorData?.detail || `HTTP ${response.status}: ${response.statusText}`
    );
  }

  return response.json();
}

export function useChatApi() {
  return useMutation({
    mutationFn: askQuestion,
    onError: (error) => {
      console.error("Chat API error:", error);
    },
  });
}
