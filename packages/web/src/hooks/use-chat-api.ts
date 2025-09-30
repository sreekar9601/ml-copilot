"use client";

import { useMutation } from "@tanstack/react-query";
import { ApiResponse } from "@/types/chat";

interface ChatRequest {
  q: string;
  top_k?: number;
  include_sources?: boolean;
}

// Use environment variable for API URL - no fallback for security
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_BASE_URL) {
  throw new Error("NEXT_PUBLIC_API_URL environment variable is required");
}

// Debug logging (only in development)
if (process.env.NODE_ENV === 'development') {
  console.log("🔍 API_BASE_URL:", API_BASE_URL);
  console.log("🔍 NEXT_PUBLIC_API_URL:", process.env.NEXT_PUBLIC_API_URL);
}

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
