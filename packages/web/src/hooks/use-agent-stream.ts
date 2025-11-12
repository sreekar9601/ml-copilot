/**
 * React hook for streaming agent responses using SSE.
 * 
 * Handles Server-Sent Events from the agent API and provides real-time updates for:
 * - Agent thoughts (thinking state)
 * - Tool executions (with tool names and status)
 * - Token-by-token responses (streaming text)
 * - Completion metadata (cost, iterations, tools used)
 */

import { useState, useCallback, useRef } from 'react';

interface AgentStatus {
  type: 'idle' | 'thinking' | 'tool_execution' | 'responding' | 'error';
  message?: string;
  tool?: string;
}

interface AgentMetadata {
  iterations: number;
  cost: number;
  tools_used: string[];
  frameworks_detected: string[];
  sources?: Array<{
    chunk_id: string;
    title: string;
    url: string;
    heading_path: string;
    anchor_link: string;
    relevance_score: number;
    vendor?: string;
  }>;
}

interface UseAgentStreamOptions {
  apiUrl?: string;
  onToken?: (token: string) => void;
  onComplete?: (response: string, metadata: AgentMetadata) => void;
  onError?: (error: string) => void;
}

export function useAgentStream(options: UseAgentStreamOptions = {}) {
  const {
    apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    onToken,
    onComplete,
    onError
  } = options;

  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<AgentStatus>({ type: 'idle' });
  const [response, setResponse] = useState('');
  const [metadata, setMetadata] = useState<AgentMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const streamQuery = useCallback(async (
    query: string,
    conversationId?: string
  ) => {
    // Reset state
    setIsStreaming(true);
    setResponse('');
    setError(null);
    setMetadata(null);
    setStatus({ type: 'thinking', message: 'Connecting to agent...' });
    
    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController();
    
    try {
      const response = await fetch(`${apiUrl}/agent/invoke/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
        }),
        signal: abortControllerRef.current.signal,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error('No response body');
      }
      
      let buffer = '';
      let currentEvent = '';
      let responseText = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (!line.trim()) continue;
          
          // Parse SSE format
          if (line.startsWith('event:')) {
            currentEvent = line.slice(6).trim();
          }
          
          else if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            
            try {
              const data = JSON.parse(dataStr);
              
              // Handle different event types
              if (currentEvent === 'thought') {
                setStatus({
                  type: 'thinking',
                  message: data.content
                });
              }
              
              else if (currentEvent === 'tool_call') {
                // Map tool names to user-friendly display names
                const toolDisplayNames: Record<string, string> = {
                  'web_search': 'Searching web',
                  'tavily_search_results_json': 'Searching web',
                  'tavily_search_results': 'Searching web',
                  'tavily': 'Searching web',
                  'hybrid_doc_search': 'Searching documentation',
                  'execute_python_code': 'Executing code',
                  'validate_code_syntax': 'Validating code',
                  'get_specific_documentation': 'Looking up documentation'
                };
                
                // Normalize tool name first
                const normalizedTool = data.tool === 'tavily_search_results_json' || 
                                      data.tool === 'tavily_search_results' || 
                                      data.tool === 'tavily' 
                                      ? 'web_search' 
                                      : data.tool;
                
                const displayName = data.display || toolDisplayNames[data.tool] || toolDisplayNames[normalizedTool] || `Using ${normalizedTool}...`;
                
                setStatus({
                  type: 'tool_execution',
                  message: displayName,
                  tool: normalizedTool
                });
              }
              
              else if (currentEvent === 'tool_result') {
                setStatus({
                  type: 'responding',
                  message: data.preview || 'Tool completed'
                });
              }
              
              else if (currentEvent === 'token') {
                const token = data.content;
                responseText += token;
                setResponse(prev => prev + token);
                setStatus({ type: 'responding' });
                
                // Call token callback
                onToken?.(token);
              }
              
              else if (currentEvent === 'done') {
                const meta: AgentMetadata = {
                  iterations: data.metadata.iterations,
                  cost: data.metadata.cost,
                  tools_used: data.metadata.tools_used || [],
                  frameworks_detected: data.metadata.frameworks_detected || [],
                  sources: data.metadata.sources || []
                };
                
                console.log('Agent completed with metadata:', meta);
                console.log('Sources received:', meta.sources?.length || 0);
                
                setMetadata(meta);
                setStatus({ type: 'idle' });
                setIsStreaming(false);
                
                // Call completion callback
                onComplete?.(responseText, meta);
              }
              
              else if (currentEvent === 'error') {
                const errorMsg = data.error || 'Unknown error';
                setError(errorMsg);
                setStatus({ type: 'error', message: errorMsg });
                setIsStreaming(false);
                
                // Call error callback
                onError?.(errorMsg);
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data:', dataStr, parseError);
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Request cancelled');
          setStatus({ type: 'idle' });
        } else {
          const errorMsg = err.message;
          setError(errorMsg);
          setStatus({ type: 'error', message: errorMsg });
          onError?.(errorMsg);
        }
      }
      setIsStreaming(false);
    }
  }, [apiUrl, onToken, onComplete, onError]);
  
  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setStatus({ type: 'idle' });
  }, []);
  
  return {
    streamQuery,
    cancelStream,
    isStreaming,
    status,
    response,
    metadata,
    error,
  };
}

