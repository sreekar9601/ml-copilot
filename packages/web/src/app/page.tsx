import { ChatInterface } from "@/components/chat/chat-interface";

export default function Home() {
  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold">ML Documentation Copilot</h1>
              <p className="text-sm text-muted-foreground">
                AI assistant for PyTorch, MLflow, KServe, and Ray Serve
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="h-2 w-2 rounded-full bg-green-500"></div>
              Connected
            </div>
          </div>
        </div>
      </header>

      {/* Main Chat Interface */}
      <main className="flex-1 container mx-auto px-4 py-4 max-w-4xl">
        <ChatInterface className="h-full" />
      </main>

      {/* Footer */}
      <footer className="border-t bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="container mx-auto px-4 py-2">
          <p className="text-xs text-center text-muted-foreground">
            Powered by Google Gemini • Built with Next.js and FastAPI
          </p>
        </div>
      </footer>
    </div>
  );
}