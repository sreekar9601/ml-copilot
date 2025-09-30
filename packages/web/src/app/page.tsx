import { ChatInterface } from "@/components/chat/chat-interface";

export default function Home() {
  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <header className="border-b bg-card/80 backdrop-blur-md supports-[backdrop-filter]:bg-card/80 shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-sm">ML</span>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
                  ML Documentation Copilot
                </h1>
                <p className="text-sm text-muted-foreground">
                  AI assistant for PyTorch, MLflow, AWS SageMaker, Kubernetes, and Docker
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
                <span>Connected</span>
              </div>
              <div className="hidden sm:flex items-center gap-2 text-xs text-muted-foreground">
                <span>Powered by</span>
                <span className="font-medium">Google Gemini</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Chat Interface */}
      <main className="flex-1 container mx-auto px-4 py-6 max-w-5xl">
        <div className="h-full">
          <ChatInterface className="h-full shadow-lg rounded-xl border bg-card/50 backdrop-blur-sm" />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-card/80 backdrop-blur-md supports-[backdrop-filter]:bg-card/80">
        <div className="container mx-auto px-4 py-3">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
            <p className="text-xs text-muted-foreground">
              Built with Next.js, FastAPI, and Qdrant Cloud
            </p>
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span>Enhanced with multi-source retrieval</span>
              <span>•</span>
              <span>Production-grade quality</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}