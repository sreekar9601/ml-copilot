"""Track and manage LLM API costs."""

from ..config import config


class CostTracker:
    """Calculate costs for Gemini API calls."""
    
    @staticmethod
    def calculate_cost(input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD for a single API call.
        
        Gemini Flash pricing (as of 2025):
        - Input: $0.075 per 1M tokens
        - Output: $0.30 per 1M tokens
        """
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output_tokens
        return input_cost + output_cost
    
    @staticmethod
    def check_budget(conversation_id: str, current_cost: float) -> bool:
        """
        Check if conversation is within budget limits.
        
        Raises:
            BudgetExceededError: If max cost exceeded
        """
        if current_cost > config.max_cost_per_session:
            raise BudgetExceededError(
                f"Session cost ${current_cost:.4f} exceeds limit "
                f"${config.max_cost_per_session}"
            )
        return True
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count from text.
        
        Uses simple heuristic: ~1.3 words per token for English.
        For production, use actual tokenizer.
        """
        words = len(text.split())
        return int(words * 1.3)


class BudgetExceededError(Exception):
    """Raised when session budget is exceeded."""
    pass

