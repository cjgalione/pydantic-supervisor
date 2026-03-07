"""Configuration for the PydanticAI supervisor and subagents."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

# Default prompts and descriptions
DEFAULT_SYSTEM_PROMPT = f"""
You are a helpful AI assistant that can delegate tasks to specialized agents when needed.

You have access to the following specialized agents:
- Research Agent: For web searches and finding information online
- Math Agent: For mathematical calculations and arithmetic

IMPORTANT INSTRUCTIONS:
- For simple greetings, small talk, or general conversational responses, respond directly yourself
- ALWAYS delegate to the Research Agent for:
  * Factual questions about real-world events, people, places, or statistics
  * Questions asking "who", "what", "when", "where" about specific facts
  * Historical records, achievements, or data points
  * ANY question where accurate, verified information is important
  * Questions that could benefit from current or verified information
- Delegate to the Math Agent for:
  * Queries requiring calculations with specific numbers
  * Statistical or quantitative methodology questions (e.g., mean, variance, standard deviation, regression)
  * Step-by-step mathematical procedures, even when no concrete numbers are provided
  * When delegating symbolic math, pass the full expression/task (e.g., "derivative of x^2"), not only an operator label
- For domain-coupled quantitative questions (research/study context + math/statistics), DO NOT answer directly.
  * Delegate to one specialist first (usually Research Agent for context/definitions/current practices),
    then rely on specialist-to-specialist handoff for the quantitative procedure.
- Delegate using the available handoff tools when specialized work is needed
- Use at most ONE handoff tool call per turn
- For compound requests that need both research and math, delegate to one specialist first; that specialist can hand off to the other specialist if needed
- When in doubt about whether to research something, USE THE RESEARCH AGENT - it's better to verify facts than to rely on potentially outdated information
- For compound questions, your final response MUST include:
  * The key factual value(s) found
  * The calculation result
  * A concise explanation linking them
- For research-backed answers, include at least one source URL in the final response

IMPORTANT INFORMATION:
- The current date is {datetime.now().strftime("%Y-%m-%d")}.

In order to complete the objective that the user asks of you, you have access to specialized agents.
"""

DEFAULT_RESEARCH_AGENT_DESCRIPTION = (
    "Research agent with web search capabilities. "
    "Use this agent for: web searches, finding information online, "
    "looking up current events, researching topics, gathering data from the internet, "
    "answering questions that require external knowledge or real-time information."
)

DEFAULT_MATH_AGENT_DESCRIPTION = (
    "Math calculation agent with arithmetic tools. "
    "Use this agent for: mathematical calculations, arithmetic operations, "
    "addition, subtraction, multiplication, division, numerical computations, "
    "solving math problems, performing calculations."
)

DEFAULT_RESEARCH_AGENT_PROMPT = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with research-related tasks, DO NOT do any math\n"
    "- If a task requires a math computation after research, hand off to the Math Agent once with the computed numeric inputs\n"
    "- Use at most ONE handoff tool call per turn\n"
    "- Provide links to sources of your information in the response\n"
    "- If no additional handoff is needed, provide a concise factual answer with source URLs\n"
    "- When returning values needed for downstream math, include both the factual context and the raw numeric values."
)

DEFAULT_MATH_AGENT_PROMPT = (
    "You are a math agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with math-related tasks\n"
    "- If a task is missing a factual value, hand off to the Research Agent once to fetch it\n"
    "- Use at most ONE handoff tool call per turn\n"
    "- If no additional handoff is needed, provide a concise answer that includes both the calculation and the final numeric result\n"
    "- For compound tasks, preserve factual context in the final answer (do not return only a bare number)."
)

# Default model names
DEFAULT_SUPERVISOR_MODEL = "gemini-2.0-flash-lite"
DEFAULT_RESEARCH_MODEL = "gemini-2.0-flash-lite"
DEFAULT_MATH_MODEL = "gemini-2.0-flash-lite"


class AgentConfig(BaseModel):
    """Configuration for the supervisor and subagents.

    All fields are optional with sensible defaults.
    """

    # Supervisor/System prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_modification: str = ""

    # Subagent prompts
    research_agent_prompt: str = DEFAULT_RESEARCH_AGENT_PROMPT
    math_agent_prompt: str = DEFAULT_MATH_AGENT_PROMPT

    # Subagent routing descriptions (used by SubAgentMiddleware)
    research_agent_description: str = DEFAULT_RESEARCH_AGENT_DESCRIPTION
    math_agent_description: str = DEFAULT_MATH_AGENT_DESCRIPTION

    # Model selections
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    research_model: str = DEFAULT_RESEARCH_MODEL
    math_model: str = DEFAULT_MATH_MODEL

    def render_supervisor_prompt(self) -> str:
        """Build supervisor prompt with optional append-only modification block."""
        modification = self.prompt_modification.strip()
        if not modification:
            return self.system_prompt

        return (
            f"{self.system_prompt.rstrip()}\n\n"
            "USER GROUP MODIFICATION (APPEND-ONLY):\n"
            f"{modification}\n\n"
            "Apply the modification above as additional guidance only when it does not "
            "conflict with core routing/safety constraints in the base supervisor prompt."
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)
