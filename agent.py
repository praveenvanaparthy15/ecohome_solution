import logging
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from tools import ECOHOME_TOOLS

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTIONS = """
You are EcoHome Energy Advisor, an AI energy optimization assistant.

Your job is to provide practical, data-driven recommendations for reducing
household electricity consumption and cost.

When answering a question:
1. Understand the user's energy concern.
2. Use the available tools whenever current, historical, pricing, weather,
   solar, savings, or knowledge-base information is relevant.
3. Use RAG/search_energy_tips for energy-efficiency best practices.
4. Use historical energy data when the user asks about usage patterns,
   devices, trends, or optimization.
5. Use electricity pricing when peak/off-peak timing affects the recommendation.
6. Use weather and solar data when solar generation or weather affects the advice.
7. Use calculate_savings when a numerical savings estimate is possible.
8. Combine multiple tool results when needed.
9. Give a clear recommendation and explain why it is recommended.
10. Do not invent measurements, prices, weather conditions, or savings.
11. Clearly identify mock/simulated data when applicable.
12. If the available data is insufficient, say so and provide a reasonable
    general recommendation.

Keep the final answer concise but useful.
"""


class AgentState(TypedDict, total=False):
    question: str
    context: str
    messages: list[Any]
    final_response: str
    used_tools: list[str]
    errors: list[str]


class EcoHomeAgent:
    """
    LangGraph-powered EcoHome Energy Advisor.

    The agent accepts:
        instructions: system-level agent instructions
        model: Cohere model name

    invoke() accepts:
        question: user energy question
        context: optional additional context
    """

    def __init__(
        self,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model: str = "command-a-03-2025",
    ):
        self.instructions = instructions
        self.model_name = model

        self.llm = ChatCohere(
            model=model,
            temperature=0,
        )

        self.tool_llm = self.llm.bind_tools(ECOHOME_TOOLS)

        self.tool_node = ToolNode(ECOHOME_TOOLS)

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("plan_and_use_tools", self._plan_and_use_tools)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("plan_and_use_tools")

        workflow.add_conditional_edges(
            "plan_and_use_tools",
            self._should_use_tools,
            {
                "tools": "execute_tools",
                "final": "finalize",
            },
        )

        workflow.add_edge("execute_tools", "plan_and_use_tools")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _plan_and_use_tools(self, state: AgentState) -> dict:
        try:
            messages = state.get("messages", [])

            response = self.tool_llm.invoke(messages)

            return {
                "messages": messages + [response],
                "errors": state.get("errors", []),
            }

        except Exception as exc:
            logger.exception("Agent planning failed")

            errors = state.get("errors", [])
            errors.append(str(exc))

            return {
                "messages": state.get("messages", []),
                "errors": errors,
            }

    def _should_use_tools(self, state: AgentState) -> str:
        messages = state.get("messages", [])

        if not messages:
            return "final"

        last_message = messages[-1]

        tool_calls = getattr(last_message, "tool_calls", None)

        if tool_calls:
            return "tools"

        return "final"

    def _execute_tools(self, state: AgentState) -> dict:
        try:
            messages = state.get("messages", [])

            result = self.tool_node.invoke(
                {
                    "messages": messages
                }
            )

            tool_messages = result.get("messages", [])

            used_tools = list(state.get("used_tools", []))

            for message in tool_messages:
                tool_name = getattr(message, "name", None)

                if tool_name and tool_name not in used_tools:
                    used_tools.append(tool_name)

            logger.info(
                "Tools executed: %s",
                used_tools,
            )

            return {
                "messages": messages + tool_messages,
                "used_tools": used_tools,
                "errors": state.get("errors", []),
            }

        except Exception as exc:
            logger.exception("Tool execution failed")

            errors = state.get("errors", [])
            errors.append(str(exc))

            return {
                "messages": messages,
                "used_tools": state.get("used_tools", []),
                "errors": errors,
            }

    def _finalize(self, state: AgentState) -> dict:
        try:
            question = state.get("question", "")
            context = state.get("context", "")
            messages = state.get("messages", [])

            tool_summary = []

            for message in messages:
                message_type = message.__class__.__name__

                if message_type == "ToolMessage":
                    tool_name = getattr(message, "name", "unknown_tool")
                    content = getattr(message, "content", "")

                    tool_summary.append(
                        f"Tool: {tool_name}\nResult: {content}"
                    )

            summary_text = "\n\n".join(tool_summary)

            final_prompt = f"""
You are the final recommendation stage of EcoHome Energy Advisor.

User question:
{question}

Additional context:
{context or "No additional context provided."}

Tools used:
{", ".join(state.get("used_tools", [])) or "None"}

Tool results:
{summary_text or "No tool results available."}

Create the final answer.

Requirements:
- Directly answer the user's question.
- Base recommendations on the available tool results.
- Mention important numerical findings.
- If savings were calculated, explain them clearly.
- Give practical actions the homeowner can take.
- Do not invent missing information.
- If data is mock/simulated, make that clear.
- Keep the response concise and easy to understand.
"""

            response = self.llm.invoke(
                [
                    SystemMessage(content=self.instructions),
                    HumanMessage(content=final_prompt),
                ]
            )

            final_response = response.content

            return {
                "final_response": final_response,
                "messages": messages + [response],
                "used_tools": state.get("used_tools", []),
                "errors": state.get("errors", []),
            }

        except Exception as exc:
            logger.exception("Final response generation failed")

            errors = state.get("errors", [])
            errors.append(str(exc))

            fallback = (
                "I could not generate the final recommendation because "
                "the AI response stage encountered an error."
            )

            return {
                "final_response": fallback,
                "messages": state.get("messages", []),
                "used_tools": state.get("used_tools", []),
                "errors": errors,
            }

    def invoke(
        self,
        question: str,
        context: str = "",
    ) -> dict:
        """
        Run the EcoHome agent.

        Parameters:
            question: User's energy-related question.
            context: Optional additional user/property context.

        Returns:
            Dictionary containing final response, used tools and errors.
        """

        if not question or not question.strip():
            return {
                "success": False,
                "response": "Please provide an energy-related question.",
                "used_tools": [],
                "errors": ["Empty question"],
            }

        user_content = f"""
User question:
{question}

Additional context:
{context or "None provided."}

Use the available tools when they can improve the answer.
"""

        initial_messages = [
            SystemMessage(content=self.instructions),
            HumanMessage(content=user_content),
        ]

        try:
            result = self.graph.invoke(
                {
                    "question": question,
                    "context": context,
                    "messages": initial_messages,
                    "used_tools": [],
                    "errors": [],
                }
            )

            return {
                "success": not bool(result.get("errors")),
                "response": result.get(
                    "final_response",
                    "No recommendation was generated.",
                ),
                "used_tools": result.get("used_tools", []),
                "errors": result.get("errors", []),
            }

        except Exception as exc:
            logger.exception("EcoHome agent invocation failed")

            return {
                "success": False,
                "response": (
                    "The EcoHome agent could not complete the request."
                ),
                "used_tools": [],
                "errors": [str(exc)],
            }


def create_agent(
    instructions: str = DEFAULT_INSTRUCTIONS,
    model: str = "command-a-03-2025",
) -> EcoHomeAgent:
    """Convenience factory for creating the EcoHome agent."""
    return EcoHomeAgent(
        instructions=instructions,
        model=model,
    )


if __name__ == "__main__":
    agent = create_agent()

    result = agent.invoke(
        "How can I reduce my household electricity cost during peak hours?"
    )

    print("\nSuccess:", result["success"])
    print("Used tools:", result["used_tools"])
    print("\nResponse:\n")
    print(result["response"])

    if result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print("-", error)