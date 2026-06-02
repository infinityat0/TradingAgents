# TradingAgents/graph/setup.py

import logging
import time
from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic
from .rate_limit import make_retry_wrapper

logger = logging.getLogger(__name__)


def _make_delay_wrapper(node_fn, delay_seconds: float, state_key: str):
    """Wrap a node function to pause before executing on any turn after the first.

    Uses ``state[state_key]["count"]`` to detect subsequent turns: count == 0
    means the debate hasn't started yet, so no delay is inserted.  Every later
    turn (count > 0) sleeps for ``delay_seconds`` before invoking the node,
    giving the upstream API time to replenish its rate-limit quota.
    """
    if delay_seconds <= 0:
        return node_fn

    def wrapper(state):
        count = (state.get(state_key) or {}).get("count", 0)
        if count > 0:
            logger.info(
                "Sleeping %.0fs before researcher turn %d (rate-limit guard)...",
                delay_seconds,
                count,
            )
            time.sleep(delay_seconds)
        return node_fn(state)

    return wrapper


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
        researcher_delay_seconds: float = 0,
        node_max_retries: int = 3,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit
        self.researcher_delay_seconds = researcher_delay_seconds
        self.node_max_retries = node_max_retries

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(
            selected_analysts,
            concurrency_limit=self.analyst_concurrency_limit,
        )

        delay = self.researcher_delay_seconds
        retries = self.node_max_retries

        def _retry(fn):
            return make_retry_wrapper(fn, retries)

        def _delay_retry(fn, state_key):
            # Composition: proactive turn delay fires once, reactive retry
            # delay fires only between attempts on 429.
            return _make_delay_wrapper(_retry(fn), delay, state_key)

        analyst_factories = {
            "market": lambda: _retry(create_market_analyst(self.quick_thinking_llm)),
            "social": lambda: _retry(create_sentiment_analyst(self.quick_thinking_llm)),
            "news": lambda: _retry(create_news_analyst(self.quick_thinking_llm)),
            "fundamentals": lambda: _retry(create_fundamentals_analyst(self.quick_thinking_llm)),
        }

        # Researcher nodes: proactive turn delay + retry on 429.
        # Bull starts first (count == 0) so it gets no proactive delay;
        # every subsequent turn sleeps to avoid hitting per-minute quotas.
        bull_researcher_node = _delay_retry(
            create_bull_researcher(self.quick_thinking_llm),
            "investment_debate_state",
        )
        bear_researcher_node = _delay_retry(
            create_bear_researcher(self.quick_thinking_llm),
            "investment_debate_state",
        )
        research_manager_node = _retry(create_research_manager(self.deep_thinking_llm))
        trader_node = _retry(create_trader(self.quick_thinking_llm))

        # Risk analysis nodes: same delay + retry pattern.
        # Aggressive starts first (count == 0, no delay).
        aggressive_analyst = _delay_retry(
            create_aggressive_debator(self.quick_thinking_llm),
            "risk_debate_state",
        )
        neutral_analyst = _delay_retry(
            create_neutral_debator(self.quick_thinking_llm),
            "risk_debate_state",
        )
        conservative_analyst = _delay_retry(
            create_conservative_debator(self.quick_thinking_llm),
            "risk_debate_state",
        )
        portfolio_manager_node = _retry(create_portfolio_manager(self.deep_thinking_llm))

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
