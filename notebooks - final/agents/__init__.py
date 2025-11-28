"""
CRM AI Assistant Agents

This package contains specialized AI agents for CRM operations:
- EmailDraftingAgent: Generate contextual sales emails
- AccountSummaryAgent: Create executive account summaries
- InsightGenerationAgent: Produce business insights and analysis
"""

from .email_drafting_agent import EmailDraftingAgent
from .account_summary_agent import AccountSummaryAgent
from .insight_generation_agent import InsightGenerationAgent

__all__ = [
    'EmailDraftingAgent',
    'AccountSummaryAgent',
    'InsightGenerationAgent'
]

__version__ = '1.0.0'