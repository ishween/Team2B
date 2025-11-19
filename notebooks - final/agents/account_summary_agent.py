"""
Account Summary Agent for generating executive-level account summaries.
"""

import google.generativeai as genai
import json
from typing import Dict, Optional


class AccountSummaryAgent:
    """
    Account Summary Agent that creates dynamic account summaries using structured prompts.
    Analyzes account data, opportunities, and activities to generate comprehensive executive 
    overviews with risk assessments and recommendations.
    """
    
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Account Summary Agent.
        
        Args:
            model_name (str): The Gemini model to use for generation
        """
        self.model = genai.GenerativeModel(model_name)
    
    def create_summary(self, account_data: dict, health_score: float = None) -> str:
        """
        Generate an executive-level account summary from account data.
        
        Args:
            account_data (dict): Account information including:
                - account_name: Name of the account
                - industry/sector: Industry vertical
                - annual_revenue: Annual revenue
                - recent_activity: List of recent activities
                - open_opportunities: List of open opportunities
                - risk_factors: List of identified risks
                - Any other relevant account details
            health_score (float): Optional health score (0-1 scale)
            
        Returns:
            str: Generated account summary with analysis and recommendations
        """
        
        prompt = f"""
You are an expert CRM account analyst.

You will be given structured account data and a predictive health score.
Return a concise, executive-level account summary.

Account Health Score: {health_score:.2f if health_score else 'Not available'}

Account Data (JSON):
{json.dumps(account_data, indent=2)}

Generate the following sections:

1. **Executive Summary** (3-4 sentences)
2. **Key Strengths** (bullet points)
3. **Key Risks or Warning Signs** (bullet points)
4. **Open Opportunities** (bullet points if any)
5. **Recommended Next Actions** (3 concrete steps with owners + timelines)
6. **Overall Priority Level** (High / Medium / Low)

Be specific, data-driven, and professional.
"""
        
        response = self.model.generate_content(prompt)
        return response.text