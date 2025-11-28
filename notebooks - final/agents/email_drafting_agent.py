"""
Email Drafting Agent for generating contextual sales emails.
"""

import google.generativeai as genai
from typing import Dict, List, Optional
from datetime import datetime


class EmailDraftingAgent:
    """
    Email Drafting Agent that generates contextual sales emails using advanced prompt engineering.
    Supports follow-up, introduction, and proposal email types with personalization.
    """
    
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Email Drafting Agent.
        
        Args:
            model_name (str): The Gemini model to use for generation
        """
        self.model = genai.GenerativeModel(model_name)
        
        # Supported email types
        self.email_types = {
            "follow_up": "Follow-up Email",
            "introduction": "Introduction Email", 
            "proposal": "Proposal Email",
            "check_in": "Check-in Email",
            "closing": "Deal Closing Email"
        }
    
    def draft_email(
        self,
        email_type: str,
        account_data: Dict,
        opportunity_data: Optional[Dict] = None,
        engagement_history: Optional[List[str]] = None,
        custom_context: Optional[str] = None,
        tone: str = "professional"
    ) -> str:
        """
        Generate a contextual sales email based on provided data.
        
        Args:
            email_type (str): Type of email ('follow_up', 'introduction', 'proposal', 'check_in', 'closing')
            account_data (dict): Account information (sector, revenue, employees, location, etc.)
            opportunity_data (dict): Opportunity details (product, deal_stage, close_value, dates, etc.)
            engagement_history (list): Recent engagement activities/interactions
            custom_context (str): Additional context or special instructions
            tone (str): Email tone ('professional', 'friendly', 'urgent', 'consultative')
            
        Returns:
            str: Generated email content
        """
        
        # Validate email type
        if email_type not in self.email_types:
            raise ValueError(
                f"Invalid email type: '{email_type}'. "
                f"Choose from: {list(self.email_types.keys())}"
            )
        
        # Build the prompt
        prompt = self._build_email_prompt(
            email_type,
            account_data,
            opportunity_data,
            engagement_history,
            custom_context,
            tone
        )
        
        # Generate the email
        response = self.model.generate_content(prompt)
        return response.text
    
    def _build_email_prompt(
        self,
        email_type: str,
        account_data: Dict,
        opportunity_data: Optional[Dict],
        engagement_history: Optional[List[str]],
        custom_context: Optional[str],
        tone: str
    ) -> str:
        """Constructs the AI prompt with context."""
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
You are an expert sales communication specialist.

Generate a {email_type} email with the following context:

**ACCOUNT INFORMATION:**
{str(account_data)}

**OPPORTUNITY DETAILS:**
{str(opportunity_data) if opportunity_data else 'No opportunity data provided'}

**RECENT ENGAGEMENT:**
{self._format_engagement_history(engagement_history) if engagement_history else 'No recent engagement history'}

**TONE:** {tone}

{self._get_email_type_guidance(email_type, opportunity_data)}

{self._get_tone_instructions(tone)}

**CUSTOM CONTEXT:**
{custom_context if custom_context else 'None'}

**OUTPUT FORMAT:**

Subject: [Your compelling subject line]

---

Dear [Appropriate greeting],

[Email body with 3-4 concise paragraphs]

[Clear call-to-action]

[Professional closing]

[Signature block]

---

**IMPORTANT:** 
- Do NOT use placeholder brackets like [Company Name] - use actual data provided
- Be specific and data-driven, not generic
- Make it sound natural and human, not robotic
- Current date is {current_date} - use for time-sensitive context

Generate the email now:
"""
        return prompt
    
    def _get_email_type_guidance(self, email_type: str, opportunity_data: Optional[Dict]) -> str:
        """Provide specific guidance based on email type and opportunity stage."""
        
        guidance = {
            "follow_up": """
This is a FOLLOW-UP email after a previous interaction.
Reference the last conversation and provide value-add information.
""",
            "introduction": """
This is an INTRODUCTION email to a new prospect.
Focus on capturing attention and establishing credibility.
""",
            "proposal": """
This is a PROPOSAL email presenting a solution.
Clearly articulate value proposition and ROI.
""",
            "check_in": """
This is a CHECK-IN email to maintain engagement.
Be helpful without being pushy, provide relevant insights.
""",
            "closing": """
This is a CLOSING email to finalize a deal.
Address any remaining concerns and create urgency.
"""
        }
        
        return guidance.get(email_type, "")
    
    def _get_tone_instructions(self, tone: str) -> str:
        """Provide tone-specific instructions."""
        
        tones = {
            "professional": "Maintain a professional, business-appropriate tone.",
            "friendly": "Use a warm, approachable tone while remaining professional.",
            "urgent": "Create a sense of urgency without being pushy.",
            "consultative": "Position yourself as a trusted advisor and thought partner."
        }
        
        return tones.get(tone, "Maintain a professional tone.")
    
    def _format_engagement_history(self, history: List[str]) -> str:
        """Format engagement history into a readable bullet list."""
        
        formatted = []
        for item in history:
            formatted.append(f"- {item}")
        return "\n".join(formatted)
    
    def generate_bulk_emails(
        self,
        email_configs: List[Dict]
    ) -> List[Dict[str, str]]:
        """
        Generate multiple emails in batch.
        
        Args:
            email_configs (list): List of email configuration dictionaries
            
        Returns:
            list: List of dictionaries with 'config', 'email', and 'status' keys
        """
        results = []
        
        for i, config in enumerate(email_configs, 1):
            try:
                email = self.draft_email(
                    email_type=config.get('email_type', 'follow_up'),
                    account_data=config.get('account_data', {}),
                    opportunity_data=config.get('opportunity_data'),
                    engagement_history=config.get('engagement_history'),
                    custom_context=config.get('custom_context'),
                    tone=config.get('tone', 'professional')
                )
                
                results.append({
                    'config': config,
                    'email': email,
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'config': config,
                    'email': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results