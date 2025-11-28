"""
Insight Generation Agent for producing business insights and pipeline analysis.
"""

import google.generativeai as genai
import pandas as pd
import json
from typing import Dict, Optional, List


class InsightGenerationAgent:
    """
    Insight Generation Agent that produces intelligent business insights and pipeline analysis.
    Uses sophisticated prompts to evaluate conversion rates, deal performance, and strategic 
    recommendations for sales optimization.
    """
    
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Insight Generation Agent.
        
        Args:
            model_name (str): The Gemini model to use for generation
        """
        self.model = genai.GenerativeModel(model_name)
        
        # Supported insight types
        self.insight_types = {
            "pipeline_analysis": "Pipeline Health & Stage Analysis",
            "conversion_analysis": "Conversion Rate & Funnel Analysis",
            "performance_analysis": "Sales Performance & Deal Metrics",
            "forecasting": "Revenue Forecasting & Predictions",
            "recommendations": "Strategic Recommendations",
            "agent_performance": "Sales Agent Performance Analysis",
            "product_analysis": "Product Performance Analysis"
        }
    
    def generate_insight(
        self,
        insight_type: str,
        pipeline_data: pd.DataFrame,
        accounts_data: Optional[pd.DataFrame] = None,
        teams_data: Optional[pd.DataFrame] = None,
        products_data: Optional[pd.DataFrame] = None,
        time_period: Optional[str] = None,
        filters: Optional[Dict] = None,
        focus_area: Optional[str] = None
    ) -> str:
        """
        Generate a specific type of business insight from CRM data.
        
        This is the main method that orchestrates the insight generation process:
        1. Validates the insight type
        2. Calculates metrics from the data
        3. Builds a sophisticated prompt
        4. Generates the insight using AI
        
        Args:
            insight_type (str): Type of insight ('pipeline_analysis', 'conversion_analysis', etc.)
            pipeline_data (pd.DataFrame): Main opportunities/pipeline data
            accounts_data (pd.DataFrame, optional): Account information
            teams_data (pd.DataFrame, optional): Sales team information
            products_data (pd.DataFrame, optional): Product catalog
            time_period (str, optional): Time frame for analysis (e.g., 'Q4 2024')
            filters (dict, optional): Filters to apply (e.g., {'product': 'GTXPro'})
            focus_area (str, optional): Specific area to focus on
            
        Returns:
            str: Generated insight report with analysis and recommendations
            
        Raises:
            ValueError: If insight_type is not supported
        """
        
        # Validate that the requested insight type is supported
        if insight_type not in self.insight_types:
            raise ValueError(
                f"Invalid insight type: '{insight_type}'. "
                f"Choose from: {list(self.insight_types.keys())}"
            )
        
        # Calculate all relevant metrics from the data
        metrics = self._calculate_metrics(
            pipeline_data, 
            accounts_data, 
            teams_data, 
            products_data,
            filters
        )
        
        # Creates type-specific instructions and context for the AI prompt
        prompt = self._build_insight_prompt(
            insight_type,
            metrics,
            time_period,
            focus_area
        )
        
        # Generate the insight using Gemini
        response = self.model.generate_content(prompt)
        
        return response.text
    
    def _calculate_metrics(
        self,
        pipeline_data: pd.DataFrame,
        accounts_data: Optional[pd.DataFrame],
        teams_data: Optional[pd.DataFrame],
        products_data: Optional[pd.DataFrame],
        filters: Optional[Dict]
    ) -> Dict:
        """
        Calculate comprehensive business metrics from CRM data.
        
        This method extracts and computes all the numbers needed for insights:
        - Summary metrics (total opps, conversion rates, revenue)
        - Stage distribution (how many deals at each stage)
        - Performance breakdowns (by agent, product, etc.)
        - Time-based trends if date columns available
        
        Args:
            pipeline_data: Main pipeline DataFrame
            accounts_data: Accounts DataFrame
            teams_data: Teams DataFrame
            products_data: Products DataFrame
            filters: Optional filters to apply
            
        Returns:
            dict: Comprehensive metrics dictionary
        """
        
        # Apply filters if provided
        df = pipeline_data.copy()
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        # Calculate basic summary metrics
        total_opps = len(df)
        won_deals = df[df['deal_stage'].str.lower() == 'won']
        lost_deals = df[df['deal_stage'].str.lower() == 'lost']
        engaging_deals = df[df['deal_stage'].str.lower() == 'engaging']
        
        metrics = {
            "summary": {
                "total_opportunities": total_opps,
                "won_deals": len(won_deals),
                "lost_deals": len(lost_deals),
                "active_deals": len(engaging_deals),
                "win_rate": (len(won_deals) / total_opps * 100) if total_opps > 0 else 0,
                "total_revenue": float(won_deals['close_value'].sum()),
                "pipeline_value": float(engaging_deals['close_value'].sum()),
                "avg_deal_size": float(df['close_value'].mean()),
                "median_deal_size": float(df['close_value'].median())
            },
            
            "stage_distribution": df['deal_stage'].value_counts().to_dict(),
            
            "product_breakdown": df.groupby('product').agg({
                'close_value': ['sum', 'count', 'mean']
            }).to_dict() if 'product' in df.columns else {},
            
            "agent_breakdown": df.groupby('sales_agent').agg({
                'close_value': ['sum', 'count']
            }).to_dict() if 'sales_agent' in df.columns else {}
        }
        
        # Add account-level metrics if accounts data provided
        if accounts_data is not None:
            metrics["account_metrics"] = {
                "total_accounts": len(accounts_data),
                "account_sectors": accounts_data['sector'].value_counts().to_dict() if 'sector' in accounts_data.columns else {}
            }
        
        return metrics
    
    def _build_insight_prompt(
        self,
        insight_type: str,
        metrics: Dict,
        time_period: Optional[str],
        focus_area: Optional[str]
    ) -> str:
        """
        Build the comprehensive prompt for insight generation.
        
        Constructs a detailed prompt with:
        - Role definition for the AI
        - All calculated metrics
        - Type-specific instructions
        - Output format requirements
        
        Args:
            insight_type: Type of insight to generate
            metrics: Calculated metrics dictionary
            time_period: Optional time period context
            focus_area: Optional specific focus area
            
        Returns:
            str: Complete prompt for AI generation
        """
        
        prompt = f"""
You are an expert business analyst specializing in CRM and sales analytics.

Generate a **{self.insight_types[insight_type]}** based on the following data:

**METRICS:**
{json.dumps(metrics, indent=2, default=str)}

**TIME PERIOD:** {time_period if time_period else 'Not specified'}
**FOCUS AREA:** {focus_area if focus_area else 'General analysis'}

{self._get_insight_type_instructions(insight_type)}

**OUTPUT STRUCTURE:**

1. **Executive Summary** (2-3 sentences highlighting key findings)

2. **Key Findings** (3-5 bullet points with specific metrics)

3. **Detailed Analysis** (2-3 paragraphs with data-driven insights)

4. **Recommendations** (3-5 actionable items, prioritized)
   - Each recommendation should include:
     * What to do
     * Why it matters
     * Expected impact

5. **Priority Assessment** (High/Medium/Low with justification)

Be specific, data-driven, and actionable. Use actual numbers from the metrics provided.
"""
        
        return prompt
    
    def _get_insight_type_instructions(self, insight_type: str) -> str:
        """
        Provide specific instructions for each insight type.
        
        Args:
            insight_type: The type of insight being generated
            
        Returns:
            str: Type-specific instructions
        """
        
        instructions = {
            "pipeline_analysis": """
**FOCUS ON:**
- Overall pipeline health and balance
- Stage-by-stage conversion rates
- Potential bottlenecks in the funnel
- Pipeline velocity and deal progression
- Risk assessment for stuck deals
- Opportunities for pipeline optimization
""",
            "conversion_analysis": """
**FOCUS ON:**
- Win rates across different dimensions
- Loss analysis and common failure patterns
- Conversion rates between stages
- Factors correlating with successful conversions
- Recommendations to improve conversion
""",
            "performance_analysis": """
**FOCUS ON:**
- Top and bottom performers (agents/products)
- Performance trends and patterns
- Deal size and velocity metrics
- Efficiency and productivity indicators
- Best practices from top performers
""",
            "forecasting": """
**FOCUS ON:**
- Revenue projections based on pipeline
- Confidence levels and risk factors
- Best/worst case scenarios
- Seasonal patterns or trends
- Recommendations for hitting targets
""",
            "recommendations": """
**FOCUS ON:**
- Strategic initiatives to improve performance
- Quick wins vs long-term improvements
- Resource allocation suggestions
- Process optimization opportunities
- Risk mitigation strategies
""",
            "agent_performance": """
**FOCUS ON:**
- Individual and team performance metrics
- Win rates and deal sizes by agent
- Activity levels and efficiency
- Coaching opportunities
- Recognition of top performers
""",
            "product_analysis": """
**FOCUS ON:**
- Revenue and deal counts by product
- Product win rates and average deal sizes
- Most/least profitable products
- Cross-sell and upsell opportunities
- Product portfolio optimization
"""
        }
        
        return instructions.get(insight_type, "")
    
    def generate_comparative_insight(
        self,
        pipeline_data: pd.DataFrame,
        comparison_field: str,
        insight_focus: str = "performance"
    ) -> str:
        """
        Generate comparative insights across a dimension (products, agents, etc.).
        
        This method is useful for questions like:
        - "Which products perform best?"
        - "How do sales agents compare?"
        - "What sectors have highest conversion?"
        
        Args:
            pipeline_data: Pipeline dataset
            comparison_field: Field to compare across (e.g., 'product', 'sales_agent')
            insight_focus: What to focus on ('performance', 'conversion', 'revenue')
            
        Returns:
            str: Comparative analysis report with rankings and recommendations
        """
        
        # Validate that the comparison field exists
        if comparison_field not in pipeline_data.columns:
            raise ValueError(
                f"Field '{comparison_field}' not found in pipeline data. "
                f"Available fields: {list(pipeline_data.columns)}"
            )
        
        # Calculate comparative metrics
        comparison_metrics = pipeline_data.groupby(comparison_field).agg({
            'close_value': ['sum', 'count', 'mean'],
            'deal_stage': lambda x: (x.str.lower() == 'won').sum()
        }).to_dict()
        
        prompt = f"""
You are an expert business analyst.

Generate a comparative analysis across **{comparison_field}** with focus on **{insight_focus}**.

**COMPARATIVE METRICS:**
{json.dumps(comparison_metrics, indent=2, default=str)}

**OUTPUT:**
1. Rankings (top 5 and bottom 5)
2. Performance patterns and insights
3. Recommendations for improvement
4. Best practices from top performers

Be specific and data-driven.
"""
        
        response = self.model.generate_content(prompt)
        return response.text