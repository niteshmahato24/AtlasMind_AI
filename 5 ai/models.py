from typing import List, Dict
from pydantic import BaseModel, Field

# --- 1. Request Model (Input from User) ---
class TripRequest(BaseModel):
    """User input for the trip planning."""
    destination: str = Field(..., example="Paris, France", description="The city and country of the trip.")
    duration_days: int = Field(..., example=5, ge=1, le=14, description="Total length of the trip in days.")
    traveler_preferences: str = Field(..., example="We love museums, history, and fine dining.", description="Detailed user preferences for activities.")
    budget_level: str = Field(..., example="Mid-Range", description="User's budget preference (Economy, Mid-Range, Luxury).")
    travel_dates: str = Field(..., example="June 2026", description="The planned month or season of travel.")

# --- 2. Agent Output Models ---
class WeatherInfo(BaseModel):
    """Structured output for the Destination Agent's weather research (Tool Use)."""
    temperature_range: str = Field(..., description="Expected temperature range and units (e.g., 15°C to 25°C).")
    conditions: str = Field(..., description="Overall weather conditions (e.g., Mostly sunny with a chance of afternoon showers).")
    clothing_advice: str = Field(..., description="Specific clothing advice based on the weather and season.")

class BudgetEstimate(BaseModel):
    """Structured output for the Budget Estimator Agent (Tool Use/Code Execution)."""
    daily_cost_usd: float = Field(..., description="Estimated average daily cost in USD (e.g., 250.0).")
    details: Dict[str, str] = Field(..., description="Cost breakdown for Accommodation, Food, Transport, Activities, etc.")
    warning: str = Field(..., description="A note about the estimate's accuracy or seasonality.")

class PackingList(BaseModel):
    """Structured output for the Packing List Agent (Sequential Agent)."""
    essentials: List[str] = Field(..., description="List of must-have items (e.g., Passport, Adapter).")
    clothing: List[str] = Field(..., description="List of specific clothing items for the trip.")
    notes: str = Field(..., description="Final packing tips.")

class ItineraryActivity(BaseModel):
    """A single activity within a day's plan."""
    time_period: str = Field(..., description="Time of day (e.g., Morning, Afternoon, Evening).")
    name: str = Field(..., description="Name of the attraction or activity.")
    details: str = Field(..., description="Brief details about the activity and estimated duration.")
    transport_tip: str = Field(..., description="Best way to get there from the previous location or hotel.")

class ItineraryDay(BaseModel):
    """Structured output for a single day of the itinerary (Loop Agent)."""
    day_number: int = Field(..., description="The sequential day number (1, 2, 3...).")
    summary: str = Field(..., description="A one-sentence summary for the day.")
    activities: List[ItineraryActivity] = Field(..., description="A list of activities planned for the day.")
    meal_recommendation: str = Field(..., description="A suggestion for lunch or dinner.")

class TripSummary(BaseModel):
    """Overall summary of the trip (Destination Research Agent)."""
    summary: str = Field(..., description="A high-level overview of the entire trip.")
    mood: str = Field(..., description="The overall theme or mood of the trip (e.g., Historical, Romantic, Adventurous).")
