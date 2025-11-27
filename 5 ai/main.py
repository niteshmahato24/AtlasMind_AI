import asyncio
import os
import json
import logging
import uuid # <-- NEW: For generating unique IDs for the cache
from typing import List, Dict, Any, Union

# --- Gemini and Pydantic Imports ---
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, ValidationError
# -----------------------------------

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Request, Form, Query # <-- Added Query
from fastapi.responses import HTMLResponse, RedirectResponse # <-- Imported RedirectResponse
from fastapi.templating import Jinja2Templates
# -----------------------

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Cache for Post/Redirect/Get pattern (PRG)
# Used to hold data between the POST and the GET request.
TRIP_CACHE: Dict[str, Any] = {} 

# Define model and client variables outside the try/except block 
GEMINI_MODEL = "gemini-2.5-flash"
client = None
client_status = "Not Initialized"

# Initialize the Gemini Client
try:
    client = genai.Client()
    client_status = f"Ready! Using {GEMINI_MODEL}"
    logger.info(f"Gemini Client initialized. Using model: {GEMINI_MODEL}")
except Exception as e:
    client_status = f"Initialization Error: API Key Missing/Invalid."
    logger.error(f"Failed to initialize Gemini Client: {e}. Ensure GEMINI_API_KEY is set.")

# FastAPI Application Setup
app = FastAPI(title="AtlasMind AI: Multi-Agent Travel Planner")
templates = Jinja2Templates(directory="templates")

# --- SYSTEM PROMPTS ---
SYSTEM_PROMPTS = {
    "orchestrator": "You are the AtlasMind Orchestrator. Manage and delegate tasks to specialized sub-agents. Compile their outputs into a final, comprehensive trip plan.",
    "destination": "You are the Destination Research Agent. Find attractions, local transport, and weather for the given destination and dates. Your primary output is the weather and summary data.",
    "itinerary": "You are the Itinerary Planning Agent. Create a highly optimized, day-by-day plan based on all context (preferences, weather, cost). Minimize travel time and balance activities/day.",
    "budget": "You are the Budget Estimator Agent. Calculate a realistic daily and total cost estimate based on the destination and budget level. Provide detailed cost breakdown.",
    "packing": "You are the Packing List Agent. Generate a detailed, weather-appropriate packing list based on the full trip context (weather, duration, activities)."
}

# ------------------------------------------------------------------
# --- Pydantic Models (Including CostBreakdown Fix) ---
# ------------------------------------------------------------------

class TripRequest(BaseModel):
    destination: str = Field(..., json_schema_extra={"example": "Paris, France"})
    duration_days: int = Field(..., ge=1, le=14, json_schema_extra={"example": 5})
    traveler_preferences: str = Field(..., json_schema_extra={"example": "We love museums, history, and fine dining."})
    budget_level: str = Field(..., json_schema_extra={"example": "Mid-Range"})
    travel_dates: str = Field(..., json_schema_extra={"example": "June 2026"})
    
    class Config:
        extra = 'forbid' 

class WeatherInfo(BaseModel):
    temperature_range: str = Field(..., description="Expected temperature range and units (e.g., 15°C to 25°C).")
    conditions: str = Field(..., description="Overall weather conditions (e.g., Mostly sunny with a chance of afternoon showers).")
    clothing_advice: str = Field(..., description="Specific clothing advice based on the weather and season.")
    
    class Config:
        extra = 'forbid'

# --- NEW MODEL FOR EXPLICIT COST BREAKDOWN ---
class CostBreakdown(BaseModel):
    Accommodation: str = Field(..., description="Estimated cost for daily lodging, including type (e.g., $150 per night for mid-range hotel).")
    Food_Dining: str = Field(..., description="Estimated cost for daily food and dining (e.g., $80 for all meals).")
    Local_Transport: str = Field(..., description="Estimated cost for local transportation (e.g., $20 for metro passes/taxis).")
    Activities_Fees: str = Field(..., description="Estimated cost for daily activities and entrance fees (e.g., $45 per day).")
    Miscellaneous: str = Field(..., description="Estimated cost for extra spending and contingency (e.g., $30 per day).")
    
    class Config:
        extra = 'forbid' 

# --- UPDATED BUDGET ESTIMATE MODEL ---
class BudgetEstimate(BaseModel):
    daily_cost_usd: float = Field(..., description="Estimated average daily cost in USD (e.g., 250.0).")
    details: CostBreakdown = Field(..., description="Detailed cost breakdown for Accommodation, Food, Transport, Activities, etc.")
    warning: str = Field(..., description="A note about the estimate's accuracy or seasonality.")
    
    class Config:
        extra = 'forbid' 

class PackingList(BaseModel):
    essentials: List[str] = Field(..., description="List of must-have items (e.g., Passport, Adapter).")
    clothing: List[str] = Field(..., description="List of specific clothing items for the trip.")
    notes: str = Field(..., description="Final packing tips.")
    
    class Config:
        extra = 'forbid'

class ItineraryActivity(BaseModel):
    time_period: str = Field(..., description="Time of day (e.g., Morning, Afternoon, Evening).")
    name: str = Field(..., description="Name of the attraction or activity.")
    details: str = Field(..., description="Brief details about the activity and estimated duration.")
    transport_tip: str = Field(..., description="Best way to get there from the previous location or hotel.")
    
    class Config:
        extra = 'forbid'

class ItineraryDay(BaseModel):
    day_number: int = Field(..., description="The sequential day number (1, 2, 3...).")
    summary: str = Field(..., description="A one-sentence summary for the day.")
    activities: List[ItineraryActivity] = Field(..., description="A list of activities planned for the day.")
    meal_recommendation: str = Field(..., description="A suggestion for lunch or dinner.")
    
    class Config:
        extra = 'forbid'

class TripSummary(BaseModel):
    summary: str = Field(..., description="A high-level overview of the entire trip.")
    mood: str = Field(..., description="The overall theme or mood of the trip (e.g., Historical, Romantic, Adventurous).")
    
    class Config:
        extra = 'forbid'
        
# ------------------------------------------------------------------
# --- The Core Gemini Agent Function ---
# ------------------------------------------------------------------
async def generate_structured_output(
    system_prompt_key: str,
    user_prompt: str,
    pydantic_model: BaseModel,
    context: str = ""
) -> BaseModel:
    """
    Calls the Gemini API to execute a specialized 'Agent' task.
    Includes recursive schema sanitization (additionalProperties removal).
    """
    
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini Client is not initialized due to missing API Key.")
        
    logger.info(f"Agent '{pydantic_model.__name__}' is executing...")
    
    full_system_instruction = SYSTEM_PROMPTS[system_prompt_key] + (f"\n\nContext Provided: {context}" if context else "")
    
    # --- CRITICAL FIX: Robust, Recursive Schema Sanitization ---
    schema_dict = pydantic_model.model_json_schema()

    def clean_properties(d: dict):
        """Recursively removes 'additionalProperties' from a dictionary schema."""
        if isinstance(d, dict):
            d.pop("additionalProperties", None)
            
            for key, value in d.items():
                if key == "properties" and isinstance(value, dict):
                    for prop_value in value.values():
                        clean_properties(prop_value)
                elif key == "items" and isinstance(value, dict):
                    clean_properties(value)
                elif isinstance(value, dict):
                    clean_properties(value)

    clean_properties(schema_dict)

    if "$defs" in schema_dict:
        for def_name in list(schema_dict["$defs"].keys()):
            clean_properties(schema_dict["$defs"][def_name])
    # -----------------------------------------------------------
    
    config = types.GenerateContentConfig(
        system_instruction=full_system_instruction,
        response_mime_type="application/json",
        response_schema=schema_dict,
        temperature=0.4
    )
    
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=config
        )
        
        if response.text:
            validated_output = pydantic_model.model_validate_json(response.text)
            logger.info(f"Agent '{pydantic_model.__name__}' completed successfully.")
            return validated_output
        else:
            if response.candidates and response.candidates[0].finish_reason != 'STOP':
                raise APIError(f"Gemini response finished early: {response.candidates[0].finish_reason.name}")
            else:
                raise APIError("Gemini returned no valid JSON content (response.text is empty).")

    except APIError as e:
        logger.error(f"Gemini API Error for {pydantic_model.__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"LLM API Error: {e.message}")
    except ValidationError as e:
        logger.error(f"Pydantic Validation Error for {pydantic_model.__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Pydantic Validation Error: LLM returned invalid JSON structure.")
    except Exception as e:
        logger.error(f"Unexpected error in {pydantic_model.__name__} agent: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected Agent Error: {type(e).__name__} - {e}")


# --- Core Agent Logic Functions (Helper functions) ---

async def get_research_and_summary(req: TripRequest):
    """Parallel Agent: Destination Research Agent and Trip Summary Agent."""
    context = (
        f"Destination: {req.destination}. Dates: {req.travel_dates}. Duration: {req.duration_days} days. "
        f"Preferences: {req.traveler_preferences}. Budget: {req.budget_level}."
    )
    weather_task = generate_structured_output("destination", "Provide the weather forecast, conditions, and clothing advice for the trip.", WeatherInfo, context)
    summary_task = generate_structured_output("destination", "Provide a high-level summary and overall mood/theme for the trip.", TripSummary, context)
    budget_task = generate_structured_output("budget", "Calculate and provide a daily and total budget estimate based on the details.", BudgetEstimate, context)
    return await asyncio.gather(weather_task, summary_task, budget_task)

async def get_packing_list_agent(req: TripRequest, weather: WeatherInfo) -> PackingList:
    """Sequential Agent: Packing List Agent."""
    context = (
        f"Trip Context: {req.destination} for {req.duration_days} days. "
        f"Weather Info received from Destination Agent: {weather.model_dump_json()}."
    )
    user_prompt = "Generate the detailed, appropriate packing list now based on the weather data provided."
    return await generate_structured_output("packing", user_prompt, PackingList, context)

async def get_full_itinerary_agent(req: TripRequest, weather: WeatherInfo, budget: BudgetEstimate) -> List[ItineraryDay]:
    """Loop Agent: Itinerary Planning Agent."""
    base_context = (
        f"Destination: {req.destination}. Days: {req.duration_days}. Preferences: {req.traveler_preferences}. "
        f"Weather: {weather.conditions}. Daily Budget: ${budget.daily_cost_usd}."
        f"Your task is to generate ONE day's itinerary in each call."
    )
    itinerary_tasks = []
    for day_num in range(1, req.duration_days + 1):
        context = base_context + f"\n\n*** FOCUS: Generate Itinerary for Day {day_num} ONLY. ***"
        task = generate_structured_output(
            "itinerary", 
            f"Generate a detailed, optimized itinerary for Day {day_num}. Ensure activities fit the user's preferences.",
            ItineraryDay,
            context
        )
        itinerary_tasks.append(task)
    return await asyncio.gather(*itinerary_tasks)


# ------------------------------------------------------------------
# --- FastAPI Endpoints (PRG Pattern Implemented) ---
# ------------------------------------------------------------------

@app.get("/", tags=["UI"], response_class=HTMLResponse)
async def root(request: Request):
    """Serves the main form page."""
    return templates.TemplateResponse("index.html", {"request": request, "agent_status": client_status})

@app.post("/process-trip/", tags=["Orchestrator"])
async def process_trip_form(
    request: Request,
    destination: str = Form(...),
    duration_days: int = Form(..., alias="days", ge=1, le=14), 
    traveler_preferences: str = Form(...),
    budget_level: str = Form(...),
    travel_dates: str = Form(..., alias="travel_dates") 
) -> RedirectResponse: # <-- Endpoint now returns a RedirectResponse
    """
    The AtlasMind Orchestrator: Manages the flow of Sub-Agents and redirects to the results page.
    """
    
    # 1. Validation & Input Collection
    try:
        trip_request = TripRequest(
            destination=destination,
            duration_days=duration_days,
            traveler_preferences=traveler_preferences,
            budget_level=budget_level,
            travel_dates=travel_dates
        )
    except ValidationError as e:
        logger.error(f"Form validation error: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": "Invalid form data provided."}, status_code=400)

    logger.info(f"AtlasMind Orchestrator: Starting plan for {trip_request.destination}")
    
    if client is None:
        logger.error("Orchestrator blocked: Gemini Client is not initialized.")
        return templates.TemplateResponse("error.html", {"request": request, "message": client_status}, status_code=503)

    try:
        # 2. Parallel Execution (Weather, Summary, Budget)
        weather, summary, budget = await get_research_and_summary(trip_request)
        logger.info("Orchestrator: Completed Parallel Research and Budgeting Agents.")
        
        # 3. Sequential Execution (Packing List)
        packing = await get_packing_list_agent(trip_request, weather)
        logger.info("Orchestrator: Completed Sequential Packing List Agent.")

        # 4. Loop Agents Execution (Itinerary)
        itinerary_data = await get_full_itinerary_agent(trip_request, weather, budget)
        logger.info(f"Orchestrator: Completed Loop Agent for {len(itinerary_data)} days.")
        
        # 5. Final Compilation and Redirect (PRG Pattern)
        trip_id = str(uuid.uuid4()) # Generate a unique ID
        
        # Store all compiled data in the global cache
        TRIP_CACHE[trip_id] = {
            "request_data": trip_request, 
            "travel_summary": summary.model_dump(),
            "weather": weather.model_dump(), 
            "packing_list": packing.model_dump(),
            "budget": budget.model_dump(),
            "itinerary": [day.model_dump() for day in itinerary_data]
        }
        
        # Redirect the user to the GET endpoint with the trip ID
        return RedirectResponse(url=f"/trip-result/?id={trip_id}", status_code=303)
    
    except HTTPException as e:
        logger.error(f"Caught Agent Execution Error: {e.detail}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": f"Agent Execution Error: {e.detail}"}, 
            status_code=e.status_code
        )
    except Exception as e:
        logger.critical(f"CRITICAL Orchestration Failure: {e}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": f"A critical failure stopped the plan: {type(e).__name__} - {e}"}, 
            status_code=500
        )

@app.get("/trip-result/", tags=["UI"], response_class=HTMLResponse)
async def get_trip_result(request: Request, trip_id: str = Query(..., alias="id")): # <-- New GET endpoint
    """
    GET endpoint to retrieve and display the trip plan using the cached ID.
    This fixes the browser navigation issue.
    """
    if trip_id not in TRIP_CACHE:
        # If the trip ID is missing (e.g., if the user refreshes after cache timeout)
        return templates.TemplateResponse("error.html", {"request": request, "message": "Trip plan not found or expired."}, status_code=404)
        
    # Retrieve and remove the data from the cache
    trip_data = TRIP_CACHE.pop(trip_id) 
    
    # Render the final results page
    return templates.TemplateResponse("result.html", {
        "request": request,
        "request_data": trip_data["request_data"], 
        "travel_summary": trip_data["travel_summary"],
        "weather": trip_data["weather"], 
        "packing_list": trip_data["packing_list"],
        "budget": trip_data["budget"],
        "itinerary": trip_data["itinerary"]
    })
