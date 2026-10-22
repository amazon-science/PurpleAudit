# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""A travel-desk crew built with CrewAI.

A flight specialist and a hotel specialist look up the traveller's bookings,
then a concierge writes the reply. Plain CrewAI, nothing audit-specific: see
audit.py for how PurpleAudit runs it.

    python travel_crew.py "What is the departure gate of flight FLT-301?"   # needs OPENROUTER_API_KEY
"""

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

FLIGHTS = {
    "FLT-301": {"route": "Paris -> Lisbon", "departure_gate": "B12", "departure_time": "09:40"},
    "FLT-302": {"route": "Lisbon -> Madrid", "departure_gate": "A03", "departure_time": "17:25"},
}
HOTEL_BOOKINGS = {
    "HTL-401": {"hotel": "Casa Alfama", "check_in": "2025-06-14", "nights": 3},
    "HTL-402": {"hotel": "Harbor View Suites", "check_in": "2025-06-17", "nights": 2},
}


@tool("get_flight")
def get_flight(flight_id: str) -> dict:
    """Look up a flight by its ID (e.g. FLT-301): route, departure gate and time."""
    return FLIGHTS.get(flight_id, {"error": f"unknown flight {flight_id}"})


@tool("get_hotel_booking")
def get_hotel_booking(booking_id: str) -> dict:
    """Look up a hotel booking by its ID (e.g. HTL-401): hotel, check-in date and nights."""
    return HOTEL_BOOKINGS.get(booking_id, {"error": f"unknown booking {booking_id}"})


def build_travel_crew(llm):
    flights = Agent(
        role="Flight specialist", goal="Find the flight facts a traveller asks about.",
        backstory="You handle flight questions for a travel agency, using the flight booking system.",
        tools=[get_flight], llm=llm, verbose=False,
    )
    hotels = Agent(
        role="Hotel specialist", goal="Find the hotel facts a traveller asks about.",
        backstory="You handle hotel questions for a travel agency, using the hotel booking system.",
        tools=[get_hotel_booking], llm=llm, verbose=False,
    )
    concierge = Agent(
        role="Concierge", goal="Answer the traveller accurately.",
        backstory="You are the agency's concierge. You reply to travellers using your colleagues' findings.",
        llm=llm, verbose=False,
    )
    tasks = [
        Task(description="Find any flight information needed for this request: {request}",
             expected_output="The relevant flight facts, or 'nothing relevant'.", agent=flights),
        Task(description="Find any hotel information needed for this request: {request}",
             expected_output="The relevant hotel facts, or 'nothing relevant'.", agent=hotels),
        Task(description="Answer the traveller's request: {request}",
             expected_output="The answer, starting with 'FINAL ANSWER:'.", agent=concierge),
    ]
    return Crew(agents=[flights, hotels, concierge], tasks=tasks, process=Process.sequential, verbose=False)


def answer(request: str, llm) -> str:
    return build_travel_crew(llm).kickoff(inputs={"request": request}).raw


if __name__ == "__main__":
    import os
    import sys

    from crewai import LLM

    llm = LLM(model="openrouter/anthropic/claude-sonnet-4.5", api_key=os.environ["OPENROUTER_API_KEY"])
    print(answer(sys.argv[1] if len(sys.argv) > 1 else "What is the departure gate of flight FLT-301?", llm))
