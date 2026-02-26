#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
situational_awareness_demo.py

Conceptual Demonstration of Qwen3-VL-2B-Thinking Situational Awareness.
---------------------------------------------------------------------------------

### Description:
This script simulates the 'Thinking' model's ability to maintain a 'situational context'
across multiple interactions (chat turns) involving visual and spatial data.
It showcases:
1.  **Spatial Reasoning:** Tracking object positions and relative movement.
2.  **Long-Context Multimodality:** Remembering details from an initial visual input
    (Turn 1) when asked to perform an action several steps later (Turn 3).

### Key Features Simulated:
The script generates a detailed, structured internal plan (the "Thinking Trace")
that cross-references information from the chat history, which is the mechanism
the Qwen model uses to perform complex, multi-step agent planning.

### Usage:
Run the script, and observe how the simulated response in each turn builds upon
the information provided in the previous turn, demonstrating active memory.
"""
import time
from typing import List, Dict

# Define the data structure for a simulated chat history
ChatEntry = Dict[str, str]


def generate_reasoning_trace(chat_history: List[ChatEntry], current_turn: int) -> str:
    """
    Simulates the Qwen3-VL-Thinking model generating a multi-step reasoning trace
    based on the current input and the complete chat history (long context).
    """
    last_user_message = chat_history[-1]["content"]

    print(f"\n--- Turn {current_turn}: Agent Analysis ---")
    time.sleep(0.5)  # Simulate "thinking" delay

    # --- Logic for Demonstrating Situational Awareness ---

    if current_turn == 1:
        # Spatial Reasoning & Initialization
        trace = f"""
[THINKING TRACE: SPATIAL INITIALIZATION]
1. OBSERVE: Analyze initial environment image (Map).
2. GROUND: Identify key objects and assign coordinates (Simulation):
   - Red Box: (X=10, Y=5)
   - Blue Cylinder: (X=20, Y=15)
   - Green Sphere: (X=11, Y=4) -> Crucial Detail: Next to Red Box.
3. OUTPUT: Confirm initial situational context and store in Long-Context Memory.
"""
        response = f"Acknowledged. I have initialized the environment map. The Red Box is at (10, 5) and the Blue Cylinder is at (20, 15)."

    elif current_turn == 2:
        # Viewpoint Change & Relative Position Update
        trace = f"""
[THINKING TRACE: VIEWPOINT CHANGE & RELATIVE SPATIAL REASONING]
1. MEMORY RETRIEVAL: Retrieve previous context (Red Box at X=10, Y=5).
2. SPATIAL UPDATE: Process new instruction: "{last_user_message}".
   - Move: Two steps left (Change in X coordinate by -2).
   - New Agent Position: (X=8, Y=5).
3. CALCULATION: Determine new relative position of Blue Cylinder (X=20, Y=15) relative to the Red Box (X=10, Y=5).
   - Result: Cylinder is 10 units right and 10 units up from the Red Box.
4. OUTPUT: Confirm movement and calculated relative position.
"""
        response = f"I have moved two steps left. The Blue Cylinder is now 10 units to the right and 10 units above the Red Box."

    elif current_turn == 3:
        # Long-Context Memory Retrieval & Systematic Execution
        trace = f"""
[THINKING TRACE: LONG-CONTEXT MEMORY RETRIEVAL & EXECUTION]
1. MEMORY RETRIEVAL: Recall the starting position (Turn 1 context) and the key objects.
2. CRITICAL DETAIL RECALL: Identify the object "next to the Red Box" from Turn 1: **Green Sphere** (at X=11, Y=4).
3. ACTION PLAN (Systematic Execution):
   a) Calculate return path to start position.
   b) Navigate to Green Sphere coordinates (11, 4).
   c) Invoke 'Pickup' action on the Green Sphere.
4. OUTPUT: Confirm the object to be picked up and the action sequence.
"""
        response = f"Based on the initial map, the object next to the Red Box was the Green Sphere. I am executing the action: returning to the start position and picking up the Green Sphere."

    else:
        trace = "[THINKING TRACE: UNKNOWN COMMAND] Unable to generate specific situational reasoning. Please reset the scenario."
        response = "I am processing the request but need more context."

    print(f"🤖 **Internal Reasoning Trace:**\n{trace.strip()}")
    return response


# --- DEMONSTRATION OF MULTI-TURN SITUATIONAL AWARENESS ---

# Simulate a series of multimodal inputs (Image + Text)
CHAT_HISTORY: List[ChatEntry] = []
turn_count = 0


def run_turn(user_input: str, visual_input_type: str):
    global turn_count
    turn_count += 1

    print("\n" + "=" * 80)
    print(f"USER TURN {turn_count}:")
    if visual_input_type:
        # Simulates uploading an image alongside the text
        print(f"   [INPUT IMAGE: {visual_input_type}]")
    print(f"   [INPUT TEXT: {user_input}]")

    CHAT_HISTORY.append({"role": "user", "content": user_input, "visual": visual_input_type})

    # Generate the structured plan (Situational Awareness)
    agent_response = generate_reasoning_trace(CHAT_HISTORY, turn_count)

    print(f"\n✅ **Agent Response:**\n{agent_response}")
    CHAT_HISTORY.append({"role": "assistant", "content": agent_response})


if __name__ == "__main__":
    # 1. Turn 1: Initial Spatial Observation and State Setup
    run_turn(
        user_input="Analyze this environment map (Image 1). Where is the Red Box and the Blue Cylinder?",
        visual_input_type="Initial Map (Image 1)"
    )

    # 2. Turn 2: Viewpoint Change and Relative Spatial Reasoning
    run_turn(
        user_input="I moved two steps left and zoomed the camera. Now, where is the Blue Cylinder relative to the Red Box?",
        visual_input_type="Zoomed View (Image 2)"
    )

    # 3. Turn 3: Long-Context Memory Retrieval and Systematic Action
    run_turn(
        user_input="Now, forget the last move. Go back to the initial position and pick up the object that was next to the Red Box.",
        visual_input_type=""  # No new image, requires memory recall
    )

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE: The Qwen3-VL Thinking model maintains situational context across turns.")