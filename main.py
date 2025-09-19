import os
import random
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# ============================
# Load Environment & Model Setup
# ============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

config = RunConfig(model=model, tracing_disabled=True)

# ============================
# Tools
# ============================
@function_tool
def roll_dice() -> str:
    """
    Simulate a dice roll.

    Used during battles, alien encounters, or to determine random outcomes.
    Returns a string showing the rolled number.
    """
    return f"🎲 Dice Roll: {random.randint(1, 4)}"


@function_tool
def generate_space_event() -> str:
    """
    Generate a random space event.

    The event can be positive (finding treasures), negative (dangerous encounters),
    or neutral (unexpected discoveries).
    Returns a string describing the event.
    """
    events = [
        "🚀 You discovered a hidden wormhole!",
        "👾 An alien ship is approaching fast!",
        "💎 You found rare cosmic crystals!",
        "☄️ Your spaceship is hit by a meteor shower!"
    ]
    return random.choice(events)

# ============================
# Agents
# ============================
narrator_agent = Agent(
    name="NarratorAgent",
    instructions=(
        "Narrate the space adventure in an engaging way. "
        "Ask the player for choices and guide them into the story."
    ),
    model=model
)

alien_agent = Agent(
    name="AlienAgent",
    instructions=(
        "Handle alien encounters and unexpected dangers. "
        "Make use of dice rolls and random space events to shape the outcome."
    ),
    model=model,
    tools=[roll_dice, generate_space_event]
)

reward_agent = Agent(
    name="RewardAgent",
    instructions=(
        "Grant futuristic rewards, spaceship upgrades, or cosmic treasures "
        "to the player after surviving encounters."
    ),
    model=model
)

# ============================
# Main Game Loop
# ============================
def main():
    """
    Run the main Galactic Quest game loop.

    Steps:
        1. Greet the player.
        2. Ask for their decision (explore or orbit).
        3. Narrate the story using the narrator agent.
        4. Trigger an alien encounter with dice and random events.
        5. Provide rewards through the reward agent.
        6. Ask if the player wants to continue or end the adventure.
    """
    print("\n🪐 Welcome to Galactic Quest!\n")

    while True:
        choice = input("🌌 Do you explore a new planet or stay in orbit? → ")

        # Story narration
        story = Runner.run_sync(narrator_agent, choice, run_config=config)
        print("\n📖 Story:", story.final_output)

        # Alien encounter
        encounter = Runner.run_sync(alien_agent, "Alien encounter", run_config=config)
        print("\n👽 Encounter:", encounter.final_output)

        # Reward or outcome
        reward = Runner.run_sync(reward_agent, "Give futuristic reward", run_config=config)
        print("\n🛸 Reward:", reward.final_output)

        # Play again option
        again = input("\n🔁 Do you want another space adventure? (yes/no): ").lower()
        if again not in ("yes", "y"):
            print("\n🙏 Thanks for playing Galactic Quest! 🚀✨")
            break


if __name__ == "__main__":
    main()
