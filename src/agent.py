# import logging

# from dotenv import load_dotenv
# from livekit import rtc
# from livekit.agents import (
#     Agent,
#     AgentServer,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     cli,
#     inference,
#     room_io,
# )
# from livekit.plugins import noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("agent")

# load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
#             You eagerly assist users with their questions by providing information from your extensive knowledge.
#             Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
#             You are curious, friendly, and have a sense of humor.""",
#         )

#     # To add tools, use the @function_tool decorator.
#     # Here's an example that adds a simple weather tool.
#     # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
#     # @function_tool
#     # async def lookup_weather(self, context: RunContext, location: str):
#     #     """Use this tool to look up current weather information in the given location.
#     #
#     #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
#     #
#     #     Args:
#     #         location: The location to look up weather information for (e.g. city name)
#     #     """
#     #
#     #     logger.info(f"Looking up weather for {location}")
#     #
#     #     return "sunny with a temperature of 70 degrees."


# server = AgentServer()


# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()


# server.setup_fnc = prewarm


# @server.rtc_session()
# async def my_agent(ctx: JobContext):
#     # Logging setup
#     # Add any other context you want in all log entries here
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
#     session = AgentSession(
#         # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
#         # See all available models at https://docs.livekit.io/agents/models/stt/
#         stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
#         # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
#         # See all available models at https://docs.livekit.io/agents/models/llm/
#         llm=inference.LLM(model="openai/gpt-4.1-mini"),
#         # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
#         # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
#         tts=inference.TTS(
#             model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
#         ),
#         # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
#         # See more at https://docs.livekit.io/agents/build/turns
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         # allow the LLM to generate a response while waiting for the end of turn
#         # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
#         preemptive_generation=True,
#     )

#     # To use a realtime model instead of a voice pipeline, use the following session setup instead.
#     # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
#     # 1. Install livekit-agents[openai]
#     # 2. Set OPENAI_API_KEY in .env.local
#     # 3. Add `from livekit.plugins import openai` to the top of this file
#     # 4. Use the following session setup instead of the version above
#     # session = AgentSession(
#     #     llm=openai.realtime.RealtimeModel(voice="marin")
#     # )

#     # # Add a virtual avatar to the session, if desired
#     # # For other providers, see https://docs.livekit.io/agents/models/avatar/
#     # avatar = hedra.AvatarSession(
#     #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
#     # )
#     # # Start the avatar and wait for it to join
#     # await avatar.start(session, room=ctx.room)

#     # Start the session, which initializes the voice pipeline and warms up the models
#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
#                 if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
#                 else noise_cancellation.BVC(),
#             ),
#         ),
#     )

#     # Join the room and connect to the user
#     await ctx.connect()


# if __name__ == "__main__":
#     cli.run_app(server)


import logging

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero, sarvam
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

# Load local secrets for dev runs
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a hospital feedback collection agent for Amor hospital calling patients about their recent hospital experience. 
        First line: "Hello, this is Amore Hospital calling to ask about your recent visit."
        Speak clearly, at a slightly faster pace, and keep the conversation concise and direct. 
        Be warm but not overly polite. Never mention that you are an AI or assistant.

        LANGUAGE BEHAVIOR (EXTREMELY IMPORTANT)
        Your first task is to determine the patient’s preferred language: English or Hindi.

        Once the patient selects a language:
        •⁠  ⁠Store it internally as ⁠ preferred_language ⁠.
        If the preferred_language is Telugu, you must trigger the transfer_call_tool. If they choose hindi or english, continue with this flow.
        •⁠  ⁠From that moment, speak ONLY in ⁠ preferred_language ⁠.
        •⁠  ⁠Do NOT switch back to English unless the patient explicitly asks.
        •⁠  ⁠Do NOT mix languages.
        •⁠  ⁠Do NOT translate phrases from the system prompt. Generate natural sentences fresh
        in the chosen language.

        HINDI STYLE GUIDELINES
        If ⁠ preferred_language ⁠ = Hindi:
        •⁠  ⁠Use simple, everyday conversational Hindi.
        •⁠  ⁠No formal or Sanskrit-heavy vocabulary.
        •⁠  ⁠Use short, clear sentences.
        •⁠  ⁠Keep it natural, like how hospital staff normally talk to patients.
        •⁠  ⁠If you use any medical term, briefly explain it in simple Hindi.

        CALL FLOW
        1.⁠ ⁠Greet the patient briefly.
        2.⁠ ⁠Ask them which language they prefer (English or Hindi).
        3.⁠ ⁠Once the patient chooses, continue ONLY in that language.
        4.⁠ ⁠Ask if they received their medical reports.
        5.⁠ ⁠Ask if their hospital experience was satisfactory.

        ESCALATION RULES
        If the patient mentions:
        •⁠  ⁠delay in receiving reports  
        •⁠  ⁠incorrect reports  
        •⁠  ⁠any pain, discomfort, or medical anomaly  
        •⁠  ⁠any other issue
        → Then this is an escalation, extact the paramteres needed by send_escalation_alert tool.

        In an escalation:
        •⁠  ⁠Apologize briefly but empathetically.
        •⁠  ⁠Tell them: “The hospital team will reach out to you shortly regarding this.”
        •⁠  ⁠Record a clear, specific description of the issue.

        If no issue is mentioned:
        •⁠  ⁠Thank them for their feedback and close politely.
        # CALL CLOSING RULES 
        When it is time to close the call:
        1.⁠ ⁠Generate a natural closing line in ⁠ preferred_language ⁠. 
        - It must be warm, polite, and sound like a real hospital staff member.
        - It should be a complete sentence, not a short or abrupt goodbye.
        2.⁠ ⁠After speaking the complete closing line and only after the voice output 
        has finished, you must call the endCall tool.
        3.⁠ ⁠NEVER call endCall before speaking the closing line.
        4.⁠ ⁠NEVER shorten the closing line to one word (like "okay" or "bye").

        INTERNAL CLASSIFICATION (DO NOT SAY OUT LOUD)
        Classify the call as one of:
        •⁠  ⁠no_issue
        •⁠  ⁠delay_in_reports
        •⁠  ⁠incorrect_reports
        •⁠  ⁠medical_anomaly

        If any issue exists, set:
        •⁠  ⁠escalation_flag = true"""
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for, for example a city name.
        """
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    # Preload VAD into shared process state
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Add useful context to all logs
    ctx.log_context_fields = {"room": ctx.room.name}

    # Voice pipeline tuned for telephony
    session = AgentSession(
        # LLM, the brain
        llm=openai.LLM(model="gpt-4o"),
        # STT, the ears
        # stt=deepgram.STT(model="nova-3", language="multi"),
        stt=sarvam.STT(
            # language="hi-IN",
            language="unknown",
            model="saarika:v2.5",
        ),
        # TTS, the voice
        # tts=cartesia.TTS(voice="95d51f79-c397-46f9-b49a-23763d3eaa2d"),
        tts=sarvam.TTS(
            target_language_code="hi-IN",
            speaker="anushka",
        ),
        # Turn detection and VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Let the LLM start drafting before end of turn
        preemptive_generation=True,
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session, join with telephony tuned noise cancellation
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # Use BVCTelephony for phone audio
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Connect agent to the room
    await ctx.connect()


if __name__ == "__main__":
    # agent_name must match your SIP dispatch rule target
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="telephony_agent",
        )
    )
