"""
Protocol example demonstrating SQLite and JSONLines storage backends.

Note: add_trace_processor() registers observers globally. In production,
consider using a single observer per process or implement cleanup if the
SDK provides remove_trace_processor() in future versions.
"""

from agents import Agent, Runner, add_trace_processor
import reagentic.providers.openrouter as openrouter

from reagentic.protocol import (
    ProtocolConfig,
    ProtocolDetailLevel,
    ProtocolExtractor,
    ProtocolObserver,
    ProtocolWriter,
    SQLiteProtocolStorage,
    JSONLinesProtocolStorage,
)


def main() -> None:
    provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
    agent = Agent(
        name="Protocol Demo",
        instructions="You are a concise assistant.",
        model=provider.get_openai_model(),
    )

    config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)

    # SQLite storage example
    sqlite_storage = SQLiteProtocolStorage("protocol.db")
    writer = ProtocolWriter(sqlite_storage)
    observer = ProtocolObserver(ProtocolExtractor(config), writer)

    # Register as trace processor for span/trace events
    # Note: trace processor remains registered globally after shutdown
    add_trace_processor(observer)

    try:
        result = Runner.run_sync(agent, "Write a short greeting.", hooks=observer)
        print(result.final_output)
    finally:
        observer.shutdown()

    # JSONLines storage example
    jsonl_storage = JSONLinesProtocolStorage("protocol.jsonl")
    jsonl_writer = ProtocolWriter(jsonl_storage)
    json_observer = ProtocolObserver(ProtocolExtractor(config), jsonl_writer)

    add_trace_processor(json_observer)

    try:
        result = Runner.run_sync(agent, "Write a one-line quote.", hooks=json_observer)
        print(result.final_output)
    finally:
        json_observer.shutdown()


if __name__ == "__main__":
    main()
