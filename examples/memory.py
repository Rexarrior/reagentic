from agents import Agent, Runner
import reagentic.providers.openrouter as openrouter
from reagentic.subsystems.memory import FileBasedMemory


HAIKU_PROMPT = 'Write a haiku about recursion in programming.'
HAIKU_SYSTEM_PROMPT = '''You are a haiku writing assistant. Your task is to write ONLY haiku poems - nothing else.

STRICT RULES:
- Write ONLY the haiku (3 lines, 5-7-5 syllable pattern)
- NO commentary, explanations, or additional text
- NO mentions of tools, memory, or previous haiku
- NO questions or requests for clarification
- NO parenthetical comments or meta-text
- Use memory context to avoid repeating themes or phrases you've used before
- Each haiku should be unique and original

Output format: Just the 3 lines of the haiku, nothing more.'''
HAIKU_AGENT_NAME = 'Haiku Assistant'

provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
# provider = openrouter.OpenrouterProvider(openrouter.GPT_4_1)


def run_agent_memory_through_prompt():
    """Original example - extend context manually."""
    print("=== Memory Through Prompt (Original) ===")
    memory = FileBasedMemory()
    agent = Agent(
        name=HAIKU_AGENT_NAME, 
        instructions=HAIKU_SYSTEM_PROMPT, 
        model=provider.get_openai_model()
    )
    for i in range(3):
        print(f'Iteration {i}')
        # Use enrichment function instead of manual concatenation
        enriched_prompt = memory.enrich_raw(HAIKU_PROMPT)
        result = Runner.run_sync(agent, enriched_prompt)
        result = result.final_output
        memory.append_raw(f"Iteration {i}: {result}\n")
        print(result)


def run_agent_memory_through_tools():
    """Original example - use memory tools."""
    print("=== Memory Through Tools (Original) ===")
    memory = FileBasedMemory()
    agent = Agent(
        name=HAIKU_AGENT_NAME,
        instructions=(
            HAIKU_SYSTEM_PROMPT
            + '\nYou must save information about haiku that your wrote to memory using append_raw_t tool'
            + '\nYou must read information about haiku that your wrote earlier from memory using read_raw_t tool'
        ),
        model=provider.get_openai_model(),
    )
    # Connect default tools to agent - read, write and append text to raw memory
    memory.connect_tools(agent)
    for i in range(3):
        print(f'Iteration {i}')
        result = Runner.run_sync(agent, HAIKU_PROMPT)
        result = result.final_output
        print(result)
    print(f'The memory content is:')
    print(memory.read_raw())


def run_agent_with_enriched_prompts():
    """New example - using enrichment functions to enhance prompts."""
    print("=== Memory Enrichment Functions ===")
    memory = FileBasedMemory()
    
    # Set up some initial memory content
    memory.modify_structure("This memory stores haiku writing progress, themes, and user preferences for poetry style.")
    memory.write_by_key("preferred_style", "traditional 5-7-5 syllable pattern")
    memory.write_by_key("favorite_theme", "nature and technology")
    memory.write_by_key("user_name", "Poetry Enthusiast")
    memory.append_raw("Session started: Beginning haiku writing practice.")
    
    agent = Agent(
        name=HAIKU_AGENT_NAME,
        instructions=HAIKU_SYSTEM_PROMPT + "\nUse the memory context provided to write better, more personalized haiku.",
        model=provider.get_openai_model()
    )
    
    # Example 1: Enrich with raw memory only
    print("\n--- Example 1: Raw Memory Enrichment ---")
    enriched_prompt = memory.enrich_raw(HAIKU_PROMPT)
    print("Enriched prompt preview:")
    print(enriched_prompt[:200] + "..." if len(enriched_prompt) > 200 else enriched_prompt)
    
    result = Runner.run_sync(agent, enriched_prompt)
    haiku1 = result.final_output
    print(f"Result: {haiku1}")
    memory.append_raw(f"Generated haiku: {haiku1}")
    
    # Example 2: Enrich with keys only
    print("\n--- Example 2: Key-Based Memory Enrichment ---")
    enriched_prompt = memory.enrich_keys("Write a haiku that incorporates the user's preferences.")
    print("Enriched prompt preview:")
    print(enriched_prompt[:200] + "..." if len(enriched_prompt) > 200 else enriched_prompt)
    
    result = Runner.run_sync(agent, enriched_prompt)
    haiku2 = result.final_output
    print(f"Result: {haiku2}")
    
    # Example 3: Enrich with structure only
    print("\n--- Example 3: Structure Memory Enrichment ---")
    enriched_prompt = memory.enrich_structure("Write a haiku following the memory system's purpose.")
    print("Enriched prompt preview:")
    print(enriched_prompt[:200] + "..." if len(enriched_prompt) > 200 else enriched_prompt)
    
    result = Runner.run_sync(agent, enriched_prompt)
    haiku3 = result.final_output
    print(f"Result: {haiku3}")
    
    # Example 4: Full enrichment with all memory types
    print("\n--- Example 4: Full Memory Enrichment ---")
    enriched_prompt = memory.enrich_full("Create a final haiku that reflects everything learned.")
    print("Enriched prompt preview (first 300 chars):")
    print(enriched_prompt[:300] + "..." if len(enriched_prompt) > 300 else enriched_prompt)
    
    result = Runner.run_sync(agent, enriched_prompt)
    final_haiku = result.final_output
    print(f"Final Result: {final_haiku}")
    
    # Show final memory state
    print("\n--- Final Memory State ---")
    print("Structure:", memory.read_structure())
    print("Keys:", memory.read_keys())
    print("Raw content length:", len(memory.read_raw()), "characters")


def run_agent_with_dynamic_enrichment():
    """Advanced example - dynamic enrichment based on context."""
    print("\n=== Dynamic Memory Enrichment ===")
    memory = FileBasedMemory()
    
    # Set up memory for a conversation assistant
    memory.modify_structure("Conversation memory tracking user interactions, preferences, and context.")
    memory.write_by_key("conversation_count", "0")
    memory.write_by_key("user_mood", "neutral")
    memory.write_by_key("topic_focus", "general")
    
    agent = Agent(
        name="Conversation Assistant",
        instructions="You are a helpful conversation assistant. Use memory context to provide personalized responses.",
        model=provider.get_openai_model()
    )
    
    conversations = [
        ("Hello! I'm feeling excited about learning poetry today.", "greeting"),
        ("Can you help me understand different types of poems?", "educational"),
        ("I'm a bit tired now, maybe something simple?", "simple_request"),
        ("Thanks for all your help today!", "farewell")
    ]
    
    for i, (user_input, context_type) in enumerate(conversations):
        print(f"\n--- Conversation {i+1}: {context_type} ---")
        
        # Update memory based on context
        memory.write_by_key("conversation_count", str(i+1))
        if "excited" in user_input.lower():
            memory.write_by_key("user_mood", "excited")
        elif "tired" in user_input.lower():
            memory.write_by_key("user_mood", "tired")
        
        if "poetry" in user_input.lower() or "poem" in user_input.lower():
            memory.write_by_key("topic_focus", "poetry")
        
        # Choose enrichment strategy based on conversation stage
        if i == 0:  # First interaction - use structure only
            enriched_prompt = memory.enrich_structure(f"User says: {user_input}\nRespond appropriately.")
        elif i < 3:  # Middle interactions - use keys for personalization
            enriched_prompt = memory.enrich_keys(f"User says: {user_input}\nRespond based on what you know about the user.")
        else:  # Final interaction - use full context
            enriched_prompt = memory.enrich_full(f"User says: {user_input}\nProvide a thoughtful farewell response.")
        
        print(f"User: {user_input}")
        
        result = Runner.run_sync(agent, enriched_prompt)
        response = result.final_output
        print(f"Assistant: {response}")
        
        # Store interaction in raw memory
        memory.append_raw(f"User: {user_input}\nAssistant: {response}\n---\n")


def demonstrate_enrichment_functions():
    """Demonstrate all enrichment functions with sample data."""
    print("\n=== Enrichment Functions Demonstration ===")
    memory = FileBasedMemory()
    
    # Set up sample memory content
    memory.modify_structure("Sample memory for demonstration: stores user preferences, session data, and conversation history.")
    memory.write_by_key("user_name", "Alice")
    memory.write_by_key("session_id", "demo_123")
    memory.write_by_key("preferences", "likes concise responses, prefers examples")
    memory.append_raw("Demo session started.\nUser asked about memory functions.\nExplained basic concepts.")
    
    base_text = "Please help me understand how memory works in this system."
    
    print("Base text:", base_text)
    print("\n" + "="*60)
    
    print("\n1. enrich_raw() - Adds raw memory content:")
    print("-" * 50)
    enriched = memory.enrich_raw(base_text)
    print(enriched)
    
    print("\n" + "="*60)
    print("\n2. enrich_keys() - Adds key-based memory:")
    print("-" * 50)
    enriched = memory.enrich_keys(base_text)
    print(enriched)
    
    print("\n" + "="*60)
    print("\n3. enrich_structure() - Adds structure definition:")
    print("-" * 50)
    enriched = memory.enrich_structure(base_text)
    print(enriched)
    
    print("\n" + "="*60)
    print("\n4. enrich_full() - Adds all memory information:")
    print("-" * 50)
    enriched = memory.enrich_full(base_text)
    print(enriched)


if __name__ == '__main__':
    print('=== Memory System Examples ===\n')
    
    # # Test just the enrichment functions first (no API calls)
    # print('Testing enrichment functions (no API calls):')
    # demonstrate_enrichment_functions()
    
    
    # Test one example with actual agent
    
    print('1. Simple agent with manual prompt enrichment')
    run_agent_memory_through_prompt()

    print('\n' + '='*60 + '\n')
    print('Note: Other examples are commented out. Uncomment to test:')
    print('- run_agent_memory_through_tools()')
    print('- run_agent_with_enriched_prompts()')  
    print('- run_agent_with_dynamic_enrichment()')
    
    # print('\n' + '='*60 + '\n')
    # print('2. Agent with memory tools')
    # run_agent_memory_through_tools()

    # print('3. NEW: Agent with enrichment functions')
    # run_agent_with_enriched_prompts()
    
    # print('\n' + '='*60 + '\n')
    # print('4. NEW: Dynamic enrichment based on context')
    # run_agent_with_dynamic_enrichment()
    
    print('\n' + '='*60)
    print('Memory system test completed!')
