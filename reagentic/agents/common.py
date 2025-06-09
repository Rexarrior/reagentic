from  ..providers import openrouter

# Create default providers with tracing disabled to prevent OpenAI API key messages
default_free_provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324, disable_tracing=True)
default_paid_provider = openrouter.OpenrouterProvider(openrouter.GPT_4_1_MINI, disable_tracing=True)
default_provider = default_free_provider