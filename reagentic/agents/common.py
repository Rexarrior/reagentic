from  ..providers import openrouter
default_free_provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
default_paid_provider = openrouter.OpenrouterProvider(openrouter.GPT_4_1_MINI)
default_provider=default_free_provider