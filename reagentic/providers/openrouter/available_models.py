from ..common import ModelInfo


YI_LARGE = ModelInfo(
    str_identifier="01-ai/yi-large",
    price_in=3e-06,
    price_out=3e-06,
    creator="01-ai",
    description='''The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service.

It stands out for its multilingual proficiency, particularly in Spanish, Chinese, Japanese, German, and French.

Check out the [launch announcement](https://01-ai.github.io/blog/01.ai-yi-large-llm-launch) to learn more.,
    created=1719273600'''
)

MN_STARCANNON_12B = ModelInfo(
    str_identifier="aetherwiing/mn-starcannon-12b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="aetherwiing",
    description='''Starcannon 12B v2 is a creative roleplay and story writing model, based on Mistral Nemo, using [nothingiisreal/mn-celeste-12b](/nothingiisreal/mn-celeste-12b) as a base, with [intervitens/mini-magnum-12b-v1.1](https://huggingface.co/intervitens/mini-magnum-12b-v1.1) merged in using the [TIES](https://arxiv.org/abs/2306.01708) method.

Although more similar to Magnum overall, the model remains very creative, with a pleasant writing style. It is recommended for people wanting more variety than Magnum, and yet more verbose prose than Celeste.,
    created=1723507200'''
)

DEEPCODER_14B_PREVIEW_FREE = ModelInfo(
    str_identifier="agentica-org/deepcoder-14b-preview:free",
    price_in=0.0,
    price_out=0.0,
    creator="agentica-org",
    description='''DeepCoder-14B-Preview is a 14B parameter code generation model fine-tuned from DeepSeek-R1-Distill-Qwen-14B using reinforcement learning with GRPO+ and iterative context lengthening. It is optimized for long-context program synthesis and achieves strong performance across coding benchmarks, including 60.6% on LiveCodeBench v5, competitive with models like o3-Mini,
    created=1744555395'''
)

JAMBA_1_6_LARGE = ModelInfo(
    str_identifier="ai21/jamba-1.6-large",
    price_in=2e-06,
    price_out=8e-06,
    creator="ai21",
    description='''AI21 Jamba Large 1.6 is a high-performance hybrid foundation model combining State Space Models (Mamba) with Transformer attention mechanisms. Developed by AI21, it excels in extremely long-context handling (256K tokens), demonstrates superior inference efficiency (up to 2.5x faster than comparable models), and supports structured JSON output and tool-use capabilities. It has 94 billion active parameters (398 billion total), optimized quantization support (ExpertsInt8), and multilingual proficiency in languages such as English, Spanish, French, Portuguese, Italian, Dutch, German, Arabic, and Hebrew.

Usage of this model is subject to the [Jamba Open Model License](https://www.ai21.com/licenses/jamba-open-model-license).,
    created=1741905173'''
)

JAMBA_1_6_MINI = ModelInfo(
    str_identifier="ai21/jamba-1.6-mini",
    price_in=2e-07,
    price_out=4e-07,
    creator="ai21",
    description='''AI21 Jamba Mini 1.6 is a hybrid foundation model combining State Space Models (Mamba) with Transformer attention mechanisms. With 12 billion active parameters (52 billion total), this model excels in extremely long-context tasks (up to 256K tokens) and achieves superior inference efficiency, outperforming comparable open models on tasks such as retrieval-augmented generation (RAG) and grounded question answering. Jamba Mini 1.6 supports multilingual tasks across English, Spanish, French, Portuguese, Italian, Dutch, German, Arabic, and Hebrew, along with structured JSON output and tool-use capabilities.

Usage of this model is subject to the [Jamba Open Model License](https://www.ai21.com/licenses/jamba-open-model-license).,
    created=1741905171'''
)

AION_1_0 = ModelInfo(
    str_identifier="aion-labs/aion-1.0",
    price_in=4e-06,
    price_out=8e-06,
    creator="aion-labs",
    description='''Aion-1.0 is a multi-model system designed for high performance across various tasks, including reasoning and coding. It is built on DeepSeek-R1, augmented with additional models and techniques such as Tree of Thoughts (ToT) and Mixture of Experts (MoE). It is Aion Lab's most powerful reasoning model.,
    created=1738697557'''
)

AION_1_0_MINI = ModelInfo(
    str_identifier="aion-labs/aion-1.0-mini",
    price_in=7e-07,
    price_out=1.4e-06,
    creator="aion-labs",
    description='''Aion-1.0-Mini 32B parameter model is a distilled version of the DeepSeek-R1 model, designed for strong performance in reasoning domains such as mathematics, coding, and logic. It is a modified variant of a FuseAI model that outperforms R1-Distill-Qwen-32B and R1-Distill-Llama-70B, with benchmark results available on its [Hugging Face page](https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview), independently replicated for verification.,
    created=1738697107'''
)

AION_RP_LLAMA_3_1_8B = ModelInfo(
    str_identifier="aion-labs/aion-rp-llama-3.1-8b",
    price_in=2e-07,
    price_out=2e-07,
    creator="aion-labs",
    description='''Aion-RP-Llama-3.1-8B ranks the highest in the character evaluation portion of the RPBench-Auto benchmark, a roleplaying-specific variant of Arena-Hard-Auto, where LLMs evaluate each other’s responses. It is a fine-tuned base model rather than an instruct model, designed to produce more natural and varied writing.,
    created=1738696718'''
)

CODELLAMA_7B_INSTRUCT_SOLIDITY = ModelInfo(
    str_identifier="alfredpros/codellama-7b-instruct-solidity",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="alfredpros",
    description='''A finetuned 7 billion parameters Code LLaMA - Instruct model to generate Solidity smart contract using 4-bit QLoRA finetuning provided by PEFT library.,
    created=1744641874'''
)

OPENHANDS_LM_32B_V0_1 = ModelInfo(
    str_identifier="all-hands/openhands-lm-32b-v0.1",
    price_in=2.6e-06,
    price_out=3.4e-06,
    creator="all-hands",
    description='''OpenHands LM v0.1 is a 32B open-source coding model fine-tuned from Qwen2.5-Coder-32B-Instruct using reinforcement learning techniques outlined in SWE-Gym. It is optimized for autonomous software development agents and achieves strong performance on SWE-Bench Verified, with a 37.2% resolve rate. The model supports a 128K token context window, making it well-suited for long-horizon code reasoning and large codebase tasks.

OpenHands LM is designed for local deployment and runs on consumer-grade GPUs such as a single 3090. It enables fully offline agent workflows without dependency on proprietary APIs. This release is intended as a research preview, and future updates aim to improve generalizability, reduce repetition, and offer smaller variants.,
    created=1743613013'''
)

GOLIATH_120B = ModelInfo(
    str_identifier="alpindale/goliath-120b",
    price_in=1e-05,
    price_out=1.25e-05,
    creator="alpindale",
    description='''A large LLM created by combining two fine-tuned Llama 70B models into one 120B model. Combines Xwin and Euryale.

Credits to
- [@chargoddard](https://huggingface.co/chargoddard) for developing the framework used to merge the model - [mergekit](https://github.com/cg123/mergekit).
- [@Undi95](https://huggingface.co/Undi95) for helping with the merge ratios.

#merge,
    created=1699574400'''
)

MAGNUM_72B = ModelInfo(
    str_identifier="alpindale/magnum-72b",
    price_in=4e-06,
    price_out=6e-06,
    creator="alpindale",
    description='''From the maker of [Goliath](https://openrouter.ai/models/alpindale/goliath-120b), Magnum 72B is the first in a new family of models designed to achieve the prose quality of the Claude 3 models, notably Opus & Sonnet.

The model is based on [Qwen2 72B](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) and trained with 55 million tokens of highly curated roleplay (RP) data.,
    created=1720656000'''
)

NOVA_LITE_V1 = ModelInfo(
    str_identifier="amazon/nova-lite-v1",
    price_in=6e-08,
    price_out=2.4e-07,
    creator="amazon",
    description='''Amazon Nova Lite 1.0 is a very low-cost multimodal model from Amazon that focused on fast processing of image, video, and text inputs to generate text output. Amazon Nova Lite can handle real-time customer interactions, document analysis, and visual question-answering tasks with high accuracy.

With an input context of 300K tokens, it can analyze multiple images or up to 30 minutes of video in a single input.,
    created=1733437363'''
)

NOVA_MICRO_V1 = ModelInfo(
    str_identifier="amazon/nova-micro-v1",
    price_in=3.5e-08,
    price_out=1.4e-07,
    creator="amazon",
    description='''Amazon Nova Micro 1.0 is a text-only model that delivers the lowest latency responses in the Amazon Nova family of models at a very low cost. With a context length of 128K tokens and optimized for speed and cost, Amazon Nova Micro excels at tasks such as text summarization, translation, content classification, interactive chat, and brainstorming. It has  simple mathematical reasoning and coding abilities.,
    created=1733437237'''
)

NOVA_PRO_V1 = ModelInfo(
    str_identifier="amazon/nova-pro-v1",
    price_in=8e-07,
    price_out=3.2e-06,
    creator="amazon",
    description='''Amazon Nova Pro 1.0 is a capable multimodal model from Amazon focused on providing a combination of accuracy, speed, and cost for a wide range of tasks. As of December 2024, it achieves state-of-the-art performance on key benchmarks including visual question answering (TextVQA) and video understanding (VATEX).

Amazon Nova Pro demonstrates strong capabilities in processing both visual and textual information and at analyzing financial documents.

**NOTE**: Video input is not supported at this time.,
    created=1733436303'''
)

MAGNUM_V2_72B = ModelInfo(
    str_identifier="anthracite-org/magnum-v2-72b",
    price_in=3e-06,
    price_out=3e-06,
    creator="anthracite-org",
    description='''From the maker of [Goliath](https://openrouter.ai/models/alpindale/goliath-120b), Magnum 72B is the seventh in a family of models designed to achieve the prose quality of the Claude 3 models, notably Opus & Sonnet.

The model is based on [Qwen2 72B](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) and trained with 55 million tokens of highly curated roleplay (RP) data.,
    created=1727654400'''
)

MAGNUM_V4_72B = ModelInfo(
    str_identifier="anthracite-org/magnum-v4-72b",
    price_in=2.5e-06,
    price_out=3e-06,
    creator="anthracite-org",
    description='''This is a series of models designed to replicate the prose quality of the Claude 3 models, specifically Sonnet(https://openrouter.ai/anthropic/claude-3.5-sonnet) and Opus(https://openrouter.ai/anthropic/claude-3-opus).

The model is fine-tuned on top of [Qwen2.5 72B](https://openrouter.ai/qwen/qwen-2.5-72b-instruct).,
    created=1729555200'''
)

CLAUDE_2 = ModelInfo(
    str_identifier="anthropic/claude-2",
    price_in=8e-06,
    price_out=2.4e-05,
    creator="anthropic",
    description='''Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use.,
    created=1700611200'''
)

CLAUDE_2_0 = ModelInfo(
    str_identifier="anthropic/claude-2.0",
    price_in=8e-06,
    price_out=2.4e-05,
    creator="anthropic",
    description='''Anthropic's flagship model. Superior performance on tasks that require complex reasoning. Supports hundreds of pages of text.,
    created=1690502400'''
)

CLAUDE_2_0_BETA = ModelInfo(
    str_identifier="anthropic/claude-2.0:beta",
    price_in=8e-06,
    price_out=2.4e-05,
    creator="anthropic",
    description='''Anthropic's flagship model. Superior performance on tasks that require complex reasoning. Supports hundreds of pages of text.,
    created=1690502400'''
)

CLAUDE_2_1 = ModelInfo(
    str_identifier="anthropic/claude-2.1",
    price_in=8e-06,
    price_out=2.4e-05,
    creator="anthropic",
    description='''Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use.,
    created=1700611200'''
)

CLAUDE_2_1_BETA = ModelInfo(
    str_identifier="anthropic/claude-2.1:beta",
    price_in=8e-06,
    price_out=2.4e-05,
    creator="anthropic",
    description='''Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use.,
    created=1700611200'''
)

CLAUDE_2_BETA = ModelInfo(
    str_identifier="anthropic/claude-2:beta",
    price_in=8e-06,
    price_out=2.4e-05,
    creator="anthropic",
    description='''Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use.,
    created=1700611200'''
)

CLAUDE_3_HAIKU = ModelInfo(
    str_identifier="anthropic/claude-3-haiku",
    price_in=2.5e-07,
    price_out=1.25e-06,
    creator="anthropic",
    description='''Claude 3 Haiku is Anthropic's fastest and most compact model for
near-instant responsiveness. Quick and accurate targeted performance.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-haiku)

#multimodal,
    created=1710288000'''
)

CLAUDE_3_HAIKU_BETA = ModelInfo(
    str_identifier="anthropic/claude-3-haiku:beta",
    price_in=2.5e-07,
    price_out=1.25e-06,
    creator="anthropic",
    description='''Claude 3 Haiku is Anthropic's fastest and most compact model for
near-instant responsiveness. Quick and accurate targeted performance.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-haiku)

#multimodal,
    created=1710288000'''
)

CLAUDE_3_OPUS = ModelInfo(
    str_identifier="anthropic/claude-3-opus",
    price_in=1.5e-05,
    price_out=7.5e-05,
    creator="anthropic",
    description='''Claude 3 Opus is Anthropic's most powerful model for highly complex tasks. It boasts top-level performance, intelligence, fluency, and understanding.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)

#multimodal,
    created=1709596800'''
)

CLAUDE_3_OPUS_BETA = ModelInfo(
    str_identifier="anthropic/claude-3-opus:beta",
    price_in=1.5e-05,
    price_out=7.5e-05,
    creator="anthropic",
    description='''Claude 3 Opus is Anthropic's most powerful model for highly complex tasks. It boasts top-level performance, intelligence, fluency, and understanding.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)

#multimodal,
    created=1709596800'''
)

CLAUDE_3_SONNET = ModelInfo(
    str_identifier="anthropic/claude-3-sonnet",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3 Sonnet is an ideal balance of intelligence and speed for enterprise workloads. Maximum utility at a lower price, dependable, balanced for scaled deployments.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)

#multimodal,
    created=1709596800'''
)

CLAUDE_3_SONNET_BETA = ModelInfo(
    str_identifier="anthropic/claude-3-sonnet:beta",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3 Sonnet is an ideal balance of intelligence and speed for enterprise workloads. Maximum utility at a lower price, dependable, balanced for scaled deployments.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)

#multimodal,
    created=1709596800'''
)

CLAUDE_3_5_HAIKU = ModelInfo(
    str_identifier="anthropic/claude-3.5-haiku",
    price_in=8e-07,
    price_out=4e-06,
    creator="anthropic",
    description='''Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy, and tool use. Engineered to excel in real-time applications, it delivers quick response times that are essential for dynamic tasks such as chat interactions and immediate coding suggestions.

This makes it highly suitable for environments that demand both speed and precision, such as software development, customer service bots, and data management systems.

This model is currently pointing to [Claude 3.5 Haiku (2024-10-22)](/anthropic/claude-3-5-haiku-20241022).,
    created=1730678400'''
)

CLAUDE_3_5_HAIKU_20241022 = ModelInfo(
    str_identifier="anthropic/claude-3.5-haiku-20241022",
    price_in=8e-07,
    price_out=4e-06,
    creator="anthropic",
    description='''Claude 3.5 Haiku features enhancements across all skill sets including coding, tool use, and reasoning. As the fastest model in the Anthropic lineup, it offers rapid response times suitable for applications that require high interactivity and low latency, such as user-facing chatbots and on-the-fly code completions. It also excels in specialized tasks like data extraction and real-time content moderation, making it a versatile tool for a broad range of industries.

It does not support image inputs.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/3-5-models-and-computer-use),
    created=1730678400'''
)

CLAUDE_3_5_HAIKU_20241022_BETA = ModelInfo(
    str_identifier="anthropic/claude-3.5-haiku-20241022:beta",
    price_in=8e-07,
    price_out=4e-06,
    creator="anthropic",
    description='''Claude 3.5 Haiku features enhancements across all skill sets including coding, tool use, and reasoning. As the fastest model in the Anthropic lineup, it offers rapid response times suitable for applications that require high interactivity and low latency, such as user-facing chatbots and on-the-fly code completions. It also excels in specialized tasks like data extraction and real-time content moderation, making it a versatile tool for a broad range of industries.

It does not support image inputs.

See the launch announcement and benchmark results [here](https://www.anthropic.com/news/3-5-models-and-computer-use),
    created=1730678400'''
)

CLAUDE_3_5_HAIKU_BETA = ModelInfo(
    str_identifier="anthropic/claude-3.5-haiku:beta",
    price_in=8e-07,
    price_out=4e-06,
    creator="anthropic",
    description='''Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy, and tool use. Engineered to excel in real-time applications, it delivers quick response times that are essential for dynamic tasks such as chat interactions and immediate coding suggestions.

This makes it highly suitable for environments that demand both speed and precision, such as software development, customer service bots, and data management systems.

This model is currently pointing to [Claude 3.5 Haiku (2024-10-22)](/anthropic/claude-3-5-haiku-20241022).,
    created=1730678400'''
)

CLAUDE_3_5_SONNET = ModelInfo(
    str_identifier="anthropic/claude-3.5-sonnet",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:

- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding
- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights
- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone
- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)

#multimodal,
    created=1729555200'''
)

CLAUDE_3_5_SONNET_20240620 = ModelInfo(
    str_identifier="anthropic/claude-3.5-sonnet-20240620",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:

- Coding: Autonomously writes, edits, and runs code with reasoning and troubleshooting
- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights
- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone
- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)

For the latest version (2024-10-23), check out [Claude 3.5 Sonnet](/anthropic/claude-3.5-sonnet).

#multimodal,
    created=1718841600'''
)

CLAUDE_3_5_SONNET_20240620_BETA = ModelInfo(
    str_identifier="anthropic/claude-3.5-sonnet-20240620:beta",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:

- Coding: Autonomously writes, edits, and runs code with reasoning and troubleshooting
- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights
- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone
- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)

For the latest version (2024-10-23), check out [Claude 3.5 Sonnet](/anthropic/claude-3.5-sonnet).

#multimodal,
    created=1718841600'''
)

CLAUDE_3_5_SONNET_BETA = ModelInfo(
    str_identifier="anthropic/claude-3.5-sonnet:beta",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:

- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding
- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights
- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone
- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)

#multimodal,
    created=1729555200'''
)

CLAUDE_3_7_SONNET = ModelInfo(
    str_identifier="anthropic/claude-3.7-sonnet",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. 

Claude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.

Read more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet),
    created=1740422110'''
)

CLAUDE_3_7_SONNET_BETA = ModelInfo(
    str_identifier="anthropic/claude-3.7-sonnet:beta",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. 

Claude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.

Read more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet),
    created=1740422110'''
)

CLAUDE_3_7_SONNET_THINKING = ModelInfo(
    str_identifier="anthropic/claude-3.7-sonnet:thinking",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. 

Claude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.

Read more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet),
    created=1740422110'''
)

CLAUDE_OPUS_4 = ModelInfo(
    str_identifier="anthropic/claude-opus-4",
    price_in=1.5e-05,
    price_out=7.5e-05,
    creator="anthropic",
    description='''Claude Opus 4 is benchmarked as the world’s best coding model, at time of release, bringing sustained performance on complex, long-running tasks and agent workflows. It sets new benchmarks in software engineering, achieving leading results on SWE-bench (72.5%) and Terminal-bench (43.2%). Opus 4 supports extended, agentic workflows, handling thousands of task steps continuously for hours without degradation. 

Read more at the [blog post here](https://www.anthropic.com/news/claude-4),
    created=1747931245'''
)

CLAUDE_SONNET_4 = ModelInfo(
    str_identifier="anthropic/claude-sonnet-4",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude Sonnet 4 significantly enhances the capabilities of its predecessor, Sonnet 3.7, excelling in both coding and reasoning tasks with improved precision and controllability. Achieving state-of-the-art performance on SWE-bench (72.7%), Sonnet 4 balances capability and computational efficiency, making it suitable for a broad range of applications from routine coding tasks to complex software development projects. Key enhancements include improved autonomous codebase navigation, reduced error rates in agent-driven workflows, and increased reliability in following intricate instructions. Sonnet 4 is optimized for practical everyday use, providing advanced reasoning capabilities while maintaining efficiency and responsiveness in diverse internal and external scenarios.

Read more at the [blog post here](https://www.anthropic.com/news/claude-4),
    created=1747930371'''
)

ARCEE_BLITZ = ModelInfo(
    str_identifier="arcee-ai/arcee-blitz",
    price_in=4.5e-07,
    price_out=7.5e-07,
    creator="arcee-ai",
    description='''Arcee Blitz is a 24 B‑parameter dense model distilled from DeepSeek and built on Mistral architecture for "everyday" chat. The distillation‑plus‑refinement pipeline trims compute while keeping DeepSeek‑style reasoning, so Blitz punches above its weight on MMLU, GSM‑8K and BBH compared with other mid‑size open models. With a default 128 k context window and competitive throughput, it serves as a cost‑efficient workhorse for summarization, brainstorming and light code help. Internally, Arcee uses Blitz as the default writer in Conductor pipelines when the heavier Virtuoso line is not required. Users therefore get near‑70 B quality at ~⅓ the latency and price. ,
    created=1746470100'''
)

CALLER_LARGE = ModelInfo(
    str_identifier="arcee-ai/caller-large",
    price_in=5.5e-07,
    price_out=8.5e-07,
    creator="arcee-ai",
    description='''Caller Large is Arcee's specialist "function‑calling" SLM built to orchestrate external tools and APIs. Instead of maximizing next‑token accuracy, training focuses on structured JSON outputs, parameter extraction and multi‑step tool chains, making Caller a natural choice for retrieval‑augmented generation, robotic process automation or data‑pull chatbots. It incorporates a routing head that decides when (and how) to invoke a tool versus answering directly, reducing hallucinated calls. The model is already the backbone of Arcee Conductor's auto‑tool mode, where it parses user intent, emits clean function signatures and hands control back once the tool response is ready. Developers thus gain an OpenAI‑style function‑calling UX without handing requests to a frontier‑scale model. ,
    created=1746487869'''
)

CODER_LARGE = ModelInfo(
    str_identifier="arcee-ai/coder-large",
    price_in=5e-07,
    price_out=8e-07,
    creator="arcee-ai",
    description='''Coder‑Large is a 32 B‑parameter offspring of Qwen 2.5‑Instruct that has been further trained on permissively‑licensed GitHub, CodeSearchNet and synthetic bug‑fix corpora. It supports a 32k context window, enabling multi‑file refactoring or long diff review in a single call, and understands 30‑plus programming languages with special attention to TypeScript, Go and Terraform. Internal benchmarks show 5–8 pt gains over CodeLlama‑34 B‑Python on HumanEval and competitive BugFix scores thanks to a reinforcement pass that rewards compilable output. The model emits structured explanations alongside code blocks by default, making it suitable for educational tooling as well as production copilot scenarios. Cost‑wise, Together AI prices it well below proprietary incumbents, so teams can scale interactive coding without runaway spend. ,
    created=1746478663'''
)

MAESTRO_REASONING = ModelInfo(
    str_identifier="arcee-ai/maestro-reasoning",
    price_in=9e-07,
    price_out=3.3e-06,
    creator="arcee-ai",
    description='''Maestro Reasoning is Arcee's flagship analysis model: a 32 B‑parameter derivative of Qwen 2.5‑32 B tuned with DPO and chain‑of‑thought RL for step‑by‑step logic. Compared to the earlier 7 B preview, the production 32 B release widens the context window to 128 k tokens and doubles pass‑rate on MATH and GSM‑8K, while also lifting code completion accuracy. Its instruction style encourages structured "thought → answer" traces that can be parsed or hidden according to user preference. That transparency pairs well with audit‑focused industries like finance or healthcare where seeing the reasoning path matters. In Arcee Conductor, Maestro is automatically selected for complex, multi‑constraint queries that smaller SLMs bounce. ,
    created=1746481269'''
)

SPOTLIGHT = ModelInfo(
    str_identifier="arcee-ai/spotlight",
    price_in=1.8e-07,
    price_out=1.8e-07,
    creator="arcee-ai",
    description='''Spotlight is a 7‑billion‑parameter vision‑language model derived from Qwen 2.5‑VL and fine‑tuned by Arcee AI for tight image‑text grounding tasks. It offers a 32 k‑token context window, enabling rich multimodal conversations that combine lengthy documents with one or more images. Training emphasized fast inference on consumer GPUs while retaining strong captioning, visual‐question‑answering, and diagram‑analysis accuracy. As a result, Spotlight slots neatly into agent workflows where screenshots, charts or UI mock‑ups need to be interpreted on the fly. Early benchmarks show it matching or out‑scoring larger VLMs such as LLaVA‑1.6 13 B on popular VQA and POPE alignment tests. ,
    created=1746481552'''
)

VIRTUOSO_LARGE = ModelInfo(
    str_identifier="arcee-ai/virtuoso-large",
    price_in=7.5e-07,
    price_out=1.2e-06,
    creator="arcee-ai",
    description='''Virtuoso‑Large is Arcee's top‑tier general‑purpose LLM at 72 B parameters, tuned to tackle cross‑domain reasoning, creative writing and enterprise QA. Unlike many 70 B peers, it retains the 128 k context inherited from Qwen 2.5, letting it ingest books, codebases or financial filings wholesale. Training blended DeepSeek R1 distillation, multi‑epoch supervised fine‑tuning and a final DPO/RLHF alignment stage, yielding strong performance on BIG‑Bench‑Hard, GSM‑8K and long‑context Needle‑In‑Haystack tests. Enterprises use Virtuoso‑Large as the "fallback" brain in Conductor pipelines when other SLMs flag low confidence. Despite its size, aggressive KV‑cache optimizations keep first‑token latency in the low‑second range on 8× H100 nodes, making it a practical production‑grade powerhouse.,
    created=1746478885'''
)

VIRTUOSO_MEDIUM_V2 = ModelInfo(
    str_identifier="arcee-ai/virtuoso-medium-v2",
    price_in=5e-07,
    price_out=8e-07,
    creator="arcee-ai",
    description='''Virtuoso‑Medium‑v2 is a 32 B model distilled from DeepSeek‑v3 logits and merged back onto a Qwen 2.5 backbone, yielding a sharper, more factual successor to the original Virtuoso Medium. The team harvested ~1.1 B logit tokens and applied "fusion‑merging" plus DPO alignment, which pushed scores past Arcee‑Nova 2024 and many 40 B‑plus peers on MMLU‑Pro, MATH and HumanEval. With a 128 k context and aggressive quantization options (from BF16 down to 4‑bit GGUF), it balances capability with deployability on single‑GPU nodes. Typical use cases include enterprise chat assistants, technical writing aids and medium‑complexity code drafting where Virtuoso‑Large would be overkill. ,
    created=1746478434'''
)

QWQ_32B_ARLIAI_RPR_V1_FREE = ModelInfo(
    str_identifier="arliai/qwq-32b-arliai-rpr-v1:free",
    price_in=0.0,
    price_out=0.0,
    creator="arliai",
    description='''QwQ-32B-ArliAI-RpR-v1 is a 32B parameter model fine-tuned from Qwen/QwQ-32B using a curated creative writing and roleplay dataset originally developed for the RPMax series. It is designed to maintain coherence and reasoning across long multi-turn conversations by introducing explicit reasoning steps per dialogue turn, generated and refined using the base model itself.

The model was trained using RS-QLORA+ on 8K sequence lengths and supports up to 128K context windows (with practical performance around 32K). It is optimized for creative roleplay and dialogue generation, with an emphasis on minimizing cross-context repetition while preserving stylistic diversity.,
    created=1744555982'''
)

DOLPHIN_MIXTRAL_8X22B = ModelInfo(
    str_identifier="cognitivecomputations/dolphin-mixtral-8x22b",
    price_in=9e-07,
    price_out=9e-07,
    creator="cognitivecomputations",
    description='''Dolphin 2.9 is designed for instruction following, conversational, and coding. This model is a finetune of [Mixtral 8x22B Instruct](/models/mistralai/mixtral-8x22b-instruct). It features a 64k context length and was fine-tuned with a 16k sequence length using ChatML templates.

This model is a successor to [Dolphin Mixtral 8x7B](/models/cognitivecomputations/dolphin-mixtral-8x7b).

The model is uncensored and is stripped of alignment and bias. It requires an external alignment layer for ethical use. Users are cautioned to use this highly compliant model responsibly, as detailed in a blog post about uncensored models at [erichartford.com/uncensored-models](https://erichartford.com/uncensored-models).

#moe #uncensored,
    created=1717804800'''
)

DOLPHIN3_0_MISTRAL_24B_FREE = ModelInfo(
    str_identifier="cognitivecomputations/dolphin3.0-mistral-24b:free",
    price_in=0.0,
    price_out=0.0,
    creator="cognitivecomputations",
    description='''Dolphin 3.0 is the next generation of the Dolphin series of instruct-tuned models.  Designed to be the ultimate general purpose local model, enabling coding, math, agentic, function calling, and general use cases.

Dolphin aims to be a general purpose instruct model, similar to the models behind ChatGPT, Claude, Gemini. 

Part of the [Dolphin 3.0 Collection](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3) Curated and trained by [Eric Hartford](https://huggingface.co/ehartford), [Ben Gitter](https://huggingface.co/bigstorm), [BlouseJury](https://huggingface.co/BlouseJury) and [Cognitive Computations](https://huggingface.co/cognitivecomputations),
    created=1739462019'''
)

DOLPHIN3_0_R1_MISTRAL_24B_FREE = ModelInfo(
    str_identifier="cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    price_in=0.0,
    price_out=0.0,
    creator="cognitivecomputations",
    description='''Dolphin 3.0 R1 is the next generation of the Dolphin series of instruct-tuned models.  Designed to be the ultimate general purpose local model, enabling coding, math, agentic, function calling, and general use cases.

The R1 version has been trained for 3 epochs to reason using 800k reasoning traces from the Dolphin-R1 dataset.

Dolphin aims to be a general purpose reasoning instruct model, similar to the models behind ChatGPT, Claude, Gemini.

Part of the [Dolphin 3.0 Collection](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3) Curated and trained by [Eric Hartford](https://huggingface.co/ehartford), [Ben Gitter](https://huggingface.co/bigstorm), [BlouseJury](https://huggingface.co/BlouseJury) and [Cognitive Computations](https://huggingface.co/cognitivecomputations),
    created=1739462498'''
)

COMMAND = ModelInfo(
    str_identifier="cohere/command",
    price_in=1e-06,
    price_out=2e-06,
    creator="cohere",
    description='''Command is an instruction-following conversational model that performs language tasks with high quality, more reliably and with a longer context than our base generative models.

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1710374400'''
)

COMMAND_A = ModelInfo(
    str_identifier="cohere/command-a",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="cohere",
    description='''Command A is an open-weights 111B parameter model with a 256k context window focused on delivering great performance across agentic, multilingual, and coding use cases.
Compared to other leading proprietary and open-weights models Command A delivers maximum performance with minimum hardware costs, excelling on business-critical agentic and multilingual tasks.,
    created=1741894342'''
)

COMMAND_R = ModelInfo(
    str_identifier="cohere/command-r",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="cohere",
    description='''Command-R is a 35B parameter model that performs conversational language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex workflows like code generation, retrieval augmented generation (RAG), tool use, and agents.

Read the launch post [here](https://txt.cohere.com/command-r/).

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1710374400'''
)

COMMAND_R_03_2024 = ModelInfo(
    str_identifier="cohere/command-r-03-2024",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="cohere",
    description='''Command-R is a 35B parameter model that performs conversational language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex workflows like code generation, retrieval augmented generation (RAG), tool use, and agents.

Read the launch post [here](https://txt.cohere.com/command-r/).

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1709341200'''
)

COMMAND_R_08_2024 = ModelInfo(
    str_identifier="cohere/command-r-08-2024",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="cohere",
    description='''command-r-08-2024 is an update of the [Command R](/models/cohere/command-r) with improved performance for multilingual retrieval-augmented generation (RAG) and tool use. More broadly, it is better at math, code and reasoning and is competitive with the previous version of the larger Command R+ model.

Read the launch post [here](https://docs.cohere.com/changelog/command-gets-refreshed).

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1724976000'''
)

COMMAND_R_PLUS = ModelInfo(
    str_identifier="cohere/command-r-plus",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="cohere",
    description='''Command R+ is a new, 104B-parameter LLM from Cohere. It's useful for roleplay, general consumer usecases, and Retrieval Augmented Generation (RAG).

It offers multilingual support for ten key languages to facilitate global business operations. See benchmarks and the launch post [here](https://txt.cohere.com/command-r-plus-microsoft-azure/).

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1712188800'''
)

COMMAND_R_PLUS_04_2024 = ModelInfo(
    str_identifier="cohere/command-r-plus-04-2024",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="cohere",
    description='''Command R+ is a new, 104B-parameter LLM from Cohere. It's useful for roleplay, general consumer usecases, and Retrieval Augmented Generation (RAG).

It offers multilingual support for ten key languages to facilitate global business operations. See benchmarks and the launch post [here](https://txt.cohere.com/command-r-plus-microsoft-azure/).

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1712016000'''
)

COMMAND_R_PLUS_08_2024 = ModelInfo(
    str_identifier="cohere/command-r-plus-08-2024",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="cohere",
    description='''command-r-plus-08-2024 is an update of the [Command R+](/models/cohere/command-r-plus) with roughly 50% higher throughput and 25% lower latencies as compared to the previous Command R+ version, while keeping the hardware footprint the same.

Read the launch post [here](https://docs.cohere.com/changelog/command-gets-refreshed).

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1724976000'''
)

COMMAND_R7B_12_2024 = ModelInfo(
    str_identifier="cohere/command-r7b-12-2024",
    price_in=3.75e-08,
    price_out=1.5e-07,
    creator="cohere",
    description='''Command R7B (12-2024) is a small, fast update of the Command R+ model, delivered in December 2024. It excels at RAG, tool use, agents, and similar tasks requiring complex reasoning and multiple steps.

Use of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement).,
    created=1734158152'''
)

DEEPSEEK_CHAT = ModelInfo(
    str_identifier="deepseek/deepseek-chat",
    price_in=3.8e-07,
    price_out=8.9e-07,
    creator="deepseek",
    description='''DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported evaluations reveal that the model outperforms other open-source models and rivals leading closed-source models.

For model details, please visit [the DeepSeek-V3 repo](https://github.com/deepseek-ai/DeepSeek-V3) for more information, or see the [launch announcement](https://api-docs.deepseek.com/news/news1226).,
    created=1735241320'''
)

DEEPSEEK_CHAT_V3_0324 = ModelInfo(
    str_identifier="deepseek/deepseek-chat-v3-0324",
    price_in=3e-07,
    price_out=8.8e-07,
    creator="deepseek",
    description='''DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.

It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) model and performs really well on a variety of tasks.,
    created=1742824755'''
)

DEEPSEEK_CHAT_V3_0324_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-chat-v3-0324:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.

It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) model and performs really well on a variety of tasks.,
    created=1742824755'''
)

DEEPSEEK_CHAT_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-chat:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported evaluations reveal that the model outperforms other open-source models and rivals leading closed-source models.

For model details, please visit [the DeepSeek-V3 repo](https://github.com/deepseek-ai/DeepSeek-V3) for more information, or see the [launch announcement](https://api-docs.deepseek.com/news/news1226).,
    created=1735241320'''
)

DEEPSEEK_PROVER_V2 = ModelInfo(
    str_identifier="deepseek/deepseek-prover-v2",
    price_in=5e-07,
    price_out=2.18e-06,
    creator="deepseek",
    description='''DeepSeek Prover V2 is a 671B parameter model, speculated to be geared towards logic and mathematics. Likely an upgrade from [DeepSeek-Prover-V1.5](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-RL) Not much is known about the model yet, as DeepSeek released it on Hugging Face without an announcement or description.,
    created=1746013094'''
)

DEEPSEEK_PROVER_V2_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-prover-v2:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek Prover V2 is a 671B parameter model, speculated to be geared towards logic and mathematics. Likely an upgrade from [DeepSeek-Prover-V1.5](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-RL) Not much is known about the model yet, as DeepSeek released it on Hugging Face without an announcement or description.,
    created=1746013094'''
)

DEEPSEEK_R1 = ModelInfo(
    str_identifier="deepseek/deepseek-r1",
    price_in=4.5e-07,
    price_out=2.15e-06,
    creator="deepseek",
    description='''DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.

Fully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).

MIT licensed: Distill & commercialize freely!,
    created=1737381095'''
)

DEEPSEEK_R1_0528 = ModelInfo(
    str_identifier="deepseek/deepseek-r1-0528",
    price_in=5e-07,
    price_out=2.15e-06,
    creator="deepseek",
    description='''May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.

Fully open-source model.,
    created=1748455170'''
)

DEEPSEEK_R1_0528_QWEN3_8B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-0528-qwen3-8b",
    price_in=5e-08,
    price_out=1e-07,
    creator="deepseek",
    description='''DeepSeek-R1-0528 is a lightly upgraded release of DeepSeek R1 that taps more compute and smarter post-training tricks, pushing its reasoning and inference to the brink of flagship models like O3 and Gemini 2.5 Pro.
It now tops math, programming, and logic leaderboards, showcasing a step-change in depth-of-thought.
The distilled variant, DeepSeek-R1-0528-Qwen3-8B, transfers this chain-of-thought into an 8 B-parameter form, beating standard Qwen3 8B by +10 pp and tying the 235 B “thinking” giant on AIME 2024.,
    created=1748538543'''
)

DEEPSEEK_R1_0528_QWEN3_8B_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1-0528-qwen3-8b:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek-R1-0528 is a lightly upgraded release of DeepSeek R1 that taps more compute and smarter post-training tricks, pushing its reasoning and inference to the brink of flagship models like O3 and Gemini 2.5 Pro.
It now tops math, programming, and logic leaderboards, showcasing a step-change in depth-of-thought.
The distilled variant, DeepSeek-R1-0528-Qwen3-8B, transfers this chain-of-thought into an 8 B-parameter form, beating standard Qwen3 8B by +10 pp and tying the 235 B “thinking” giant on AIME 2024.,
    created=1748538543'''
)

DEEPSEEK_R1_0528_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1-0528:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.

Fully open-source model.,
    created=1748455170'''
)

DEEPSEEK_R1_DISTILL_LLAMA_70B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-llama-70b",
    price_in=1e-07,
    price_out=4e-07,
    creator="deepseek",
    description='''DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3.3-70b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:

- AIME 2024 pass@1: 70.0
- MATH-500 pass@1: 94.5
- CodeForces Rating: 1633

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1737663169'''
)

DEEPSEEK_R1_DISTILL_LLAMA_70B_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-llama-70b:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3.3-70b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:

- AIME 2024 pass@1: 70.0
- MATH-500 pass@1: 94.5
- CodeForces Rating: 1633

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1737663169'''
)

DEEPSEEK_R1_DISTILL_LLAMA_8B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-llama-8b",
    price_in=4e-08,
    price_out=4e-08,
    creator="deepseek",
    description='''DeepSeek R1 Distill Llama 8B is a distilled large language model based on [Llama-3.1-8B-Instruct](/meta-llama/llama-3.1-8b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:

- AIME 2024 pass@1: 50.4
- MATH-500 pass@1: 89.1
- CodeForces Rating: 1205

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.

Hugging Face: 
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) 
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)   |,
    created=1738937718'''
)

DEEPSEEK_R1_DISTILL_QWEN_1_5B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-1.5b",
    price_in=1.8e-07,
    price_out=1.8e-07,
    creator="deepseek",
    description='''DeepSeek R1 Distill Qwen 1.5B is a distilled large language model based on  [Qwen 2.5 Math 1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It's a very small and efficient model which outperforms [GPT 4o 0513](/openai/gpt-4o-2024-05-13) on Math Benchmarks.

Other benchmark results include:

- AIME 2024 pass@1: 28.9
- AIME 2024 cons@64: 52.7
- MATH-500 pass@1: 83.9

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1738328067'''
)

DEEPSEEK_R1_DISTILL_QWEN_14B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-14b",
    price_in=1.5e-07,
    price_out=1.5e-07,
    creator="deepseek",
    description='''DeepSeek R1 Distill Qwen 14B is a distilled large language model based on [Qwen 2.5 14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.

Other benchmark results include:

- AIME 2024 pass@1: 69.7
- MATH-500 pass@1: 93.9
- CodeForces Rating: 1481

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1738193940'''
)

DEEPSEEK_R1_DISTILL_QWEN_14B_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-14b:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek R1 Distill Qwen 14B is a distilled large language model based on [Qwen 2.5 14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.

Other benchmark results include:

- AIME 2024 pass@1: 69.7
- MATH-500 pass@1: 93.9
- CodeForces Rating: 1481

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1738193940'''
)

DEEPSEEK_R1_DISTILL_QWEN_32B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-32b",
    price_in=1.2e-07,
    price_out=1.8e-07,
    creator="deepseek",
    description='''DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 72.6\n- MATH-500 pass@1: 94.3\n- CodeForces Rating: 1691\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1738194830'''
)

DEEPSEEK_R1_DISTILL_QWEN_32B_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-32b:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 72.6\n- MATH-500 pass@1: 94.3\n- CodeForces Rating: 1691\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1738194830'''
)

DEEPSEEK_R1_DISTILL_QWEN_7B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-7b",
    price_in=1e-07,
    price_out=2e-07,
    creator="deepseek",
    description='''DeepSeek-R1-Distill-Qwen-7B is a 7 billion parameter dense language model distilled from DeepSeek-R1, leveraging reinforcement learning-enhanced reasoning data generated by DeepSeek's larger models. The distillation process transfers advanced reasoning, math, and code capabilities into a smaller, more efficient model architecture based on Qwen2.5-Math-7B. This model demonstrates strong performance across mathematical benchmarks (92.8% pass@1 on MATH-500), coding tasks (Codeforces rating 1189), and general reasoning (49.1% pass@1 on GPQA Diamond), achieving competitive accuracy relative to larger models while maintaining smaller inference costs.,
    created=1748628237'''
)

DEEPSEEK_R1_ZERO_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1-zero:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek-R1-Zero is a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step. It's 671B parameters in size, with 37B active in an inference pass.

It demonstrates remarkable performance on reasoning. With RL, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors.

DeepSeek-R1-Zero encounters challenges such as endless repetition, poor readability, and language mixing. See [DeepSeek R1](/deepseek/deepseek-r1) for the SFT model.

,
    created=1741297434'''
)

DEEPSEEK_R1_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-r1:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.

Fully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).

MIT licensed: Distill & commercialize freely!,
    created=1737381095'''
)

DEEPSEEK_V3_BASE_FREE = ModelInfo(
    str_identifier="deepseek/deepseek-v3-base:free",
    price_in=0.0,
    price_out=0.0,
    creator="deepseek",
    description='''Note that this is a base model mostly meant for testing, you need to provide detailed prompts for the model to return useful responses. 

DeepSeek-V3 Base is a 671B parameter open Mixture-of-Experts (MoE) language model with 37B active parameters per forward pass and a context length of 128K tokens. Trained on 14.8T tokens using FP8 mixed precision, it achieves high training efficiency and stability, with strong performance across language, reasoning, math, and coding tasks. 

DeepSeek-V3 Base is the pre-trained model behind [DeepSeek V3](/deepseek/deepseek-chat-v3),
    created=1743272023'''
)

LLEMMA_7B = ModelInfo(
    str_identifier="eleutherai/llemma_7b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="eleutherai",
    description='''Llemma 7B is a language model for mathematics. It was initialized with Code Llama 7B weights, and trained on the Proof-Pile-2 for 200B tokens. Llemma models are particularly strong at chain-of-thought mathematical reasoning and using computational tools for mathematics, such as Python and formal theorem provers.,
    created=1744643225'''
)

EVA_LLAMA_3_33_70B = ModelInfo(
    str_identifier="eva-unit-01/eva-llama-3.33-70b",
    price_in=4e-06,
    price_out=6e-06,
    creator="eva-unit-01",
    description='''EVA Llama 3.33 70b is a roleplay and storywriting specialist model. It is a full-parameter finetune of [Llama-3.3-70B-Instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct) on mixture of synthetic and natural data.

It uses Celeste 70B 0.1 data mixture, greatly expanding it to improve versatility, creativity and "flavor" of the resulting model

This model was built with Llama by Meta.
,
    created=1734377303'''
)

EVA_QWEN_2_5_32B = ModelInfo(
    str_identifier="eva-unit-01/eva-qwen-2.5-32b",
    price_in=2.6e-06,
    price_out=3.4e-06,
    creator="eva-unit-01",
    description='''EVA Qwen2.5 32B is a roleplaying/storywriting specialist model. It's a full-parameter finetune of Qwen2.5-32B on mixture of synthetic and natural data.

It uses Celeste 70B 0.1 data mixture, greatly expanding it to improve versatility, creativity and "flavor" of the resulting model.,
    created=1731104847'''
)

EVA_QWEN_2_5_72B = ModelInfo(
    str_identifier="eva-unit-01/eva-qwen-2.5-72b",
    price_in=4e-06,
    price_out=6e-06,
    creator="eva-unit-01",
    description='''EVA Qwen2.5 72B is a roleplay and storywriting specialist model. It's a full-parameter finetune of Qwen2.5-72B on mixture of synthetic and natural data.

It uses Celeste 70B 0.1 data mixture, greatly expanding it to improve versatility, creativity and "flavor" of the resulting model.,
    created=1732210606'''
)

QWERKY_72B_FREE = ModelInfo(
    str_identifier="featherless/qwerky-72b:free",
    price_in=0.0,
    price_out=0.0,
    creator="featherless",
    description='''Qwerky-72B is a linear-attention RWKV variant of the Qwen 2.5 72B model, optimized to significantly reduce computational cost at scale. Leveraging linear attention, it achieves substantial inference speedups (>1000x) while retaining competitive accuracy on common benchmarks like ARC, HellaSwag, Lambada, and MMLU. It inherits knowledge and language support from Qwen 2.5, supporting approximately 30 languages, making it suitable for efficient inference in large-context applications.,
    created=1742481597'''
)

GEMINI_2_0_FLASH_001 = ModelInfo(
    str_identifier="google/gemini-2.0-flash-001",
    price_in=1e-07,
    price_out=4e-07,
    creator="google",
    description='''Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5). It introduces notable enhancements in multimodal understanding, coding capabilities, complex instruction following, and function calling. These advancements come together to deliver more seamless and robust agentic experiences.,
    created=1738769413'''
)

GEMINI_2_0_FLASH_EXP_FREE = ModelInfo(
    str_identifier="google/gemini-2.0-flash-exp:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5). It introduces notable enhancements in multimodal understanding, coding capabilities, complex instruction following, and function calling. These advancements come together to deliver more seamless and robust agentic experiences.,
    created=1733937523'''
)

GEMINI_2_0_FLASH_LITE_001 = ModelInfo(
    str_identifier="google/gemini-2.0-flash-lite-001",
    price_in=7.5e-08,
    price_out=3e-07,
    creator="google",
    description='''Gemini 2.0 Flash Lite offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5), all at extremely economical token prices.,
    created=1740506212'''
)

GEMINI_2_5_FLASH_PREVIEW = ModelInfo(
    str_identifier="google/gemini-2.5-flash-preview",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="google",
    description='''Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling. 

Note: This model is available in two variants: thinking and non-thinking. The output pricing varies significantly depending on whether the thinking capability is active. If you select the standard variant (without the ":thinking" suffix), the model will explicitly avoid generating thinking tokens. 

To utilize the thinking capability and receive thinking tokens, you must choose the ":thinking" variant, which will then incur the higher thinking-output pricing. 

Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).,
    created=1744914667'''
)

GEMINI_2_5_FLASH_PREVIEW_05_20 = ModelInfo(
    str_identifier="google/gemini-2.5-flash-preview-05-20",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="google",
    description='''Gemini 2.5 Flash May 20th Checkpoint is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling. 

Note: This model is available in two variants: thinking and non-thinking. The output pricing varies significantly depending on whether the thinking capability is active. If you select the standard variant (without the ":thinking" suffix), the model will explicitly avoid generating thinking tokens. 

To utilize the thinking capability and receive thinking tokens, you must choose the ":thinking" variant, which will then incur the higher thinking-output pricing. 

Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).,
    created=1747761924'''
)

GEMINI_2_5_FLASH_PREVIEW_05_20_THINKING = ModelInfo(
    str_identifier="google/gemini-2.5-flash-preview-05-20:thinking",
    price_in=1.5e-07,
    price_out=3.5e-06,
    creator="google",
    description='''Gemini 2.5 Flash May 20th Checkpoint is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling. 

Note: This model is available in two variants: thinking and non-thinking. The output pricing varies significantly depending on whether the thinking capability is active. If you select the standard variant (without the ":thinking" suffix), the model will explicitly avoid generating thinking tokens. 

To utilize the thinking capability and receive thinking tokens, you must choose the ":thinking" variant, which will then incur the higher thinking-output pricing. 

Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).,
    created=1747761924'''
)

GEMINI_2_5_FLASH_PREVIEW_THINKING = ModelInfo(
    str_identifier="google/gemini-2.5-flash-preview:thinking",
    price_in=1.5e-07,
    price_out=3.5e-06,
    creator="google",
    description='''Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling. 

Note: This model is available in two variants: thinking and non-thinking. The output pricing varies significantly depending on whether the thinking capability is active. If you select the standard variant (without the ":thinking" suffix), the model will explicitly avoid generating thinking tokens. 

To utilize the thinking capability and receive thinking tokens, you must choose the ":thinking" variant, which will then incur the higher thinking-output pricing. 

Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).,
    created=1744914667'''
)

GEMINI_2_5_PRO_EXP_03_25 = ModelInfo(
    str_identifier="google/gemini-2.5-pro-exp-03-25",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''This model has been deprecated by Google in favor of the (paid Preview model)[google/gemini-2.5-pro-preview]
 
Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.,
    created=1742922099'''
)

GEMINI_2_5_PRO_PREVIEW = ModelInfo(
    str_identifier="google/gemini-2.5-pro-preview",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="google",
    description='''Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.
,
    created=1749137257'''
)

GEMINI_2_5_PRO_PREVIEW_05_06 = ModelInfo(
    str_identifier="google/gemini-2.5-pro-preview-05-06",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="google",
    description='''Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.,
    created=1746578513'''
)

GEMINI_FLASH_1_5 = ModelInfo(
    str_identifier="google/gemini-flash-1.5",
    price_in=7.5e-08,
    price_out=3e-07,
    creator="google",
    description='''Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video. It's adept at processing visual and text inputs such as photographs, documents, infographics, and screenshots.

Gemini 1.5 Flash is designed for high-volume, high-frequency tasks where cost and latency matter. On most common tasks, Flash achieves comparable quality to other Gemini Pro models at a significantly reduced cost. Flash is well-suited for applications like chat assistants and on-demand content generation where speed and scale matter.

Usage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).

#multimodal,
    created=1715644800'''
)

GEMINI_FLASH_1_5_8B = ModelInfo(
    str_identifier="google/gemini-flash-1.5-8b",
    price_in=3.75e-08,
    price_out=1.5e-07,
    creator="google",
    description='''Gemini Flash 1.5 8B is optimized for speed and efficiency, offering enhanced performance in small prompt tasks like chat, transcription, and translation. With reduced latency, it is highly effective for real-time and large-scale operations. This model focuses on cost-effective solutions while maintaining high-quality results.

[Click here to learn more about this model](https://developers.googleblog.com/en/gemini-15-flash-8b-is-now-generally-available-for-use/).

Usage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).,
    created=1727913600'''
)

GEMINI_PRO_1_5 = ModelInfo(
    str_identifier="google/gemini-pro-1.5",
    price_in=1.25e-06,
    price_out=5e-06,
    creator="google",
    description='''Google's latest multimodal model, supports image and video[0] in text or chat prompts.

Optimized for language tasks including:

- Code generation
- Text generation
- Text editing
- Problem solving
- Recommendations
- Information extraction
- Data extraction or generation
- AI agents

Usage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).

* [0]: Video input is not available through OpenRouter at this time.,
    created=1712620800'''
)

GEMMA_2_27B_IT = ModelInfo(
    str_identifier="google/gemma-2-27b-it",
    price_in=8e-07,
    price_out=8e-07,
    creator="google",
    description='''Gemma 2 27B by Google is an open model built from the same research and technology used to create the [Gemini models](/models?q=gemini).

Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning.

See the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).,
    created=1720828800'''
)

GEMMA_2_9B_IT = ModelInfo(
    str_identifier="google/gemma-2-9b-it",
    price_in=2e-07,
    price_out=2e-07,
    creator="google",
    description='''Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.

Designed for a wide variety of tasks, it empowers developers and researchers to build innovative applications, while maintaining accessibility, safety, and cost-effectiveness.

See the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).,
    created=1719532800'''
)

GEMMA_2_9B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-2-9b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.

Designed for a wide variety of tasks, it empowers developers and researchers to build innovative applications, while maintaining accessibility, safety, and cost-effectiveness.

See the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).,
    created=1719532800'''
)

GEMMA_2B_IT = ModelInfo(
    str_identifier="google/gemma-2b-it",
    price_in=1e-07,
    price_out=1e-07,
    creator="google",
    description='''Gemma 1 2B by Google is an open model built from the same research and technology used to create the [Gemini models](/models?q=gemini).

Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning.

Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).,
    created=1748460815'''
)

GEMMA_3_12B_IT = ModelInfo(
    str_identifier="google/gemma-3-12b-it",
    price_in=5e-08,
    price_out=1e-07,
    creator="google",
    description='''Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 12B is the second largest in the family of Gemma 3 models after [Gemma 3 27B](google/gemma-3-27b-it),
    created=1741902625'''
)

GEMMA_3_12B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-3-12b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 12B is the second largest in the family of Gemma 3 models after [Gemma 3 27B](google/gemma-3-27b-it),
    created=1741902625'''
)

GEMMA_3_1B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-3-1b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 3 1B is the smallest of the new Gemma 3 family. It handles context windows up to 32k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Note: Gemma 3 1B is not multimodal. For the smallest multimodal Gemma 3 model, please see [Gemma 3 4B](google/gemma-3-4b-it),
    created=1741963556'''
)

GEMMA_3_27B_IT = ModelInfo(
    str_identifier="google/gemma-3-27b-it",
    price_in=1e-07,
    price_out=2e-07,
    creator="google",
    description='''Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 27B is Google's latest open source model, successor to [Gemma 2](google/gemma-2-27b-it),
    created=1741756359'''
)

GEMMA_3_27B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-3-27b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 27B is Google's latest open source model, successor to [Gemma 2](google/gemma-2-27b-it),
    created=1741756359'''
)

GEMMA_3_4B_IT = ModelInfo(
    str_identifier="google/gemma-3-4b-it",
    price_in=2e-08,
    price_out=4e-08,
    creator="google",
    description='''Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling.,
    created=1741905510'''
)

GEMMA_3_4B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-3-4b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling.,
    created=1741905510'''
)

GEMMA_3N_E4B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-3n-e4b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource devices, such as phones, laptops, and tablets. It supports multimodal inputs—including text, visual data, and audio—enabling diverse tasks such as text generation, speech recognition, translation, and image analysis. Leveraging innovations like Per-Layer Embedding (PLE) caching and the MatFormer architecture, Gemma 3n dynamically manages memory usage and computational load by selectively activating model parameters, significantly reducing runtime resource requirements.

This model supports a wide linguistic range (trained in over 140 languages) and features a flexible 32K token context window. Gemma 3n can selectively load parameters, optimizing memory and computational efficiency based on the task or device capabilities, making it well-suited for privacy-focused, offline-capable applications and on-device AI solutions. [Read more in the blog post](https://developers.googleblog.com/en/introducing-gemma-3n/),
    created=1747776824'''
)

MYTHOMAX_L2_13B = ModelInfo(
    str_identifier="gryphe/mythomax-l2-13b",
    price_in=6.5e-08,
    price_out=6.5e-08,
    creator="gryphe",
    description='''One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge,
    created=1688256000'''
)

MERCURY_CODER_SMALL_BETA = ModelInfo(
    str_identifier="inception/mercury-coder-small-beta",
    price_in=2.5e-07,
    price_out=1e-06,
    creator="inception",
    description='''Mercury Coder Small is the first diffusion large language model (dLLM). Applying a breakthrough discrete diffusion approach, the model runs 5-10x faster than even speed optimized models like Claude 3.5 Haiku and GPT-4o Mini while matching their performance. Mercury Coder Small's speed means that developers can stay in the flow while coding, enjoying rapid chat-based iteration and responsive code completion suggestions. On Copilot Arena, Mercury Coder ranks 1st in speed and ties for 2nd in quality. Read more in the [blog post here](https://www.inceptionlabs.ai/introducing-mercury).,
    created=1746033880'''
)

MN_INFEROR_12B = ModelInfo(
    str_identifier="infermatic/mn-inferor-12b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="infermatic",
    description='''Inferor 12B is a merge of top roleplay models, expert on immersive narratives and storytelling.

This model was merged using the [Model Stock](https://arxiv.org/abs/2403.19522) merge method using [anthracite-org/magnum-v4-12b](https://openrouter.ai/anthracite-org/magnum-v4-72b) as a base.
,
    created=1731464428'''
)

INFLECTION_3_PI = ModelInfo(
    str_identifier="inflection/inflection-3-pi",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="inflection",
    description='''Inflection 3 Pi powers Inflection's [Pi](https://pi.ai) chatbot, including backstory, emotional intelligence, productivity, and safety. It has access to recent news, and excels in scenarios like customer support and roleplay.

Pi has been trained to mirror your tone and style, if you use more emojis, so will Pi! Try experimenting with various prompts and conversation styles.,
    created=1728604800'''
)

INFLECTION_3_PRODUCTIVITY = ModelInfo(
    str_identifier="inflection/inflection-3-productivity",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="inflection",
    description='''Inflection 3 Productivity is optimized for following instructions. It is better for tasks requiring JSON output or precise adherence to provided guidelines. It has access to recent news.

For emotional intelligence similar to Pi, see [Inflect 3 Pi](/inflection/inflection-3-pi)

See [Inflection's announcement](https://inflection.ai/blog/enterprise) for more details.,
    created=1728604800'''
)

LFM_3B = ModelInfo(
    str_identifier="liquid/lfm-3b",
    price_in=2e-08,
    price_out=2e-08,
    creator="liquid",
    description='''Liquid's LFM 3B delivers incredible performance for its size. It positions itself as first place among 3B parameter transformers, hybrids, and RNN models It is also on par with Phi-3.5-mini on multiple benchmarks, while being 18.4% smaller.

LFM-3B is the ideal choice for mobile and other edge text-based applications.

See the [launch announcement](https://www.liquid.ai/liquid-foundation-models) for benchmarks and more info.,
    created=1737806501'''
)

LFM_40B = ModelInfo(
    str_identifier="liquid/lfm-40b",
    price_in=1.5e-07,
    price_out=1.5e-07,
    creator="liquid",
    description='''Liquid's 40.3B Mixture of Experts (MoE) model. Liquid Foundation Models (LFMs) are large neural networks built with computational units rooted in dynamic systems.

LFMs are general-purpose AI models that can be used to model any kind of sequential data, including video, audio, text, time series, and signals.

See the [launch announcement](https://www.liquid.ai/liquid-foundation-models) for benchmarks and more info.,
    created=1727654400'''
)

LFM_7B = ModelInfo(
    str_identifier="liquid/lfm-7b",
    price_in=1e-08,
    price_out=1e-08,
    creator="liquid",
    description='''LFM-7B, a new best-in-class language model. LFM-7B is designed for exceptional chat capabilities, including languages like Arabic and Japanese. Powered by the Liquid Foundation Model (LFM) architecture, it exhibits unique features like low memory footprint and fast inference speed. 

LFM-7B is the world’s best-in-class multilingual language model in English, Arabic, and Japanese.

See the [launch announcement](https://www.liquid.ai/lfm-7b) for benchmarks and more info.,
    created=1737806883'''
)

WEAVER = ModelInfo(
    str_identifier="mancer/weaver",
    price_in=1.5e-06,
    price_out=1.5e-06,
    creator="mancer",
    description='''An attempt to recreate Claude-style verbosity, but don't expect the same level of coherence or memory. Meant for use in roleplay/narrative situations.,
    created=1690934400'''
)

LLAMA_2_70B_CHAT = ModelInfo(
    str_identifier="meta-llama/llama-2-70b-chat",
    price_in=9e-07,
    price_out=9e-07,
    creator="meta-llama",
    description='''The flagship, 70 billion parameter language model from Meta, fine tuned for chat completions. Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety.,
    created=1687219200'''
)

LLAMA_3_70B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3-70b-instruct",
    price_in=3e-07,
    price_out=4e-07,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 70B instruct-tuned version was optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1713398400'''
)

LLAMA_3_8B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3-8b-instruct",
    price_in=3e-08,
    price_out=6e-08,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 8B instruct-tuned version was optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1713398400'''
)

LLAMA_3_1_405B = ModelInfo(
    str_identifier="meta-llama/llama-3.1-405b",
    price_in=2e-06,
    price_out=2e-06,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This is the base 405B pre-trained version.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1722556800'''
)

LLAMA_3_1_405B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.1-405b-instruct",
    price_in=8e-07,
    price_out=8e-07,
    creator="meta-llama",
    description='''The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.

Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 405B instruct-tuned version is optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models including GPT-4o and Claude 3.5 Sonnet in evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_1_405B_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.1-405b:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This is the base 405B pre-trained version.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1722556800'''
)

LLAMA_3_1_70B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.1-70b-instruct",
    price_in=1e-07,
    price_out=2.8e-07,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instruct-tuned version is optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_1_8B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.1-8b-instruct",
    price_in=1.9e-08,
    price_out=3e-08,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 8B instruct-tuned version is fast and efficient.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_1_8B_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.1-8b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 8B instruct-tuned version is fast and efficient.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_2_11B_VISION_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.2-11b-vision-instruct",
    price_in=4.9e-08,
    price_out=4.9e-08,
    creator="meta-llama",
    description='''Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks combining visual and textual data. It excels in tasks such as image captioning and visual question answering, bridging the gap between language generation and visual reasoning. Pre-trained on a massive dataset of image-text pairs, it performs well in complex, high-accuracy image analysis.

Its ability to integrate visual understanding with language processing makes it an ideal solution for industries requiring comprehensive visual-linguistic AI applications, such as content creation, AI-driven customer service, and research.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_11B_VISION_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.2-11b-vision-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks combining visual and textual data. It excels in tasks such as image captioning and visual question answering, bridging the gap between language generation and visual reasoning. Pre-trained on a massive dataset of image-text pairs, it performs well in complex, high-accuracy image analysis.

Its ability to integrate visual understanding with language processing makes it an ideal solution for industries requiring comprehensive visual-linguistic AI applications, such as content creation, AI-driven customer service, and research.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_1B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.2-1b-instruct",
    price_in=5e-09,
    price_out=1e-08,
    creator="meta-llama",
    description='''Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural language tasks, such as summarization, dialogue, and multilingual text analysis. Its smaller size allows it to operate efficiently in low-resource environments while maintaining strong task performance.

Supporting eight core languages and fine-tunable for more, Llama 1.3B is ideal for businesses or developers seeking lightweight yet powerful AI solutions that can operate in diverse multilingual settings without the high computational demand of larger models.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_1B_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.2-1b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural language tasks, such as summarization, dialogue, and multilingual text analysis. Its smaller size allows it to operate efficiently in low-resource environments while maintaining strong task performance.

Supporting eight core languages and fine-tunable for more, Llama 1.3B is ideal for businesses or developers seeking lightweight yet powerful AI solutions that can operate in diverse multilingual settings without the high computational demand of larger models.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_3B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.2-3b-instruct",
    price_in=1e-08,
    price_out=2e-08,
    creator="meta-llama",
    description='''Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and summarization. Designed with the latest transformer architecture, it supports eight languages, including English, Spanish, and Hindi, and is adaptable for additional languages.

Trained on 9 trillion tokens, the Llama 3.2 3B model excels in instruction-following, complex reasoning, and tool use. Its balanced performance makes it ideal for applications needing accuracy and efficiency in text generation across multilingual settings.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_3B_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.2-3b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and summarization. Designed with the latest transformer architecture, it supports eight languages, including English, Spanish, and Hindi, and is adaptable for additional languages.

Trained on 9 trillion tokens, the Llama 3.2 3B model excels in instruction-following, complex reasoning, and tool use. Its balanced performance makes it ideal for applications needing accuracy and efficiency in text generation across multilingual settings.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_90B_VISION_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.2-90b-vision-instruct",
    price_in=1.2e-06,
    price_out=1.2e-06,
    creator="meta-llama",
    description='''The Llama 90B Vision model is a top-tier, 90-billion-parameter multimodal model designed for the most challenging visual reasoning and language tasks. It offers unparalleled accuracy in image captioning, visual question answering, and advanced image-text comprehension. Pre-trained on vast multimodal datasets and fine-tuned with human feedback, the Llama 90B Vision is engineered to handle the most demanding image-based AI tasks.

This model is perfect for industries requiring cutting-edge multimodal AI capabilities, particularly those dealing with complex, real-time visual and textual analysis.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_3_70B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.3-70b-instruct",
    price_in=7e-08,
    price_out=2.5e-07,
    creator="meta-llama",
    description='''The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.

Supported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

[Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md),
    created=1733506137'''
)

LLAMA_3_3_70B_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.3-70b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.

Supported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

[Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md),
    created=1733506137'''
)

LLAMA_3_3_8B_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.3-8b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''A lightweight and ultra-fast variant of Llama 3.3 70B, for use when quick response times are needed most.,
    created=1747230154'''
)

LLAMA_4_MAVERICK = ModelInfo(
    str_identifier="meta-llama/llama-4-maverick",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="meta-llama",
    description='''Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total). It supports multilingual text and image input, and produces multilingual text and code output across 12 supported languages. Optimized for vision-language tasks, Maverick is instruction-tuned for assistant-like behavior, image reasoning, and general-purpose multimodal interaction.

Maverick features early fusion for native multimodality and a 1 million token context window. It was trained on a curated mixture of public, licensed, and Meta-platform data, covering ~22 trillion tokens, with a knowledge cutoff in August 2024. Released on April 5, 2025 under the Llama 4 Community License, Maverick is suited for research and commercial applications requiring advanced multimodal understanding and high model throughput.,
    created=1743881822'''
)

LLAMA_4_MAVERICK_FREE = ModelInfo(
    str_identifier="meta-llama/llama-4-maverick:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total). It supports multilingual text and image input, and produces multilingual text and code output across 12 supported languages. Optimized for vision-language tasks, Maverick is instruction-tuned for assistant-like behavior, image reasoning, and general-purpose multimodal interaction.

Maverick features early fusion for native multimodality and a 1 million token context window. It was trained on a curated mixture of public, licensed, and Meta-platform data, covering ~22 trillion tokens, with a knowledge cutoff in August 2024. Released on April 5, 2025 under the Llama 4 Community License, Maverick is suited for research and commercial applications requiring advanced multimodal understanding and high model throughput.,
    created=1743881822'''
)

LLAMA_4_SCOUT = ModelInfo(
    str_identifier="meta-llama/llama-4-scout",
    price_in=8e-08,
    price_out=3e-07,
    creator="meta-llama",
    description='''Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, activating 17 billion parameters out of a total of 109B. It supports native multimodal input (text and image) and multilingual output (text and code) across 12 supported languages. Designed for assistant-style interaction and visual reasoning, Scout uses 16 experts per forward pass and features a context length of 10 million tokens, with a training corpus of ~40 trillion tokens.

Built for high efficiency and local or commercial deployment, Llama 4 Scout incorporates early fusion for seamless modality integration. It is instruction-tuned for use in multilingual chat, captioning, and image understanding tasks. Released under the Llama 4 Community License, it was last trained on data up to August 2024 and launched publicly on April 5, 2025.,
    created=1743881519'''
)

LLAMA_4_SCOUT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-4-scout:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, activating 17 billion parameters out of a total of 109B. It supports native multimodal input (text and image) and multilingual output (text and code) across 12 supported languages. Designed for assistant-style interaction and visual reasoning, Scout uses 16 experts per forward pass and features a context length of 10 million tokens, with a training corpus of ~40 trillion tokens.

Built for high efficiency and local or commercial deployment, Llama 4 Scout incorporates early fusion for seamless modality integration. It is instruction-tuned for use in multilingual chat, captioning, and image understanding tasks. Released under the Llama 4 Community License, it was last trained on data up to August 2024 and launched publicly on April 5, 2025.,
    created=1743881519'''
)

LLAMA_GUARD_2_8B = ModelInfo(
    str_identifier="meta-llama/llama-guard-2-8b",
    price_in=2e-07,
    price_out=2e-07,
    creator="meta-llama",
    description='''This safeguard model has 8B parameters and is based on the Llama 3 family. Just like is predecessor, [LlamaGuard 1](https://huggingface.co/meta-llama/LlamaGuard-7b), it can do both prompt and response classification.

LlamaGuard 2 acts as a normal LLM would, generating text that indicates whether the given input/output is safe/unsafe. If deemed unsafe, it will also share the content categories violated.

For best results, please use raw prompt input or the `/completions` endpoint, instead of the chat API.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1715558400'''
)

LLAMA_GUARD_3_8B = ModelInfo(
    str_identifier="meta-llama/llama-guard-3-8b",
    price_in=2e-08,
    price_out=6e-08,
    creator="meta-llama",
    description='''Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 3 was aligned to safeguard against the MLCommons standardized hazards taxonomy and designed to support Llama 3.1 capabilities. Specifically, it provides content moderation in 8 languages, and was optimized to support safety and security for search and code interpreter tool calls.
,
    created=1739401318'''
)

LLAMA_GUARD_4_12B = ModelInfo(
    str_identifier="meta-llama/llama-guard-4-12b",
    price_in=5e-08,
    price_out=5e-08,
    creator="meta-llama",
    description='''Llama Guard 4 is a Llama 4 Scout-derived multimodal pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM—generating text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 4 was aligned to safeguard against the standardized MLCommons hazards taxonomy and designed to support multimodal Llama 4 capabilities. Specifically, it combines features from previous Llama Guard models, providing content moderation for English and multiple supported languages, along with enhanced capabilities to handle mixed text-and-image prompts, including multiple images. Additionally, Llama Guard 4 is integrated into the Llama Moderations API, extending robust safety classification to text and images.,
    created=1745975193'''
)

MAI_DS_R1_FREE = ModelInfo(
    str_identifier="microsoft/mai-ds-r1:free",
    price_in=0.0,
    price_out=0.0,
    creator="microsoft",
    description='''MAI-DS-R1 is a post-trained variant of DeepSeek-R1 developed by the Microsoft AI team to improve the model’s responsiveness on previously blocked topics while enhancing its safety profile. Built on top of DeepSeek-R1’s reasoning foundation, it integrates 110k examples from the Tulu-3 SFT dataset and 350k internally curated multilingual safety-alignment samples. The model retains strong reasoning, coding, and problem-solving capabilities, while unblocking a wide range of prompts previously restricted in R1.

MAI-DS-R1 demonstrates improved performance on harm mitigation benchmarks and maintains competitive results across general reasoning tasks. It surpasses R1-1776 in satisfaction metrics for blocked queries and reduces leakage in harmful content categories. The model is based on a transformer MoE architecture and is suitable for general-purpose use cases, excluding high-stakes domains such as legal, medical, or autonomous systems.,
    created=1745194100'''
)

PHI_3_MEDIUM_128K_INSTRUCT = ModelInfo(
    str_identifier="microsoft/phi-3-medium-128k-instruct",
    price_in=1e-06,
    price_out=1e-06,
    creator="microsoft",
    description='''Phi-3 128K Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjustments, it excels in tasks involving common sense, mathematics, logical reasoning, and code processing.

At time of release, Phi-3 Medium demonstrated state-of-the-art performance among lightweight models. In the MMLU-Pro eval, the model even comes close to a Llama3 70B level of performance.

For 4k context length, try [Phi-3 Medium 4K](/models/microsoft/phi-3-medium-4k-instruct).,
    created=1716508800'''
)

PHI_3_MINI_128K_INSTRUCT = ModelInfo(
    str_identifier="microsoft/phi-3-mini-128k-instruct",
    price_in=1e-07,
    price_out=1e-07,
    creator="microsoft",
    description='''Phi-3 Mini is a powerful 3.8B parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjustments, it excels in tasks involving common sense, mathematics, logical reasoning, and code processing.

At time of release, Phi-3 Medium demonstrated state-of-the-art performance among lightweight models. This model is static, trained on an offline dataset with an October 2023 cutoff date.,
    created=1716681600'''
)

PHI_3_5_MINI_128K_INSTRUCT = ModelInfo(
    str_identifier="microsoft/phi-3.5-mini-128k-instruct",
    price_in=1e-07,
    price_out=1e-07,
    creator="microsoft",
    description='''Phi-3.5 models are lightweight, state-of-the-art open models. These models were trained with Phi-3 datasets that include both synthetic data and the filtered, publicly available websites data, with a focus on high quality and reasoning-dense properties. Phi-3.5 Mini uses 3.8B parameters, and is a dense decoder-only transformer model using the same tokenizer as [Phi-3 Mini](/models/microsoft/phi-3-mini-128k-instruct).

The models underwent a rigorous enhancement process, incorporating both supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures. When assessed against benchmarks that test common sense, language understanding, math, code, long context and logical reasoning, Phi-3.5 models showcased robust and state-of-the-art performance among models with less than 13 billion parameters.,
    created=1724198400'''
)

PHI_4 = ModelInfo(
    str_identifier="microsoft/phi-4",
    price_in=7e-08,
    price_out=1.4e-07,
    creator="microsoft",
    description='''[Microsoft Research](/microsoft) Phi-4 is designed to perform well in complex reasoning tasks and can operate efficiently in situations with limited memory or where quick responses are needed. 

At 14 billion parameters, it was trained on a mix of high-quality synthetic datasets, data from curated websites, and academic materials. It has undergone careful improvement to follow instructions accurately and maintain strong safety standards. It works best with English language inputs.

For more information, please see [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905)
,
    created=1736489872'''
)

PHI_4_MULTIMODAL_INSTRUCT = ModelInfo(
    str_identifier="microsoft/phi-4-multimodal-instruct",
    price_in=5e-08,
    price_out=1e-07,
    creator="microsoft",
    description='''Phi-4 Multimodal Instruct is a versatile 5.6B parameter foundation model that combines advanced reasoning and instruction-following capabilities across both text and visual inputs, providing accurate text outputs. The unified architecture enables efficient, low-latency inference, suitable for edge and mobile deployments. Phi-4 Multimodal Instruct supports text inputs in multiple languages including Arabic, Chinese, English, French, German, Japanese, Spanish, and more, with visual input optimized primarily for English. It delivers impressive performance on multimodal tasks involving mathematical, scientific, and document reasoning, providing developers and enterprises a powerful yet compact model for sophisticated interactive applications. For more information, see the [Phi-4 Multimodal blog post](https://azure.microsoft.com/en-us/blog/empowering-innovation-the-next-generation-of-the-phi-family/).
,
    created=1741396284'''
)

PHI_4_REASONING_PLUS = ModelInfo(
    str_identifier="microsoft/phi-4-reasoning-plus",
    price_in=7e-08,
    price_out=3.5e-07,
    creator="microsoft",
    description='''Phi-4-reasoning-plus is an enhanced 14B parameter model from Microsoft, fine-tuned from Phi-4 with additional reinforcement learning to boost accuracy on math, science, and code reasoning tasks. It uses the same dense decoder-only transformer architecture as Phi-4, but generates longer, more comprehensive outputs structured into a step-by-step reasoning trace and final answer.

While it offers improved benchmark scores over Phi-4-reasoning across tasks like AIME, OmniMath, and HumanEvalPlus, its responses are typically ~50% longer, resulting in higher latency. Designed for English-only applications, it is well-suited for structured reasoning workflows where output quality takes priority over response speed.,
    created=1746130961'''
)

PHI_4_REASONING_PLUS_FREE = ModelInfo(
    str_identifier="microsoft/phi-4-reasoning-plus:free",
    price_in=0.0,
    price_out=0.0,
    creator="microsoft",
    description='''Phi-4-reasoning-plus is an enhanced 14B parameter model from Microsoft, fine-tuned from Phi-4 with additional reinforcement learning to boost accuracy on math, science, and code reasoning tasks. It uses the same dense decoder-only transformer architecture as Phi-4, but generates longer, more comprehensive outputs structured into a step-by-step reasoning trace and final answer.

While it offers improved benchmark scores over Phi-4-reasoning across tasks like AIME, OmniMath, and HumanEvalPlus, its responses are typically ~50% longer, resulting in higher latency. Designed for English-only applications, it is well-suited for structured reasoning workflows where output quality takes priority over response speed.,
    created=1746130961'''
)

PHI_4_REASONING_FREE = ModelInfo(
    str_identifier="microsoft/phi-4-reasoning:free",
    price_in=0.0,
    price_out=0.0,
    creator="microsoft",
    description='''Phi-4-reasoning is a 14B parameter dense decoder-only transformer developed by Microsoft, fine-tuned from Phi-4 to enhance complex reasoning capabilities. It uses a combination of supervised fine-tuning on chain-of-thought traces and reinforcement learning, targeting math, science, and code reasoning tasks. With a 32k context window and high inference efficiency, it is optimized for structured responses in a two-part format: reasoning trace followed by a final solution.

The model achieves strong results on specialized benchmarks such as AIME, OmniMath, and LiveCodeBench, outperforming many larger models in structured reasoning tasks. It is released under the MIT license and intended for use in latency-constrained, English-only environments requiring reliable step-by-step logic. Recommended usage includes ChatML prompts and structured reasoning format for best results.,
    created=1746121275'''
)

WIZARDLM_2_8X22B = ModelInfo(
    str_identifier="microsoft/wizardlm-2-8x22b",
    price_in=4.8e-07,
    price_out=4.8e-07,
    creator="microsoft",
    description='''WizardLM-2 8x22B is Microsoft AI's most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing state-of-the-art opensource models.

It is an instruct finetune of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b).

To read more about the model release, [click here](https://wizardlm.github.io/WizardLM2/).

#moe,
    created=1713225600'''
)

MINIMAX_01 = ModelInfo(
    str_identifier="minimax/minimax-01",
    price_in=2e-07,
    price_out=1.1e-06,
    creator="minimax",
    description='''MiniMax-01 is a combines MiniMax-Text-01 for text generation and MiniMax-VL-01 for image understanding. It has 456 billion parameters, with 45.9 billion parameters activated per inference, and can handle a context of up to 4 million tokens.

The text model adopts a hybrid architecture that combines Lightning Attention, Softmax Attention, and Mixture-of-Experts (MoE). The image model adopts the “ViT-MLP-LLM” framework and is trained on top of the text model.

To read more about the release, see: https://www.minimaxi.com/en/news/minimax-01-series-2,
    created=1736915462'''
)

CODESTRAL_2501 = ModelInfo(
    str_identifier="mistralai/codestral-2501",
    price_in=3e-07,
    price_out=9e-07,
    creator="mistralai",
    description='''[Mistral](/mistralai)'s cutting-edge language model for coding. Codestral specializes in low-latency, high-frequency tasks such as fill-in-the-middle (FIM), code correction and test generation. 

Learn more on their blog post: https://mistral.ai/news/codestral-2501/,
    created=1736895522'''
)

DEVSTRAL_SMALL = ModelInfo(
    str_identifier="mistralai/devstral-small",
    price_in=6e-08,
    price_out=1.2e-07,
    creator="mistralai",
    description='''Devstral-Small-2505 is a 24B parameter agentic LLM fine-tuned from Mistral-Small-3.1, jointly developed by Mistral AI and All Hands AI for advanced software engineering tasks. It is optimized for codebase exploration, multi-file editing, and integration into coding agents, achieving state-of-the-art results on SWE-Bench Verified (46.8%).

Devstral supports a 128k context window and uses a custom Tekken tokenizer. It is text-only, with the vision encoder removed, and is suitable for local deployment on high-end consumer hardware (e.g., RTX 4090, 32GB RAM Macs). Devstral is best used in agentic workflows via the OpenHands scaffold and is compatible with inference frameworks like vLLM, Transformers, and Ollama. It is released under the Apache 2.0 license.,
    created=1747837379'''
)

DEVSTRAL_SMALL_FREE = ModelInfo(
    str_identifier="mistralai/devstral-small:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''Devstral-Small-2505 is a 24B parameter agentic LLM fine-tuned from Mistral-Small-3.1, jointly developed by Mistral AI and All Hands AI for advanced software engineering tasks. It is optimized for codebase exploration, multi-file editing, and integration into coding agents, achieving state-of-the-art results on SWE-Bench Verified (46.8%).

Devstral supports a 128k context window and uses a custom Tekken tokenizer. It is text-only, with the vision encoder removed, and is suitable for local deployment on high-end consumer hardware (e.g., RTX 4090, 32GB RAM Macs). Devstral is best used in agentic workflows via the OpenHands scaffold and is compatible with inference frameworks like vLLM, Transformers, and Ollama. It is released under the Apache 2.0 license.,
    created=1747837379'''
)

MINISTRAL_3B = ModelInfo(
    str_identifier="mistralai/ministral-3b",
    price_in=4e-08,
    price_out=4e-08,
    creator="mistralai",
    description='''Ministral 3B is a 3B parameter model optimized for on-device and edge computing. It excels in knowledge, commonsense reasoning, and function-calling, outperforming larger models like Mistral 7B on most benchmarks. Supporting up to 128k context length, it’s ideal for orchestrating agentic workflows and specialist tasks with efficient inference.,
    created=1729123200'''
)

MINISTRAL_8B = ModelInfo(
    str_identifier="mistralai/ministral-8b",
    price_in=1e-07,
    price_out=1e-07,
    creator="mistralai",
    description='''Ministral 8B is an 8B parameter model featuring a unique interleaved sliding-window attention pattern for faster, memory-efficient inference. Designed for edge use cases, it supports up to 128k context length and excels in knowledge and reasoning tasks. It outperforms peers in the sub-10B category, making it perfect for low-latency, privacy-first applications.,
    created=1729123200'''
)

MISTRAL_7B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mistral-7b-instruct",
    price_in=2.8e-08,
    price_out=5.4e-08,
    creator="mistralai",
    description='''A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.

*Mistral 7B Instruct has multiple version variants, and this is intended to be the latest version.*,
    created=1716768000'''
)

MISTRAL_7B_INSTRUCT_V0_1 = ModelInfo(
    str_identifier="mistralai/mistral-7b-instruct-v0.1",
    price_in=1.1e-07,
    price_out=1.9e-07,
    creator="mistralai",
    description='''A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with optimizations for speed and context length.,
    created=1695859200'''
)

MISTRAL_7B_INSTRUCT_V0_2 = ModelInfo(
    str_identifier="mistralai/mistral-7b-instruct-v0.2",
    price_in=2e-07,
    price_out=2e-07,
    creator="mistralai",
    description='''A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.

An improved version of [Mistral 7B Instruct](/modelsmistralai/mistral-7b-instruct-v0.1), with the following changes:

- 32k context window (vs 8k context in v0.1)
- Rope-theta = 1e6
- No Sliding-Window Attention,
    created=1703721600'''
)

MISTRAL_7B_INSTRUCT_V0_3 = ModelInfo(
    str_identifier="mistralai/mistral-7b-instruct-v0.3",
    price_in=2.8e-08,
    price_out=5.4e-08,
    creator="mistralai",
    description='''A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.

An improved version of [Mistral 7B Instruct v0.2](/models/mistralai/mistral-7b-instruct-v0.2), with the following changes:

- Extended vocabulary to 32768
- Supports v3 Tokenizer
- Supports function calling

NOTE: Support for function calling depends on the provider.,
    created=1716768000'''
)

MISTRAL_7B_INSTRUCT_FREE = ModelInfo(
    str_identifier="mistralai/mistral-7b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.

*Mistral 7B Instruct has multiple version variants, and this is intended to be the latest version.*,
    created=1716768000'''
)

MISTRAL_LARGE = ModelInfo(
    str_identifier="mistralai/mistral-large",
    price_in=2e-06,
    price_out=6e-06,
    creator="mistralai",
    description='''This is Mistral AI's flagship model, Mistral Large 2 (version `mistral-large-2407`). It's a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https://mistral.ai/news/mistral-large-2407/).

It supports dozens of languages including French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, and Korean, along with 80+ coding languages including Python, Java, C, C++, JavaScript, and Bash. Its long context window allows precise information recall from large documents.,
    created=1708905600'''
)

MISTRAL_LARGE_2407 = ModelInfo(
    str_identifier="mistralai/mistral-large-2407",
    price_in=2e-06,
    price_out=6e-06,
    creator="mistralai",
    description='''This is Mistral AI's flagship model, Mistral Large 2 (version mistral-large-2407). It's a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https://mistral.ai/news/mistral-large-2407/).

It supports dozens of languages including French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, and Korean, along with 80+ coding languages including Python, Java, C, C++, JavaScript, and Bash. Its long context window allows precise information recall from large documents.
,
    created=1731978415'''
)

MISTRAL_LARGE_2411 = ModelInfo(
    str_identifier="mistralai/mistral-large-2411",
    price_in=2e-06,
    price_out=6e-06,
    creator="mistralai",
    description='''Mistral Large 2 2411 is an update of [Mistral Large 2](/mistralai/mistral-large) released together with [Pixtral Large 2411](/mistralai/pixtral-large-2411)

It provides a significant upgrade on the previous [Mistral Large 24.07](/mistralai/mistral-large-2407), with notable improvements in long context understanding, a new system prompt, and more accurate function calling.,
    created=1731978685'''
)

MISTRAL_MEDIUM = ModelInfo(
    str_identifier="mistralai/mistral-medium",
    price_in=2.75e-06,
    price_out=8.1e-06,
    creator="mistralai",
    description='''This is Mistral AI's closed-source, medium-sided model. It's powered by a closed-source prototype and excels at reasoning, code, JSON, chat, and more. In benchmarks, it compares with many of the flagship models of other companies.,
    created=1704844800'''
)

MISTRAL_MEDIUM_3 = ModelInfo(
    str_identifier="mistralai/mistral-medium-3",
    price_in=4e-07,
    price_out=2e-06,
    creator="mistralai",
    description='''Mistral Medium 3 is a high-performance enterprise-grade language model designed to deliver frontier-level capabilities at significantly reduced operational cost. It balances state-of-the-art reasoning and multimodal performance with 8× lower cost compared to traditional large models, making it suitable for scalable deployments across professional and industrial use cases.

The model excels in domains such as coding, STEM reasoning, and enterprise adaptation. It supports hybrid, on-prem, and in-VPC deployments and is optimized for integration into custom workflows. Mistral Medium 3 offers competitive accuracy relative to larger models like Claude Sonnet 3.5/3.7, Llama 4 Maverick, and Command R+, while maintaining broad compatibility across cloud environments.,
    created=1746627341'''
)

MISTRAL_NEMO = ModelInfo(
    str_identifier="mistralai/mistral-nemo",
    price_in=1e-08,
    price_out=2.9e-08,
    creator="mistralai",
    description='''A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.

The model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and Hindi.

It supports function calling and is released under the Apache 2.0 license.,
    created=1721347200'''
)

MISTRAL_NEMO_FREE = ModelInfo(
    str_identifier="mistralai/mistral-nemo:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.

The model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and Hindi.

It supports function calling and is released under the Apache 2.0 license.,
    created=1721347200'''
)

MISTRAL_SABA = ModelInfo(
    str_identifier="mistralai/mistral-saba",
    price_in=2e-07,
    price_out=6e-07,
    creator="mistralai",
    description='''Mistral Saba is a 24B-parameter language model specifically designed for the Middle East and South Asia, delivering accurate and contextually relevant responses while maintaining efficient performance. Trained on curated regional datasets, it supports multiple Indian-origin languages—including Tamil and Malayalam—alongside Arabic. This makes it a versatile option for a range of regional and multilingual applications. Read more at the blog post [here](https://mistral.ai/en/news/mistral-saba),
    created=1739803239'''
)

MISTRAL_SMALL = ModelInfo(
    str_identifier="mistralai/mistral-small",
    price_in=2e-07,
    price_out=6e-07,
    creator="mistralai",
    description='''With 22 billion parameters, Mistral Small v24.09 offers a convenient mid-point between (Mistral NeMo 12B)[/mistralai/mistral-nemo] and (Mistral Large 2)[/mistralai/mistral-large], providing a cost-effective solution that can be deployed across various platforms and environments. It has better reasoning, exhibits more capabilities, can produce and reason about code, and is multiligual, supporting English, French, German, Italian, and Spanish.,
    created=1704844800'''
)

MISTRAL_SMALL_24B_INSTRUCT_2501 = ModelInfo(
    str_identifier="mistralai/mistral-small-24b-instruct-2501",
    price_in=5e-08,
    price_out=1.1e-07,
    creator="mistralai",
    description='''Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained and instruction-tuned versions designed for efficient local deployment.

The model achieves 81% accuracy on the MMLU benchmark and performs competitively with larger models like Llama 3.3 70B and Qwen 32B, while operating at three times the speed on equivalent hardware. [Read the blog post about the model here.](https://mistral.ai/news/mistral-small-3/),
    created=1738255409'''
)

MISTRAL_SMALL_24B_INSTRUCT_2501_FREE = ModelInfo(
    str_identifier="mistralai/mistral-small-24b-instruct-2501:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained and instruction-tuned versions designed for efficient local deployment.

The model achieves 81% accuracy on the MMLU benchmark and performs competitively with larger models like Llama 3.3 70B and Qwen 32B, while operating at three times the speed on equivalent hardware. [Read the blog post about the model here.](https://mistral.ai/news/mistral-small-3/),
    created=1738255409'''
)

MISTRAL_SMALL_3_1_24B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mistral-small-3.1-24b-instruct",
    price_in=5e-08,
    price_out=1.5e-07,
    creator="mistralai",
    description='''Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational agents, function calling, long-document comprehension, and privacy-sensitive deployments.,
    created=1742238937'''
)

MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE = ModelInfo(
    str_identifier="mistralai/mistral-small-3.1-24b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational agents, function calling, long-document comprehension, and privacy-sensitive deployments.,
    created=1742238937'''
)

MISTRAL_TINY = ModelInfo(
    str_identifier="mistralai/mistral-tiny",
    price_in=2.5e-07,
    price_out=2.5e-07,
    creator="mistralai",
    description='''Note: This model is being deprecated. Recommended replacement is the newer [Ministral 8B](/mistral/ministral-8b)

This model is currently powered by Mistral-7B-v0.2, and incorporates a "better" fine-tuning than [Mistral 7B](/models/mistralai/mistral-7b-instruct-v0.1), inspired by community work. It's best used for large batch processing tasks where cost is a significant factor but reasoning capabilities are not crucial.,
    created=1704844800'''
)

MIXTRAL_8X22B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mixtral-8x22b-instruct",
    price_in=9e-07,
    price_out=9e-07,
    creator="mistralai",
    description='''Mistral's official instruct fine-tuned version of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b). It uses 39B active parameters out of 141B, offering unparalleled cost efficiency for its size. Its strengths include:
- strong math, coding, and reasoning
- large context length (64k)
- fluency in English, French, Italian, German, and Spanish

See benchmarks on the launch announcement [here](https://mistral.ai/news/mixtral-8x22b/).
#moe,
    created=1713312000'''
)

MIXTRAL_8X7B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mixtral-8x7b-instruct",
    price_in=8e-08,
    price_out=2.4e-07,
    creator="mistralai",
    description='''Mixtral 8x7B Instruct is a pretrained generative Sparse Mixture of Experts, by Mistral AI, for chat and instruction use. Incorporates 8 experts (feed-forward networks) for a total of 47 billion parameters.

Instruct model fine-tuned by Mistral. #moe,
    created=1702166400'''
)

PIXTRAL_12B = ModelInfo(
    str_identifier="mistralai/pixtral-12b",
    price_in=1e-07,
    price_out=1e-07,
    creator="mistralai",
    description='''The first multi-modal, text+image-to-text model from Mistral AI. Its weights were launched via torrent: https://x.com/mistralai/status/1833758285167722836.,
    created=1725926400'''
)

PIXTRAL_LARGE_2411 = ModelInfo(
    str_identifier="mistralai/pixtral-large-2411",
    price_in=2e-06,
    price_out=6e-06,
    creator="mistralai",
    description='''Pixtral Large is a 124B parameter, open-weight, multimodal model built on top of [Mistral Large 2](/mistralai/mistral-large-2411). The model is able to understand documents, charts and natural images.

The model is available under the Mistral Research License (MRL) for research and educational use, and the Mistral Commercial License for experimentation, testing, and production for commercial purposes.

,
    created=1731977388'''
)

KIMI_VL_A3B_THINKING_FREE = ModelInfo(
    str_identifier="moonshotai/kimi-vl-a3b-thinking:free",
    price_in=0.0,
    price_out=0.0,
    creator="moonshotai",
    description='''Kimi-VL is a lightweight Mixture-of-Experts vision-language model that activates only 2.8B parameters per step while delivering strong performance on multimodal reasoning and long-context tasks. The Kimi-VL-A3B-Thinking variant, fine-tuned with chain-of-thought and reinforcement learning, excels in math and visual reasoning benchmarks like MathVision, MMMU, and MathVista, rivaling much larger models such as Qwen2.5-VL-7B and Gemma-3-12B. It supports 128K context and high-resolution input via its MoonViT encoder.,
    created=1744304841'''
)

MOONLIGHT_16B_A3B_INSTRUCT_FREE = ModelInfo(
    str_identifier="moonshotai/moonlight-16b-a3b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="moonshotai",
    description='''Moonlight-16B-A3B-Instruct is a 16B-parameter Mixture-of-Experts (MoE) language model developed by Moonshot AI. It is optimized for instruction-following tasks with 3B activated parameters per inference. The model advances the Pareto frontier in performance per FLOP across English, coding, math, and Chinese benchmarks. It outperforms comparable models like Llama3-3B and Deepseek-v2-Lite while maintaining efficient deployment capabilities through Hugging Face integration and compatibility with popular inference engines like vLLM12.,
    created=1740719801'''
)

LLAMA_3_LUMIMAID_70B = ModelInfo(
    str_identifier="neversleep/llama-3-lumimaid-70b",
    price_in=4e-06,
    price_out=6e-06,
    creator="neversleep",
    description='''The NeverSleep team is back, with a Llama 3 70B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessary.

To enhance it's overall intelligence and chat capability, roughly 40% of the training data was not roleplay. This provides a breadth of knowledge to access, while still keeping roleplay as the primary strength.

Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1715817600'''
)

LLAMA_3_LUMIMAID_8B = ModelInfo(
    str_identifier="neversleep/llama-3-lumimaid-8b",
    price_in=2e-07,
    price_out=1.25e-06,
    creator="neversleep",
    description='''The NeverSleep team is back, with a Llama 3 8B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessary.

To enhance it's overall intelligence and chat capability, roughly 40% of the training data was not roleplay. This provides a breadth of knowledge to access, while still keeping roleplay as the primary strength.

Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1714780800'''
)

LLAMA_3_1_LUMIMAID_70B = ModelInfo(
    str_identifier="neversleep/llama-3.1-lumimaid-70b",
    price_in=2.5e-06,
    price_out=3e-06,
    creator="neversleep",
    description='''Lumimaid v0.2 70B is a finetune of [Llama 3.1 70B](/meta-llama/llama-3.1-70b-instruct) with a "HUGE step up dataset wise" compared to Lumimaid v0.1. Sloppy chats output were purged.

Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1729555200'''
)

LLAMA_3_1_LUMIMAID_8B = ModelInfo(
    str_identifier="neversleep/llama-3.1-lumimaid-8b",
    price_in=2e-07,
    price_out=1.25e-06,
    creator="neversleep",
    description='''Lumimaid v0.2 8B is a finetune of [Llama 3.1 8B](/models/meta-llama/llama-3.1-8b-instruct) with a "HUGE step up dataset wise" compared to Lumimaid v0.1. Sloppy chats output were purged.

Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1726358400'''
)

NOROMAID_20B = ModelInfo(
    str_identifier="neversleep/noromaid-20b",
    price_in=1.25e-06,
    price_out=2e-06,
    creator="neversleep",
    description='''A collab between IkariDev and Undi. This merge is suitable for RP, ERP, and general knowledge.

#merge #uncensored,
    created=1700956800'''
)

MN_CELESTE_12B = ModelInfo(
    str_identifier="nothingiisreal/mn-celeste-12b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="nothingiisreal",
    description='''A specialized story writing and roleplaying model based on Mistral's NeMo 12B Instruct. Fine-tuned on curated datasets including Reddit Writing Prompts and Opus Instruct 25K.

This model excels at creative writing, offering improved NSFW capabilities, with smarter and more active narration. It demonstrates remarkable versatility in both SFW and NSFW scenarios, with strong Out of Character (OOC) steering capabilities, allowing fine-tuned control over narrative direction and character behavior.

Check out the model's [HuggingFace page](https://huggingface.co/nothingiisreal/MN-12B-Celeste-V1.9) for details on what parameters and prompts work best!,
    created=1722556800'''
)

DEEPHERMES_3_LLAMA_3_8B_PREVIEW_FREE = ModelInfo(
    str_identifier="nousresearch/deephermes-3-llama-3-8b-preview:free",
    price_in=0.0,
    price_out=0.0,
    creator="nousresearch",
    description='''DeepHermes 3 Preview is the latest version of our flagship Hermes series of LLMs by Nous Research, and one of the first models in the world to unify Reasoning (long chains of thought that improve answer accuracy) and normal LLM response modes into one model. We have also improved LLM annotation, judgement, and function calling.

DeepHermes 3 Preview is one of the first LLM models to unify both "intuitive", traditional mode responses and long chain of thought reasoning responses into a single model, toggled by a system prompt.,
    created=1740719372'''
)

DEEPHERMES_3_MISTRAL_24B_PREVIEW_FREE = ModelInfo(
    str_identifier="nousresearch/deephermes-3-mistral-24b-preview:free",
    price_in=0.0,
    price_out=0.0,
    creator="nousresearch",
    description='''DeepHermes 3 (Mistral 24B Preview) is an instruction-tuned language model by Nous Research based on Mistral-Small-24B, designed for chat, function calling, and advanced multi-turn reasoning. It introduces a dual-mode system that toggles between intuitive chat responses and structured “deep reasoning” mode using special system prompts. Fine-tuned via distillation from R1, it supports structured output (JSON mode) and function call syntax for agent-based applications.

DeepHermes 3 supports a **reasoning toggle via system prompt**, allowing users to switch between fast, intuitive responses and deliberate, multi-step reasoning. When activated with the following specific system instruction, the model enters a *"deep thinking"* mode—generating extended chains of thought wrapped in `<think></think>` tags before delivering a final answer. 

System Prompt: You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
,
    created=1746830904'''
)

HERMES_2_PRO_LLAMA_3_8B = ModelInfo(
    str_identifier="nousresearch/hermes-2-pro-llama-3-8b",
    price_in=2.5e-08,
    price_out=4e-08,
    creator="nousresearch",
    description='''Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house.,
    created=1716768000'''
)

HERMES_3_LLAMA_3_1_405B = ModelInfo(
    str_identifier="nousresearch/hermes-3-llama-3.1-405b",
    price_in=7e-07,
    price_out=8e-07,
    creator="nousresearch",
    description='''Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.

Hermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.

The Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills.

Hermes 3 is competitive, if not superior, to Llama-3.1 Instruct models at general capabilities, with varying strengths and weaknesses attributable between the two.,
    created=1723766400'''
)

HERMES_3_LLAMA_3_1_70B = ModelInfo(
    str_identifier="nousresearch/hermes-3-llama-3.1-70b",
    price_in=1.2e-07,
    price_out=3e-07,
    creator="nousresearch",
    description='''Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.

Hermes 3 70B is a competitive, if not superior finetune of the [Llama-3.1 70B foundation model](/models/meta-llama/llama-3.1-70b-instruct), focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.

The Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills.,
    created=1723939200'''
)

NOUS_HERMES_2_MIXTRAL_8X7B_DPO = ModelInfo(
    str_identifier="nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
    price_in=6e-07,
    price_out=6e-07,
    creator="nousresearch",
    description='''Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained over the [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b).

The model was trained on over 1,000,000 entries of primarily [GPT-4](/models/openai/gpt-4) generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks.

#moe,
    created=1705363200'''
)

LLAMA_3_1_NEMOTRON_70B_INSTRUCT = ModelInfo(
    str_identifier="nvidia/llama-3.1-nemotron-70b-instruct",
    price_in=1.2e-07,
    price_out=3e-07,
    creator="nvidia",
    description='''NVIDIA's Llama 3.1 Nemotron 70B is a language model designed for generating precise and useful responses. Leveraging [Llama 3.1 70B](/models/meta-llama/llama-3.1-70b-instruct) architecture and Reinforcement Learning from Human Feedback (RLHF), it excels in automatic alignment benchmarks. This model is tailored for applications requiring high accuracy in helpfulness and response generation, suitable for diverse user queries across multiple domains.

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1728950400'''
)

LLAMA_3_1_NEMOTRON_ULTRA_253B_V1 = ModelInfo(
    str_identifier="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    price_in=6e-07,
    price_out=1.8e-06,
    creator="nvidia",
    description='''Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for advanced reasoning, human-interactive chat, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta’s Llama-3.1-405B-Instruct, it has been significantly customized using Neural Architecture Search (NAS), resulting in enhanced efficiency, reduced memory usage, and improved inference latency. The model supports a context length of up to 128K tokens and can operate efficiently on an 8x NVIDIA H100 node.

Note: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more.,
    created=1744115059'''
)

LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE = ModelInfo(
    str_identifier="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    price_in=0.0,
    price_out=0.0,
    creator="nvidia",
    description='''Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for advanced reasoning, human-interactive chat, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta’s Llama-3.1-405B-Instruct, it has been significantly customized using Neural Architecture Search (NAS), resulting in enhanced efficiency, reduced memory usage, and improved inference latency. The model supports a context length of up to 128K tokens and can operate efficiently on an 8x NVIDIA H100 node.

Note: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more.,
    created=1744115059'''
)

LLAMA_3_3_NEMOTRON_SUPER_49B_V1 = ModelInfo(
    str_identifier="nvidia/llama-3.3-nemotron-super-49b-v1",
    price_in=1.3e-07,
    price_out=4e-07,
    creator="nvidia",
    description='''Llama-3.3-Nemotron-Super-49B-v1 is a large language model (LLM) optimized for advanced reasoning, conversational interactions, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta's Llama-3.3-70B-Instruct, it employs a Neural Architecture Search (NAS) approach, significantly enhancing efficiency and reducing memory requirements. This allows the model to support a context length of up to 128K tokens and fit efficiently on single high-performance GPUs, such as NVIDIA H200.

Note: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more.,
    created=1744119494'''
)

LLAMA_3_3_NEMOTRON_SUPER_49B_V1_FREE = ModelInfo(
    str_identifier="nvidia/llama-3.3-nemotron-super-49b-v1:free",
    price_in=0.0,
    price_out=0.0,
    creator="nvidia",
    description='''Llama-3.3-Nemotron-Super-49B-v1 is a large language model (LLM) optimized for advanced reasoning, conversational interactions, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta's Llama-3.3-70B-Instruct, it employs a Neural Architecture Search (NAS) approach, significantly enhancing efficiency and reducing memory requirements. This allows the model to support a context length of up to 128K tokens and fit efficiently on single high-performance GPUs, such as NVIDIA H200.

Note: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more.,
    created=1744119494'''
)

OLYMPICCODER_32B_FREE = ModelInfo(
    str_identifier="open-r1/olympiccoder-32b:free",
    price_in=0.0,
    price_out=0.0,
    creator="open-r1",
    description='''OlympicCoder-32B is a high-performing open-source model fine-tuned using the CodeForces-CoTs dataset, containing approximately 100,000 chain-of-thought programming samples. It excels at complex competitive programming benchmarks, such as IOI 2024 and Codeforces-style challenges, frequently surpassing state-of-the-art closed-source models. OlympicCoder-32B provides advanced reasoning, coherent multi-step problem-solving, and robust code generation capabilities, demonstrating significant potential for olympiad-level competitive programming applications.,
    created=1742077228'''
)

CHATGPT_4O_LATEST = ModelInfo(
    str_identifier="openai/chatgpt-4o-latest",
    price_in=5e-06,
    price_out=1.5e-05,
    creator="openai",
    description='''OpenAI ChatGPT 4o is continually updated by OpenAI to point to the current version of GPT-4o used by ChatGPT. It therefore differs slightly from the API version of [GPT-4o](/models/openai/gpt-4o) in that it has additional RLHF. It is intended for research and evaluation.

OpenAI notes that this model is not suited for production use-cases as it may be removed or redirected to another model in the future.,
    created=1723593600'''
)

CODEX_MINI = ModelInfo(
    str_identifier="openai/codex-mini",
    price_in=1.5e-06,
    price_out=6e-06,
    creator="openai",
    description='''codex-mini-latest is a fine-tuned version of o4-mini specifically for use in Codex CLI. For direct use in the API, we recommend starting with gpt-4.1.,
    created=1747409761'''
)

GPT_3_5_TURBO = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="openai",
    description='''GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.

Training data up to Sep 2021.,
    created=1685232000'''
)

GPT_3_5_TURBO_0125 = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo-0125",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="openai",
    description='''The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Sep 2021.

This version has a higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls.,
    created=1685232000'''
)

GPT_3_5_TURBO_0613 = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo-0613",
    price_in=1e-06,
    price_out=2e-06,
    creator="openai",
    description='''GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.

Training data up to Sep 2021.,
    created=1706140800'''
)

GPT_3_5_TURBO_1106 = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo-1106",
    price_in=1e-06,
    price_out=2e-06,
    creator="openai",
    description='''An older GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Sep 2021.,
    created=1699228800'''
)

GPT_3_5_TURBO_16K = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo-16k",
    price_in=3e-06,
    price_out=4e-06,
    creator="openai",
    description='''This model offers four times the context length of gpt-3.5-turbo, allowing it to support approximately 20 pages of text in a single request at a higher cost. Training data: up to Sep 2021.,
    created=1693180800'''
)

GPT_3_5_TURBO_INSTRUCT = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo-instruct",
    price_in=1.5e-06,
    price_out=2e-06,
    creator="openai",
    description='''This model is a variant of GPT-3.5 Turbo tuned for instructional prompts and omitting chat-related optimizations. Training data: up to Sep 2021.,
    created=1695859200'''
)

GPT_4 = ModelInfo(
    str_identifier="openai/gpt-4",
    price_in=3e-05,
    price_out=6e-05,
    creator="openai",
    description='''OpenAI's flagship model, GPT-4 is a large-scale multimodal language model capable of solving difficult problems with greater accuracy than previous models due to its broader general knowledge and advanced reasoning capabilities. Training data: up to Sep 2021.,
    created=1685232000'''
)

GPT_4_0314 = ModelInfo(
    str_identifier="openai/gpt-4-0314",
    price_in=3e-05,
    price_out=6e-05,
    creator="openai",
    description='''GPT-4-0314 is the first version of GPT-4 released, with a context length of 8,192 tokens, and was supported until June 14. Training data: up to Sep 2021.,
    created=1685232000'''
)

GPT_4_1106_PREVIEW = ModelInfo(
    str_identifier="openai/gpt-4-1106-preview",
    price_in=1e-05,
    price_out=3e-05,
    creator="openai",
    description='''The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.

Training data: up to April 2023.,
    created=1699228800'''
)

GPT_4_TURBO = ModelInfo(
    str_identifier="openai/gpt-4-turbo",
    price_in=1e-05,
    price_out=3e-05,
    creator="openai",
    description='''The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.

Training data: up to December 2023.,
    created=1712620800'''
)

GPT_4_TURBO_PREVIEW = ModelInfo(
    str_identifier="openai/gpt-4-turbo-preview",
    price_in=1e-05,
    price_out=3e-05,
    creator="openai",
    description='''The preview GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Dec 2023.

**Note:** heavily rate limited by OpenAI while in preview.,
    created=1706140800'''
)

GPT_4_1 = ModelInfo(
    str_identifier="openai/gpt-4.1",
    price_in=2e-06,
    price_out=8e-06,
    creator="openai",
    description='''GPT-4.1 is a flagship large language model optimized for advanced instruction following, real-world software engineering, and long-context reasoning. It supports a 1 million token context window and outperforms GPT-4o and GPT-4.5 across coding (54.6% SWE-bench Verified), instruction compliance (87.4% IFEval), and multimodal understanding benchmarks. It is tuned for precise code diffs, agent reliability, and high recall in large document contexts, making it ideal for agents, IDE tooling, and enterprise knowledge retrieval.,
    created=1744651385'''
)

GPT_4_1_MINI = ModelInfo(
    str_identifier="openai/gpt-4.1-mini",
    price_in=4e-07,
    price_out=1.6e-06,
    creator="openai",
    description='''GPT-4.1 Mini is a mid-sized model delivering performance competitive with GPT-4o at substantially lower latency and cost. It retains a 1 million token context window and scores 45.1% on hard instruction evals, 35.8% on MultiChallenge, and 84.1% on IFEval. Mini also shows strong coding ability (e.g., 31.6% on Aider’s polyglot diff benchmark) and vision understanding, making it suitable for interactive applications with tight performance constraints.,
    created=1744651381'''
)

GPT_4_1_NANO = ModelInfo(
    str_identifier="openai/gpt-4.1-nano",
    price_in=1e-07,
    price_out=4e-07,
    creator="openai",
    description='''For tasks that demand low latency, GPT‑4.1 nano is the fastest and cheapest model in the GPT-4.1 series. It delivers exceptional performance at a small size with its 1 million token context window, and scores 80.1% on MMLU, 50.3% on GPQA, and 9.8% on Aider polyglot coding – even higher than GPT‑4o mini. It’s ideal for tasks like classification or autocompletion.,
    created=1744651369'''
)

GPT_4_5_PREVIEW = ModelInfo(
    str_identifier="openai/gpt-4.5-preview",
    price_in=7.5e-05,
    price_out=0.00015,
    creator="openai",
    description='''GPT-4.5 (Preview) is a research preview of OpenAI’s latest language model, designed to advance capabilities in reasoning, creativity, and multi-turn conversation. It builds on previous iterations with improvements in world knowledge, contextual coherence, and the ability to follow user intent more effectively.

The model demonstrates enhanced performance in tasks that require open-ended thinking, problem-solving, and communication. Early testing suggests it is better at generating nuanced responses, maintaining long-context coherence, and reducing hallucinations compared to earlier versions.

This research preview is intended to help evaluate GPT-4.5’s strengths and limitations in real-world use cases as OpenAI continues to refine and develop future models. Read more at the [blog post here.](https://openai.com/index/introducing-gpt-4-5/),
    created=1740687810'''
)

GPT_4O = ModelInfo(
    str_identifier="openai/gpt-4o",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.

For benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)

#multimodal,
    created=1715558400'''
)

GPT_4O_2024_05_13 = ModelInfo(
    str_identifier="openai/gpt-4o-2024-05-13",
    price_in=5e-06,
    price_out=1.5e-05,
    creator="openai",
    description='''GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.

For benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)

#multimodal,
    created=1715558400'''
)

GPT_4O_2024_08_06 = ModelInfo(
    str_identifier="openai/gpt-4o-2024-08-06",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="openai",
    description='''The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai.com/index/introducing-structured-outputs-in-the-api/).

GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.

For benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209),
    created=1722902400'''
)

GPT_4O_2024_11_20 = ModelInfo(
    str_identifier="openai/gpt-4o-2024-11-20",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="openai",
    description='''The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, engaging, and tailored writing to improve relevance & readability. It’s also better at working with uploaded files, providing deeper insights & more thorough responses.

GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.,
    created=1732127594'''
)

GPT_4O_MINI = ModelInfo(
    str_identifier="openai/gpt-4o-mini",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="openai",
    description='''GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.

As their most advanced small model, it is many multiples more affordable than other recent frontier models, and more than 60% cheaper than [GPT-3.5 Turbo](/models/openai/gpt-3.5-turbo). It maintains SOTA intelligence, while being significantly more cost-effective.

GPT-4o mini achieves an 82% score on MMLU and presently ranks higher than GPT-4 on chat preferences [common leaderboards](https://arena.lmsys.org/).

Check out the [launch announcement](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) to learn more.

#multimodal,
    created=1721260800'''
)

GPT_4O_MINI_2024_07_18 = ModelInfo(
    str_identifier="openai/gpt-4o-mini-2024-07-18",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="openai",
    description='''GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.

As their most advanced small model, it is many multiples more affordable than other recent frontier models, and more than 60% cheaper than [GPT-3.5 Turbo](/models/openai/gpt-3.5-turbo). It maintains SOTA intelligence, while being significantly more cost-effective.

GPT-4o mini achieves an 82% score on MMLU and presently ranks higher than GPT-4 on chat preferences [common leaderboards](https://arena.lmsys.org/).

Check out the [launch announcement](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) to learn more.

#multimodal,
    created=1721260800'''
)

GPT_4O_MINI_SEARCH_PREVIEW = ModelInfo(
    str_identifier="openai/gpt-4o-mini-search-preview",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="openai",
    description='''GPT-4o mini Search Preview is a specialized model for web search in Chat Completions. It is trained to understand and execute web search queries.,
    created=1741818122'''
)

GPT_4O_SEARCH_PREVIEW = ModelInfo(
    str_identifier="openai/gpt-4o-search-preview",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-4o Search Previewis a specialized model for web search in Chat Completions. It is trained to understand and execute web search queries.,
    created=1741817949'''
)

GPT_4O_EXTENDED = ModelInfo(
    str_identifier="openai/gpt-4o:extended",
    price_in=6e-06,
    price_out=1.8e-05,
    creator="openai",
    description='''GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.

For benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)

#multimodal,
    created=1715558400'''
)

O1 = ModelInfo(
    str_identifier="openai/o1",
    price_in=1.5e-05,
    price_out=6e-05,
    creator="openai",
    description='''The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding. The o1 model series is trained with large-scale reinforcement learning to reason using chain of thought. 

The o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).
,
    created=1734459999'''
)

O1_MINI = ModelInfo(
    str_identifier="openai/o1-mini",
    price_in=1.1e-06,
    price_out=4.4e-06,
    creator="openai",
    description='''The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.

The o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).

Note: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited.,
    created=1726099200'''
)

O1_MINI_2024_09_12 = ModelInfo(
    str_identifier="openai/o1-mini-2024-09-12",
    price_in=1.1e-06,
    price_out=4.4e-06,
    creator="openai",
    description='''The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.

The o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).

Note: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited.,
    created=1726099200'''
)

O1_PREVIEW = ModelInfo(
    str_identifier="openai/o1-preview",
    price_in=1.5e-05,
    price_out=6e-05,
    creator="openai",
    description='''The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.

The o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).

Note: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited.,
    created=1726099200'''
)

O1_PREVIEW_2024_09_12 = ModelInfo(
    str_identifier="openai/o1-preview-2024-09-12",
    price_in=1.5e-05,
    price_out=6e-05,
    creator="openai",
    description='''The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.

The o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).

Note: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited.,
    created=1726099200'''
)

O1_PRO = ModelInfo(
    str_identifier="openai/o1-pro",
    price_in=0.00015,
    price_out=0.0006,
    creator="openai",
    description='''The o1 series of models are trained with reinforcement learning to think before they answer and perform complex reasoning. The o1-pro model uses more compute to think harder and provide consistently better answers.,
    created=1742423211'''
)

O3 = ModelInfo(
    str_identifier="openai/o3",
    price_in=1e-05,
    price_out=4e-05,
    creator="openai",
    description='''o3 is a well-rounded and powerful model across domains. It sets a new standard for math, science, coding, and visual reasoning tasks. It also excels at technical writing and instruction-following. Use it to think through multi-step problems that involve analysis across text, code, and images. Note that BYOK is required for this model. Set up here: https://openrouter.ai/settings/integrations,
    created=1744823457'''
)

O3_MINI = ModelInfo(
    str_identifier="openai/o3-mini",
    price_in=1.1e-06,
    price_out=4.4e-06,
    creator="openai",
    description='''OpenAI o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding.

This model supports the `reasoning_effort` parameter, which can be set to "high", "medium", or "low" to control the thinking time of the model. The default is "medium". OpenRouter also offers the model slug `openai/o3-mini-high` to default the parameter to "high".

The model features three adjustable reasoning effort levels and supports key developer capabilities including function calling, structured outputs, and streaming, though it does not include vision processing capabilities.

The model demonstrates significant improvements over its predecessor, with expert testers preferring its responses 56% of the time and noting a 39% reduction in major errors on complex questions. With medium reasoning effort settings, o3-mini matches the performance of the larger o1 model on challenging reasoning evaluations like AIME and GPQA, while maintaining lower latency and cost.,
    created=1738351721'''
)

O3_MINI_HIGH = ModelInfo(
    str_identifier="openai/o3-mini-high",
    price_in=1.1e-06,
    price_out=4.4e-06,
    creator="openai",
    description='''OpenAI o3-mini-high is the same model as [o3-mini](/openai/o3-mini) with reasoning_effort set to high. 

o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding. The model features three adjustable reasoning effort levels and supports key developer capabilities including function calling, structured outputs, and streaming, though it does not include vision processing capabilities.

The model demonstrates significant improvements over its predecessor, with expert testers preferring its responses 56% of the time and noting a 39% reduction in major errors on complex questions. With medium reasoning effort settings, o3-mini matches the performance of the larger o1 model on challenging reasoning evaluations like AIME and GPQA, while maintaining lower latency and cost.,
    created=1739372611'''
)

O4_MINI = ModelInfo(
    str_identifier="openai/o4-mini",
    price_in=1.1e-06,
    price_out=4.4e-06,
    creator="openai",
    description='''OpenAI o4-mini is a compact reasoning model in the o-series, optimized for fast, cost-efficient performance while retaining strong multimodal and agentic capabilities. It supports tool use and demonstrates competitive reasoning and coding performance across benchmarks like AIME (99.5% with Python) and SWE-bench, outperforming its predecessor o3-mini and even approaching o3 in some domains.

Despite its smaller size, o4-mini exhibits high accuracy in STEM tasks, visual problem solving (e.g., MathVista, MMMU), and code editing. It is especially well-suited for high-throughput scenarios where latency or cost is critical. Thanks to its efficient architecture and refined reinforcement learning training, o4-mini can chain tools, generate structured outputs, and solve multi-step tasks with minimal delay—often in under a minute.,
    created=1744820942'''
)

O4_MINI_HIGH = ModelInfo(
    str_identifier="openai/o4-mini-high",
    price_in=1.1e-06,
    price_out=4.4e-06,
    creator="openai",
    description='''OpenAI o4-mini-high is the same model as [o4-mini](/openai/o4-mini) with reasoning_effort set to high. 

OpenAI o4-mini is a compact reasoning model in the o-series, optimized for fast, cost-efficient performance while retaining strong multimodal and agentic capabilities. It supports tool use and demonstrates competitive reasoning and coding performance across benchmarks like AIME (99.5% with Python) and SWE-bench, outperforming its predecessor o3-mini and even approaching o3 in some domains.

Despite its smaller size, o4-mini exhibits high accuracy in STEM tasks, visual problem solving (e.g., MathVista, MMMU), and code editing. It is especially well-suited for high-throughput scenarios where latency or cost is critical. Thanks to its efficient architecture and refined reinforcement learning training, o4-mini can chain tools, generate structured outputs, and solve multi-step tasks with minimal delay—often in under a minute.,
    created=1744824212'''
)

INTERNVL3_14B_FREE = ModelInfo(
    str_identifier="opengvlab/internvl3-14b:free",
    price_in=0.0,
    price_out=0.0,
    creator="opengvlab",
    description='''The 14b version of the InternVL3 series. An advanced multimodal large language model (MLLM) series that demonstrates superior overall performance. Compared to InternVL 2.5, InternVL3 exhibits superior multimodal perception and reasoning capabilities, while further extending its multimodal capabilities to encompass tool usage, GUI agents, industrial image analysis, 3D vision perception, and more.,
    created=1746021355'''
)

INTERNVL3_2B_FREE = ModelInfo(
    str_identifier="opengvlab/internvl3-2b:free",
    price_in=0.0,
    price_out=0.0,
    creator="opengvlab",
    description='''The 2b version of the InternVL3 series, for an even higher inference speed and very reasonable performance. An advanced multimodal large language model (MLLM) series that demonstrates superior overall performance. Compared to InternVL 2.5, InternVL3 exhibits superior multimodal perception and reasoning capabilities, while further extending its multimodal capabilities to encompass tool usage, GUI agents, industrial image analysis, 3D vision perception, and more.,
    created=1746019807'''
)

AUTO = ModelInfo(
    str_identifier="openrouter/auto",
    price_in=-1.0,
    price_out=-1.0,
    creator="openrouter",
    description='''Your prompt will be processed by a meta-model and routed to one of dozens of models (see below), optimizing for the best possible output.

To see which model was used, visit [Activity](/activity), or read the `model` attribute of the response. Your response will be priced at the same rate as the routed model.

The meta-model is powered by [Not Diamond](https://docs.notdiamond.ai/docs/how-not-diamond-works). Learn more in our [docs](/docs/model-routing).

Requests will be routed to the following models:
- [openai/gpt-4o-2024-08-06](/openai/gpt-4o-2024-08-06)
- [openai/gpt-4o-2024-05-13](/openai/gpt-4o-2024-05-13)
- [openai/gpt-4o-mini-2024-07-18](/openai/gpt-4o-mini-2024-07-18)
- [openai/chatgpt-4o-latest](/openai/chatgpt-4o-latest)
- [openai/o1-preview-2024-09-12](/openai/o1-preview-2024-09-12)
- [openai/o1-mini-2024-09-12](/openai/o1-mini-2024-09-12)
- [anthropic/claude-3.5-sonnet](/anthropic/claude-3.5-sonnet)
- [anthropic/claude-3.5-haiku](/anthropic/claude-3.5-haiku)
- [anthropic/claude-3-opus](/anthropic/claude-3-opus)
- [anthropic/claude-2.1](/anthropic/claude-2.1)
- [google/gemini-pro-1.5](/google/gemini-pro-1.5)
- [google/gemini-flash-1.5](/google/gemini-flash-1.5)
- [mistralai/mistral-large-2407](/mistralai/mistral-large-2407)
- [mistralai/mistral-nemo](/mistralai/mistral-nemo)
- [deepseek/deepseek-r1](/deepseek/deepseek-r1)
- [meta-llama/llama-3.1-70b-instruct](/meta-llama/llama-3.1-70b-instruct)
- [meta-llama/llama-3.1-405b-instruct](/meta-llama/llama-3.1-405b-instruct)
- [mistralai/mixtral-8x22b-instruct](/mistralai/mixtral-8x22b-instruct)
- [cohere/command-r-plus](/cohere/command-r-plus)
- [cohere/command-r](/cohere/command-r),
    created=1699401600'''
)

LLAMA_3_1_SONAR_LARGE_128K_ONLINE = ModelInfo(
    str_identifier="perplexity/llama-3.1-sonar-large-128k-online",
    price_in=1e-06,
    price_out=1e-06,
    creator="perplexity",
    description='''Llama 3.1 Sonar is Perplexity's latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and performance.

This is the online version of the [offline chat model](/models/perplexity/llama-3.1-sonar-large-128k-chat). It is focused on delivering helpful, up-to-date, and factual responses. #online,
    created=1722470400'''
)

LLAMA_3_1_SONAR_SMALL_128K_ONLINE = ModelInfo(
    str_identifier="perplexity/llama-3.1-sonar-small-128k-online",
    price_in=2e-07,
    price_out=2e-07,
    creator="perplexity",
    description='''Llama 3.1 Sonar is Perplexity's latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and performance.

This is the online version of the [offline chat model](/models/perplexity/llama-3.1-sonar-small-128k-chat). It is focused on delivering helpful, up-to-date, and factual responses. #online,
    created=1722470400'''
)

R1_1776 = ModelInfo(
    str_identifier="perplexity/r1-1776",
    price_in=2e-06,
    price_out=8e-06,
    creator="perplexity",
    description='''R1 1776 is a version of DeepSeek-R1 that has been post-trained to remove censorship constraints related to topics restricted by the Chinese government. The model retains its original reasoning capabilities while providing direct responses to a wider range of queries. R1 1776 is an offline chat model that does not use the perplexity search subsystem.

The model was tested on a multilingual dataset of over 1,000 examples covering sensitive topics to measure its likelihood of refusal or overly filtered responses. [Evaluation Results](https://cdn-uploads.huggingface.co/production/uploads/675c8332d01f593dc90817f5/GiN2VqC5hawUgAGJ6oHla.png) Its performance on math and reasoning benchmarks remains similar to the base R1 model. [Reasoning Performance](https://cdn-uploads.huggingface.co/production/uploads/675c8332d01f593dc90817f5/n4Z9Byqp2S7sKUvCvI40R.png)

Read more on the [Blog Post](https://perplexity.ai/hub/blog/open-sourcing-r1-1776),
    created=1740004929'''
)

SONAR = ModelInfo(
    str_identifier="perplexity/sonar",
    price_in=1e-06,
    price_out=1e-06,
    creator="perplexity",
    description='''Sonar is lightweight, affordable, fast, and simple to use — now featuring citations and the ability to customize sources. It is designed for companies seeking to integrate lightweight question-and-answer features optimized for speed.,
    created=1738013808'''
)

SONAR_DEEP_RESEARCH = ModelInfo(
    str_identifier="perplexity/sonar-deep-research",
    price_in=2e-06,
    price_out=8e-06,
    creator="perplexity",
    description='''Sonar Deep Research is a research-focused model designed for multi-step retrieval, synthesis, and reasoning across complex topics. It autonomously searches, reads, and evaluates sources, refining its approach as it gathers information. This enables comprehensive report generation across domains like finance, technology, health, and current events.

Notes on Pricing ([Source](https://docs.perplexity.ai/guides/pricing#detailed-pricing-breakdown-for-sonar-deep-research)) 
- Input tokens comprise of Prompt tokens (user prompt) + Citation tokens (these are processed tokens from running searches)
- Deep Research runs multiple searches to conduct exhaustive research. Searches are priced at $5/1000 searches. A request that does 30 searches will cost $0.15 in this step.
- Reasoning is a distinct step in Deep Research since it does extensive automated reasoning through all the material it gathers during its research phase. Reasoning tokens here are a bit different than the CoTs in the answer - these are tokens that we use to reason through the research material prior to generating the outputs via the CoTs. Reasoning tokens are priced at $3/1M tokens,
    created=1741311246'''
)

SONAR_PRO = ModelInfo(
    str_identifier="perplexity/sonar-pro",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="perplexity",
    description='''Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](https://docs.perplexity.ai/guides/pricing#detailed-pricing-breakdown-for-sonar-reasoning-pro-and-sonar-pro)

For enterprises seeking more advanced capabilities, the Sonar Pro API can handle in-depth, multi-step queries with added extensibility, like double the number of citations per search as Sonar on average. Plus, with a larger context window, it can handle longer and more nuanced searches and follow-up questions. ,
    created=1741312423'''
)

SONAR_REASONING = ModelInfo(
    str_identifier="perplexity/sonar-reasoning",
    price_in=1e-06,
    price_out=5e-06,
    creator="perplexity",
    description='''Sonar Reasoning is a reasoning model provided by Perplexity based on [DeepSeek R1](/deepseek/deepseek-r1).

It allows developers to utilize long chain of thought with built-in web search. Sonar Reasoning is uncensored and hosted in US datacenters. ,
    created=1738131107'''
)

SONAR_REASONING_PRO = ModelInfo(
    str_identifier="perplexity/sonar-reasoning-pro",
    price_in=2e-06,
    price_out=8e-06,
    creator="perplexity",
    description='''Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](https://docs.perplexity.ai/guides/pricing#detailed-pricing-breakdown-for-sonar-reasoning-pro-and-sonar-pro)

Sonar Reasoning Pro is a premier reasoning model powered by DeepSeek R1 with Chain of Thought (CoT). Designed for advanced use cases, it supports in-depth, multi-step queries with a larger context window and can surface more citations per search, enabling more comprehensive and extensible responses.,
    created=1741313308'''
)

MYTHALION_13B = ModelInfo(
    str_identifier="pygmalionai/mythalion-13b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="pygmalionai",
    description='''A blend of the new Pygmalion-13b and MythoMax. #merge,
    created=1693612800'''
)

QWEN_2_72B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen-2-72b-instruct",
    price_in=9e-07,
    price_out=9e-07,
    creator="qwen",
    description='''Qwen2 72B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.

It features SwiGLU activation, attention QKV bias, and group query attention. It is pretrained on extensive data with supervised finetuning and direct preference optimization.

For more details, see this [blog post](https://qwenlm.github.io/blog/qwen2/) and [GitHub repo](https://github.com/QwenLM/Qwen2).

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1717718400'''
)

QWEN_2_5_72B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen-2.5-72b-instruct",
    price_in=1.2e-07,
    price_out=3.9e-07,
    creator="qwen",
    description='''Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.

- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.

- Long-context Support up to 128K tokens and can generate up to 8K tokens.

- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1726704000'''
)

QWEN_2_5_72B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen-2.5-72b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.

- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.

- Long-context Support up to 128K tokens and can generate up to 8K tokens.

- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1726704000'''
)

QWEN_2_5_7B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen-2.5-7b-instruct",
    price_in=4e-08,
    price_out=1e-07,
    creator="qwen",
    description='''Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.

- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.

- Long-context Support up to 128K tokens and can generate up to 8K tokens.

- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1729036800'''
)

QWEN_2_5_7B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen-2.5-7b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:

- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.

- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.

- Long-context Support up to 128K tokens and can generate up to 8K tokens.

- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1729036800'''
)

QWEN_2_5_CODER_32B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen-2.5-coder-32b-instruct",
    price_in=6e-08,
    price_out=1.5e-07,
    creator="qwen",
    description='''Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:

- Significantly improvements in **code generation**, **code reasoning** and **code fixing**. 
- A more comprehensive foundation for real-world applications such as **Code Agents**. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.

To read more about its evaluation results, check out [Qwen 2.5 Coder's blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/).,
    created=1731368400'''
)

QWEN_2_5_CODER_32B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen-2.5-coder-32b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:

- Significantly improvements in **code generation**, **code reasoning** and **code fixing**. 
- A more comprehensive foundation for real-world applications such as **Code Agents**. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.

To read more about its evaluation results, check out [Qwen 2.5 Coder's blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/).,
    created=1731368400'''
)

QWEN_2_5_VL_7B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen-2.5-vl-7b-instruct",
    price_in=2e-07,
    price_out=2e-07,
    creator="qwen",
    description='''Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:

- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

- Understanding videos of 20min+: Qwen2.5-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.

- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

For more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1724803200'''
)

QWEN_2_5_VL_7B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen-2.5-vl-7b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:

- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

- Understanding videos of 20min+: Qwen2.5-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.

- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

For more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1724803200'''
)

QWEN_MAX = ModelInfo(
    str_identifier="qwen/qwen-max",
    price_in=1.6e-06,
    price_out=6.4e-06,
    creator="qwen",
    description='''Qwen-Max, based on Qwen2.5, provides the best inference performance among [Qwen models](/qwen), especially for complex multi-step tasks. It's a large-scale MoE model that has been pretrained on over 20 trillion tokens and further post-trained with curated Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) methodologies. The parameter count is unknown.,
    created=1738402289'''
)

QWEN_PLUS = ModelInfo(
    str_identifier="qwen/qwen-plus",
    price_in=4e-07,
    price_out=1.2e-06,
    creator="qwen",
    description='''Qwen-Plus, based on the Qwen2.5 foundation model, is a 131K context model with a balanced performance, speed, and cost combination.,
    created=1738409840'''
)

QWEN_TURBO = ModelInfo(
    str_identifier="qwen/qwen-turbo",
    price_in=5e-08,
    price_out=2e-07,
    creator="qwen",
    description='''Qwen-Turbo, based on Qwen2.5, is a 1M context model that provides fast speed and low cost, suitable for simple tasks.,
    created=1738410974'''
)

QWEN_VL_MAX = ModelInfo(
    str_identifier="qwen/qwen-vl-max",
    price_in=8e-07,
    price_out=3.2e-06,
    creator="qwen",
    description='''Qwen VL Max is a visual understanding model with 7500 tokens context length. It excels in delivering optimal performance for a broader spectrum of complex tasks.
,
    created=1738434304'''
)

QWEN_VL_PLUS = ModelInfo(
    str_identifier="qwen/qwen-vl-plus",
    price_in=2.1e-07,
    price_out=6.3e-07,
    creator="qwen",
    description='''Qwen's Enhanced Large Visual Language Model. Significantly upgraded for detailed recognition capabilities and text recognition abilities, supporting ultra-high pixel resolutions up to millions of pixels and extreme aspect ratios for image input. It delivers significant performance across a broad range of visual tasks.
,
    created=1738731255'''
)

QWEN2_5_VL_32B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-32b-instruct",
    price_in=9e-07,
    price_out=9e-07,
    creator="qwen",
    description='''Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solving capabilities. It excels at visual analysis tasks, including object recognition, textual interpretation within images, and precise event localization in extended videos. Qwen2.5-VL-32B demonstrates state-of-the-art performance across multimodal benchmarks such as MMMU, MathVista, and VideoMME, while maintaining strong reasoning and clarity in text-based tasks like MMLU, mathematical problem-solving, and code generation.,
    created=1742839838'''
)

QWEN2_5_VL_32B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-32b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solving capabilities. It excels at visual analysis tasks, including object recognition, textual interpretation within images, and precise event localization in extended videos. Qwen2.5-VL-32B demonstrates state-of-the-art performance across multimodal benchmarks such as MMMU, MathVista, and VideoMME, while maintaining strong reasoning and clarity in text-based tasks like MMLU, mathematical problem-solving, and code generation.,
    created=1742839838'''
)

QWEN2_5_VL_3B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-3b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5 VL 3B is a multimodal LLM from the Qwen Team with the following key enhancements:

- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

For more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).

Usage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE).,
    created=1743014573'''
)

QWEN2_5_VL_72B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-72b-instruct",
    price_in=2.5e-07,
    price_out=7.5e-07,
    creator="qwen",
    description='''Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within images.,
    created=1738410311'''
)

QWEN2_5_VL_72B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-72b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within images.,
    created=1738410311'''
)

QWEN3_14B = ModelInfo(
    str_identifier="qwen/qwen3-14b",
    price_in=6e-08,
    price_out=2.4e-07,
    creator="qwen",
    description='''Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series, designed for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, programming, and logical inference, and a "non-thinking" mode for general-purpose conversation. The model is fine-tuned for instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling.,
    created=1745876478'''
)

QWEN3_14B_FREE = ModelInfo(
    str_identifier="qwen/qwen3-14b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series, designed for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, programming, and logical inference, and a "non-thinking" mode for general-purpose conversation. The model is fine-tuned for instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling.,
    created=1745876478'''
)

QWEN3_235B_A22B = ModelInfo(
    str_identifier="qwen/qwen3-235b-a22b",
    price_in=1.3e-07,
    price_out=6e-07,
    creator="qwen",
    description='''Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by Qwen, activating 22B parameters per forward pass. It supports seamless switching between a "thinking" mode for complex reasoning, math, and code tasks, and a "non-thinking" mode for general conversational efficiency. The model demonstrates strong reasoning ability, multilingual support (100+ languages and dialects), advanced instruction-following, and agent tool-calling capabilities. It natively handles a 32K token context window and extends up to 131K tokens using YaRN-based scaling.,
    created=1745875757'''
)

QWEN3_235B_A22B_FREE = ModelInfo(
    str_identifier="qwen/qwen3-235b-a22b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by Qwen, activating 22B parameters per forward pass. It supports seamless switching between a "thinking" mode for complex reasoning, math, and code tasks, and a "non-thinking" mode for general conversational efficiency. The model demonstrates strong reasoning ability, multilingual support (100+ languages and dialects), advanced instruction-following, and agent tool-calling capabilities. It natively handles a 32K token context window and extends up to 131K tokens using YaRN-based scaling.,
    created=1745875757'''
)

QWEN3_30B_A3B = ModelInfo(
    str_identifier="qwen/qwen3-30b-a3b",
    price_in=8e-08,
    price_out=2.9e-07,
    creator="qwen",
    description='''Qwen3, the latest generation in the Qwen large language model series, features both dense and mixture-of-experts (MoE) architectures to excel in reasoning, multilingual support, and advanced agent tasks. Its unique ability to switch seamlessly between a thinking mode for complex reasoning and a non-thinking mode for efficient dialogue ensures versatile, high-quality performance.

Significantly outperforming prior models like QwQ and Qwen2.5, Qwen3 delivers superior mathematics, coding, commonsense reasoning, creative writing, and interactive dialogue capabilities. The Qwen3-30B-A3B variant includes 30.5 billion parameters (3.3 billion activated), 48 layers, 128 experts (8 activated per task), and supports up to 131K token contexts with YaRN, setting a new standard among open-source models.,
    created=1745878604'''
)

QWEN3_30B_A3B_FREE = ModelInfo(
    str_identifier="qwen/qwen3-30b-a3b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3, the latest generation in the Qwen large language model series, features both dense and mixture-of-experts (MoE) architectures to excel in reasoning, multilingual support, and advanced agent tasks. Its unique ability to switch seamlessly between a thinking mode for complex reasoning and a non-thinking mode for efficient dialogue ensures versatile, high-quality performance.

Significantly outperforming prior models like QwQ and Qwen2.5, Qwen3 delivers superior mathematics, coding, commonsense reasoning, creative writing, and interactive dialogue capabilities. The Qwen3-30B-A3B variant includes 30.5 billion parameters (3.3 billion activated), 48 layers, 128 experts (8 activated per task), and supports up to 131K token contexts with YaRN, setting a new standard among open-source models.,
    created=1745878604'''
)

QWEN3_32B = ModelInfo(
    str_identifier="qwen/qwen3-32b",
    price_in=1e-07,
    price_out=3e-07,
    creator="qwen",
    description='''Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series, optimized for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, coding, and logical inference, and a "non-thinking" mode for faster, general-purpose conversation. The model demonstrates strong performance in instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling. ,
    created=1745875945'''
)

QWEN3_32B_FREE = ModelInfo(
    str_identifier="qwen/qwen3-32b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series, optimized for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, coding, and logical inference, and a "non-thinking" mode for faster, general-purpose conversation. The model demonstrates strong performance in instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling. ,
    created=1745875945'''
)

QWEN3_8B = ModelInfo(
    str_identifier="qwen/qwen3-8b",
    price_in=3.5e-08,
    price_out=1.38e-07,
    creator="qwen",
    description='''Qwen3-8B is a dense 8.2B parameter causal language model from the Qwen3 series, designed for both reasoning-heavy tasks and efficient dialogue. It supports seamless switching between "thinking" mode for math, coding, and logical inference, and "non-thinking" mode for general conversation. The model is fine-tuned for instruction-following, agent integration, creative writing, and multilingual use across 100+ languages and dialects. It natively supports a 32K token context window and can extend to 131K tokens with YaRN scaling.,
    created=1745876632'''
)

QWEN3_8B_FREE = ModelInfo(
    str_identifier="qwen/qwen3-8b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-8B is a dense 8.2B parameter causal language model from the Qwen3 series, designed for both reasoning-heavy tasks and efficient dialogue. It supports seamless switching between "thinking" mode for math, coding, and logical inference, and "non-thinking" mode for general conversation. The model is fine-tuned for instruction-following, agent integration, creative writing, and multilingual use across 100+ languages and dialects. It natively supports a 32K token context window and can extend to 131K tokens with YaRN scaling.,
    created=1745876632'''
)

QWQ_32B = ModelInfo(
    str_identifier="qwen/qwq-32b",
    price_in=1.5e-07,
    price_out=2e-07,
    creator="qwen",
    description='''QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini.,
    created=1741208814'''
)

QWQ_32B_PREVIEW = ModelInfo(
    str_identifier="qwen/qwq-32b-preview",
    price_in=2e-07,
    price_out=2e-07,
    creator="qwen",
    description='''QwQ-32B-Preview is an experimental research model focused on AI reasoning capabilities developed by the Qwen Team. As a preview release, it demonstrates promising analytical abilities while having several important limitations:

1. **Language Mixing and Code-Switching**: The model may mix languages or switch between them unexpectedly, affecting response clarity.
2. **Recursive Reasoning Loops**: The model may enter circular reasoning patterns, leading to lengthy responses without a conclusive answer.
3. **Safety and Ethical Considerations**: The model requires enhanced safety measures to ensure reliable and secure performance, and users should exercise caution when deploying it.
4. **Performance and Benchmark Limitations**: The model excels in math and coding but has room for improvement in other areas, such as common sense reasoning and nuanced language understanding.

,
    created=1732754541'''
)

QWQ_32B_FREE = ModelInfo(
    str_identifier="qwen/qwq-32b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini.,
    created=1741208814'''
)

SORCERERLM_8X22B = ModelInfo(
    str_identifier="raifle/sorcererlm-8x22b",
    price_in=4.5e-06,
    price_out=4.5e-06,
    creator="raifle",
    description='''SorcererLM is an advanced RP and storytelling model, built as a Low-rank 16-bit LoRA fine-tuned on [WizardLM-2 8x22B](/microsoft/wizardlm-2-8x22b).

- Advanced reasoning and emotional intelligence for engaging and immersive interactions
- Vivid writing capabilities enriched with spatial and contextual awareness
- Enhanced narrative depth, promoting creative and dynamic storytelling,
    created=1731105083'''
)

REKA_FLASH_3_FREE = ModelInfo(
    str_identifier="rekaai/reka-flash-3:free",
    price_in=0.0,
    price_out=0.0,
    creator="rekaai",
    description='''Reka Flash 3 is a general-purpose, instruction-tuned large language model with 21 billion parameters, developed by Reka. It excels at general chat, coding tasks, instruction-following, and function calling. Featuring a 32K context length and optimized through reinforcement learning (RLOO), it provides competitive performance comparable to proprietary models within a smaller parameter footprint. Ideal for low-latency, local, or on-device deployments, Reka Flash 3 is compact, supports efficient quantization (down to 11GB at 4-bit precision), and employs explicit reasoning tags ("<reasoning>") to indicate its internal thought process.

Reka Flash 3 is primarily an English model with limited multilingual understanding capabilities. The model weights are released under the Apache 2.0 license.,
    created=1741812813'''
)

FIMBULVETR_11B_V2 = ModelInfo(
    str_identifier="sao10k/fimbulvetr-11b-v2",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="sao10k",
    description='''Creative writing model, routed with permission. It's fast, it keeps the conversation going, and it stays in character.

If you submit a raw prompt, you can use Alpaca or Vicuna formats.,
    created=1713657600'''
)

L3_EURYALE_70B = ModelInfo(
    str_identifier="sao10k/l3-euryale-70b",
    price_in=1.48e-06,
    price_out=1.48e-06,
    creator="sao10k",
    description='''Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k).

- Better prompt adherence.
- Better anatomy / spatial awareness.
- Adapts much better to unique and custom formatting / reply formats.
- Very creative, lots of unique swipes.
- Is not restrictive during roleplays.,
    created=1718668800'''
)

L3_LUNARIS_8B = ModelInfo(
    str_identifier="sao10k/l3-lunaris-8b",
    price_in=2e-08,
    price_out=5e-08,
    creator="sao10k",
    description='''Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It's a strategic merge of multiple models, designed to balance creativity with improved logic and general knowledge.

Created by [Sao10k](https://huggingface.co/Sao10k), this model aims to offer an improved experience over Stheno v3.2, with enhanced creativity and logical reasoning.

For best results, use with Llama 3 Instruct context template, temperature 1.4, and min_p 0.1.,
    created=1723507200'''
)

L3_1_EURYALE_70B = ModelInfo(
    str_identifier="sao10k/l3.1-euryale-70b",
    price_in=7e-07,
    price_out=8e-07,
    creator="sao10k",
    description='''Euryale L3.1 70B v2.2 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.1](/models/sao10k/l3-euryale-70b).,
    created=1724803200'''
)

L3_3_EURYALE_70B = ModelInfo(
    str_identifier="sao10k/l3.3-euryale-70b",
    price_in=7e-07,
    price_out=8e-07,
    creator="sao10k",
    description='''Euryale L3.3 70B is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.2](/models/sao10k/l3-euryale-70b).,
    created=1734535928'''
)

SARVAM_M = ModelInfo(
    str_identifier="sarvamai/sarvam-m",
    price_in=2.5e-07,
    price_out=7.5e-07,
    creator="sarvamai",
    description='''Sarvam-M is a 24 B-parameter, instruction-tuned derivative of Mistral-Small-3.1-24B-Base-2503, post-trained on English plus eleven major Indic languages (bn, hi, kn, gu, mr, ml, or, pa, ta, te). The model introduces a dual-mode interface: “non-think” for low-latency chat and a optional “think” phase that exposes chain-of-thought tokens for more demanding reasoning, math, and coding tasks. 

Benchmark reports show solid gains versus similarly sized open models on Indic-language QA, GSM-8K math, and SWE-Bench coding, making Sarvam-M a practical general-purpose choice for multilingual conversational agents as well as analytical workloads that mix English, native Indic scripts, or romanized text.,
    created=1748188413'''
)

SARVAM_M_FREE = ModelInfo(
    str_identifier="sarvamai/sarvam-m:free",
    price_in=0.0,
    price_out=0.0,
    creator="sarvamai",
    description='''Sarvam-M is a 24 B-parameter, instruction-tuned derivative of Mistral-Small-3.1-24B-Base-2503, post-trained on English plus eleven major Indic languages (bn, hi, kn, gu, mr, ml, or, pa, ta, te). The model introduces a dual-mode interface: “non-think” for low-latency chat and a optional “think” phase that exposes chain-of-thought tokens for more demanding reasoning, math, and coding tasks. 

Benchmark reports show solid gains versus similarly sized open models on Indic-language QA, GSM-8K math, and SWE-Bench coding, making Sarvam-M a practical general-purpose choice for multilingual conversational agents as well as analytical workloads that mix English, native Indic scripts, or romanized text.,
    created=1748188413'''
)

LLAMA3_1_TYPHOON2_70B_INSTRUCT = ModelInfo(
    str_identifier="scb10x/llama3.1-typhoon2-70b-instruct",
    price_in=8.8e-07,
    price_out=8.8e-07,
    creator="scb10x",
    description='''Llama3.1-Typhoon2-70B-Instruct is a Thai-English instruction-tuned language model with 70 billion parameters, built on Llama 3.1. It demonstrates strong performance across general instruction-following, math, coding, and tool-use tasks, with state-of-the-art results in Thai-specific benchmarks such as IFEval, MT-Bench, and Thai-English code-switching.

The model excels in bilingual reasoning and function-calling scenarios, offering high accuracy across diverse domains. Comparative evaluations show consistent improvements over prior Thai LLMs and other Llama-based baselines. Full results and methodology are available in the [technical report.](https://arxiv.org/abs/2412.13702),
    created=1743196170'''
)

DOBBY_MINI_UNHINGED_PLUS_LLAMA_3_1_8B = ModelInfo(
    str_identifier="sentientagi/dobby-mini-unhinged-plus-llama-3.1-8b",
    price_in=2e-07,
    price_out=2e-07,
    creator="sentientagi",
    description='''Dobby-Mini-Leashed-Llama-3.1-8B and Dobby-Mini-Unhinged-Llama-3.1-8B are language models fine-tuned from Llama-3.1-8B-Instruct. Dobby models have a strong conviction towards personal freedom, decentralization, and all things crypto — even when coerced to speak otherwise. 

Dobby-Mini-Leashed-Llama-3.1-8B and Dobby-Mini-Unhinged-Llama-3.1-8B have their own unique, uhh, personalities. The two versions are being released to be improved using the community’s feedback, which will steer the development of a 70B model.

,
    created=1748885619'''
)

SHISA_V2_LLAMA3_3_70B_FREE = ModelInfo(
    str_identifier="shisa-ai/shisa-v2-llama3.3-70b:free",
    price_in=0.0,
    price_out=0.0,
    creator="shisa-ai",
    description='''Shisa V2 Llama 3.3 70B is a bilingual Japanese-English chat model fine-tuned by Shisa.AI on Meta’s Llama-3.3-70B-Instruct base. It prioritizes Japanese language performance while retaining strong English capabilities. The model was optimized entirely through post-training, using a refined mix of supervised fine-tuning (SFT) and DPO datasets including regenerated ShareGPT-style data, translation tasks, roleplaying conversations, and instruction-following prompts. Unlike earlier Shisa releases, this version avoids tokenizer modifications or extended pretraining.

Shisa V2 70B achieves leading Japanese task performance across a wide range of custom and public benchmarks, including JA MT Bench, ELYZA 100, and Rakuda. It supports a 128K token context length and integrates smoothly with inference frameworks like vLLM and SGLang. While it inherits safety characteristics from its base model, no additional alignment was applied. The model is intended for high-performance bilingual chat, instruction following, and translation tasks across JA/EN.,
    created=1744754858'''
)

MIDNIGHT_ROSE_70B = ModelInfo(
    str_identifier="sophosympatheia/midnight-rose-70b",
    price_in=8e-07,
    price_out=8e-07,
    creator="sophosympatheia",
    description='''A merge with a complex family tree, this model was crafted for roleplaying and storytelling. Midnight Rose is a successor to Rogue Rose and Aurora Nights and improves upon them both. It wants to produce lengthy output by default and is the best creative writing merge produced so far by sophosympatheia.

Descending from earlier versions of Midnight Rose and [Wizard Tulu Dolphin 70B](https://huggingface.co/sophosympatheia/Wizard-Tulu-Dolphin-70B-v1.0), it inherits the best qualities of each.,
    created=1711065600'''
)

ANUBIS_PRO_105B_V1 = ModelInfo(
    str_identifier="thedrummer/anubis-pro-105b-v1",
    price_in=8e-07,
    price_out=1e-06,
    creator="thedrummer",
    description='''Anubis Pro 105B v1 is an expanded and refined variant of Meta’s Llama 3.3 70B, featuring 50% additional layers and further fine-tuning to leverage its increased capacity. Designed for advanced narrative, roleplay, and instructional tasks, it demonstrates enhanced emotional intelligence, creativity, nuanced character portrayal, and superior prompt adherence compared to smaller models. Its larger parameter count allows for deeper contextual understanding and extended reasoning capabilities, optimized for engaging, intelligent, and coherent interactions.,
    created=1741642290'''
)

ROCINANTE_12B = ModelInfo(
    str_identifier="thedrummer/rocinante-12b",
    price_in=2.5e-07,
    price_out=5e-07,
    creator="thedrummer",
    description='''Rocinante 12B is designed for engaging storytelling and rich prose.

Early testers have reported:
- Expanded vocabulary with unique and expressive word choices
- Enhanced creativity for vivid narratives
- Adventure-filled and captivating stories,
    created=1727654400'''
)

SKYFALL_36B_V2 = ModelInfo(
    str_identifier="thedrummer/skyfall-36b-v2",
    price_in=5e-07,
    price_out=8e-07,
    creator="thedrummer",
    description='''Skyfall 36B v2 is an enhanced iteration of Mistral Small 2501, specifically fine-tuned for improved creativity, nuanced writing, role-playing, and coherent storytelling.,
    created=1741636566'''
)

UNSLOPNEMO_12B = ModelInfo(
    str_identifier="thedrummer/unslopnemo-12b",
    price_in=4.5e-07,
    price_out=4.5e-07,
    creator="thedrummer",
    description='''UnslopNemo v4.1 is the latest addition from the creator of Rocinante, designed for adventure writing and role-play scenarios.,
    created=1731103448'''
)

VALKYRIE_49B_V1 = ModelInfo(
    str_identifier="thedrummer/valkyrie-49b-v1",
    price_in=5e-07,
    price_out=8e-07,
    creator="thedrummer",
    description='''Built on top of NVIDIA's Llama 3.3 Nemotron Super 49B, Valkyrie is TheDrummer's newest model drop for creative writing.,
    created=1748022670'''
)

GLM_4_32B = ModelInfo(
    str_identifier="thudm/glm-4-32b",
    price_in=2.4e-07,
    price_out=2.4e-07,
    creator="thudm",
    description='''GLM-4-32B-0414 is a 32B bilingual (Chinese-English) open-weight language model optimized for code generation, function calling, and agent-style tasks. Pretrained on 15T of high-quality and reasoning-heavy data, it was further refined using human preference alignment, rejection sampling, and reinforcement learning. The model excels in complex reasoning, artifact generation, and structured output tasks, achieving performance comparable to GPT-4o and DeepSeek-V3-0324 across several benchmarks.,
    created=1744920915'''
)

GLM_4_32B_FREE = ModelInfo(
    str_identifier="thudm/glm-4-32b:free",
    price_in=0.0,
    price_out=0.0,
    creator="thudm",
    description='''GLM-4-32B-0414 is a 32B bilingual (Chinese-English) open-weight language model optimized for code generation, function calling, and agent-style tasks. Pretrained on 15T of high-quality and reasoning-heavy data, it was further refined using human preference alignment, rejection sampling, and reinforcement learning. The model excels in complex reasoning, artifact generation, and structured output tasks, achieving performance comparable to GPT-4o and DeepSeek-V3-0324 across several benchmarks.,
    created=1744920915'''
)

GLM_Z1_32B = ModelInfo(
    str_identifier="thudm/glm-z1-32b",
    price_in=2.4e-07,
    price_out=2.4e-07,
    creator="thudm",
    description='''GLM-Z1-32B-0414 is an enhanced reasoning variant of GLM-4-32B, built for deep mathematical, logical, and code-oriented problem solving. It applies extended reinforcement learning—both task-specific and general pairwise preference-based—to improve performance on complex multi-step tasks. Compared to the base GLM-4-32B model, Z1 significantly boosts capabilities in structured reasoning and formal domains.

The model supports enforced “thinking” steps via prompt engineering and offers improved coherence for long-form outputs. It’s optimized for use in agentic workflows, and includes support for long context (via YaRN), JSON tool calling, and fine-grained sampling configuration for stable inference. Ideal for use cases requiring deliberate, multi-step reasoning or formal derivations.,
    created=1744924148'''
)

GLM_Z1_32B_FREE = ModelInfo(
    str_identifier="thudm/glm-z1-32b:free",
    price_in=0.0,
    price_out=0.0,
    creator="thudm",
    description='''GLM-Z1-32B-0414 is an enhanced reasoning variant of GLM-4-32B, built for deep mathematical, logical, and code-oriented problem solving. It applies extended reinforcement learning—both task-specific and general pairwise preference-based—to improve performance on complex multi-step tasks. Compared to the base GLM-4-32B model, Z1 significantly boosts capabilities in structured reasoning and formal domains.

The model supports enforced “thinking” steps via prompt engineering and offers improved coherence for long-form outputs. It’s optimized for use in agentic workflows, and includes support for long context (via YaRN), JSON tool calling, and fine-grained sampling configuration for stable inference. Ideal for use cases requiring deliberate, multi-step reasoning or formal derivations.,
    created=1744924148'''
)

GLM_Z1_RUMINATION_32B = ModelInfo(
    str_identifier="thudm/glm-z1-rumination-32b",
    price_in=2.4e-07,
    price_out=2.4e-07,
    creator="thudm",
    description='''THUDM: GLM Z1 Rumination 32B is a 32B-parameter deep reasoning model from the GLM-4-Z1 series, optimized for complex, open-ended tasks requiring prolonged deliberation. It builds upon glm-4-32b-0414 with additional reinforcement learning phases and multi-stage alignment strategies, introducing “rumination” capabilities designed to emulate extended cognitive processing. This includes iterative reasoning, multi-hop analysis, and tool-augmented workflows such as search, retrieval, and citation-aware synthesis.

The model excels in research-style writing, comparative analysis, and intricate question answering. It supports function calling for search and navigation primitives (`search`, `click`, `open`, `finish`), enabling use in agent-style pipelines. Rumination behavior is governed by multi-turn loops with rule-based reward shaping and delayed decision mechanisms, benchmarked against Deep Research frameworks such as OpenAI’s internal alignment stacks. This variant is suitable for scenarios requiring depth over speed.,
    created=1745601495'''
)

DEEPSEEK_R1T_CHIMERA_FREE = ModelInfo(
    str_identifier="tngtech/deepseek-r1t-chimera:free",
    price_in=0.0,
    price_out=0.0,
    creator="tngtech",
    description='''DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), combining the reasoning capabilities of R1 with the token efficiency improvements of V3. It is based on a DeepSeek-MoE Transformer architecture and is optimized for general text generation tasks.

The model merges pretrained weights from both source models to balance performance across reasoning, efficiency, and instruction-following tasks. It is released under the MIT license and intended for research and commercial use.,
    created=1745760875'''
)

REMM_SLERP_L2_13B = ModelInfo(
    str_identifier="undi95/remm-slerp-l2-13b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="undi95",
    description='''A recreation trial of the original MythoMax-L2-B13 but with updated models. #merge,
    created=1689984000'''
)

TOPPY_M_7B = ModelInfo(
    str_identifier="undi95/toppy-m-7b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="undi95",
    description='''A wild 7B parameter model that merges several models using the new task_arithmetic merge method from mergekit.
List of merged models:
- NousResearch/Nous-Capybara-7B-V1.9
- [HuggingFaceH4/zephyr-7b-beta](/models/huggingfaceh4/zephyr-7b-beta)
- lemonilia/AshhLimaRP-Mistral-7B
- Vulkane/120-Days-of-Sodom-LoRA-Mistral-7b
- Undi95/Mistral-pippa-sharegpt-7b-qlora

#merge #uncensored,
    created=1699574400'''
)

GROK_2_1212 = ModelInfo(
    str_identifier="x-ai/grok-2-1212",
    price_in=2e-06,
    price_out=1e-05,
    creator="x-ai",
    description='''Grok 2 1212 introduces significant enhancements to accuracy, instruction adherence, and multilingual support, making it a powerful and flexible choice for developers seeking a highly steerable, intelligent model.,
    created=1734232814'''
)

GROK_2_VISION_1212 = ModelInfo(
    str_identifier="x-ai/grok-2-vision-1212",
    price_in=2e-06,
    price_out=1e-05,
    creator="x-ai",
    description='''Grok 2 Vision 1212 advances image-based AI with stronger visual comprehension, refined instruction-following, and multilingual support. From object recognition to style analysis, it empowers developers to build more intuitive, visually aware applications. Its enhanced steerability and reasoning establish a robust foundation for next-generation image solutions.

To read more about this model, check out [xAI's announcement](https://x.ai/blog/grok-1212).,
    created=1734237338'''
)

GROK_3_BETA = ModelInfo(
    str_identifier="x-ai/grok-3-beta",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="x-ai",
    description='''Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases like data extraction, coding, and text summarization. Possesses deep domain knowledge in finance, healthcare, law, and science.

Excels in structured tasks and benchmarks like GPQA, LCB, and MMLU-Pro where it outperforms Grok 3 Mini even on high thinking. 

Note: That there are two xAI endpoints for this model. By default when using this model we will always route you to the base endpoint. If you want the fast endpoint you can add `provider: { sort: throughput}`, to sort by throughput instead. 
,
    created=1744240068'''
)

GROK_3_MINI_BETA = ModelInfo(
    str_identifier="x-ai/grok-3-mini-beta",
    price_in=3e-07,
    price_out=5e-07,
    creator="x-ai",
    description='''Grok 3 Mini is a lightweight, smaller thinking model. Unlike traditional models that generate answers immediately, Grok 3 Mini thinks before responding. It’s ideal for reasoning-heavy tasks that don’t demand extensive domain knowledge, and shines in math-specific and quantitative use cases, such as solving challenging puzzles or math problems.

Transparent "thinking" traces accessible. Defaults to low reasoning, can boost with setting `reasoning: { effort: "high" }`

Note: That there are two xAI endpoints for this model. By default when using this model we will always route you to the base endpoint. If you want the fast endpoint you can add `provider: { sort: throughput}`, to sort by throughput instead. 
,
    created=1744240195'''
)

GROK_BETA = ModelInfo(
    str_identifier="x-ai/grok-beta",
    price_in=5e-06,
    price_out=1.5e-05,
    creator="x-ai",
    description='''Grok Beta is xAI's experimental language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases.

It is the successor of [Grok 2](https://x.ai/blog/grok-2) with enhanced context length.,
    created=1729382400'''
)

GROK_VISION_BETA = ModelInfo(
    str_identifier="x-ai/grok-vision-beta",
    price_in=5e-06,
    price_out=1.5e-05,
    creator="x-ai",
    description='''Grok Vision Beta is xAI's experimental language model with vision capability.

,
    created=1731976624'''
)

ALL_MODELS = [
    YI_LARGE,
    MN_STARCANNON_12B,
    DEEPCODER_14B_PREVIEW_FREE,
    JAMBA_1_6_LARGE,
    JAMBA_1_6_MINI,
    AION_1_0,
    AION_1_0_MINI,
    AION_RP_LLAMA_3_1_8B,
    CODELLAMA_7B_INSTRUCT_SOLIDITY,
    OPENHANDS_LM_32B_V0_1,
    GOLIATH_120B,
    MAGNUM_72B,
    NOVA_LITE_V1,
    NOVA_MICRO_V1,
    NOVA_PRO_V1,
    MAGNUM_V2_72B,
    MAGNUM_V4_72B,
    CLAUDE_2,
    CLAUDE_2_0,
    CLAUDE_2_0_BETA,
    CLAUDE_2_1,
    CLAUDE_2_1_BETA,
    CLAUDE_2_BETA,
    CLAUDE_3_HAIKU,
    CLAUDE_3_HAIKU_BETA,
    CLAUDE_3_OPUS,
    CLAUDE_3_OPUS_BETA,
    CLAUDE_3_SONNET,
    CLAUDE_3_SONNET_BETA,
    CLAUDE_3_5_HAIKU,
    CLAUDE_3_5_HAIKU_20241022,
    CLAUDE_3_5_HAIKU_20241022_BETA,
    CLAUDE_3_5_HAIKU_BETA,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_SONNET_20240620,
    CLAUDE_3_5_SONNET_20240620_BETA,
    CLAUDE_3_5_SONNET_BETA,
    CLAUDE_3_7_SONNET,
    CLAUDE_3_7_SONNET_BETA,
    CLAUDE_3_7_SONNET_THINKING,
    CLAUDE_OPUS_4,
    CLAUDE_SONNET_4,
    ARCEE_BLITZ,
    CALLER_LARGE,
    CODER_LARGE,
    MAESTRO_REASONING,
    SPOTLIGHT,
    VIRTUOSO_LARGE,
    VIRTUOSO_MEDIUM_V2,
    QWQ_32B_ARLIAI_RPR_V1_FREE,
    DOLPHIN_MIXTRAL_8X22B,
    DOLPHIN3_0_MISTRAL_24B_FREE,
    DOLPHIN3_0_R1_MISTRAL_24B_FREE,
    COMMAND,
    COMMAND_A,
    COMMAND_R,
    COMMAND_R_03_2024,
    COMMAND_R_08_2024,
    COMMAND_R_PLUS,
    COMMAND_R_PLUS_04_2024,
    COMMAND_R_PLUS_08_2024,
    COMMAND_R7B_12_2024,
    DEEPSEEK_CHAT,
    DEEPSEEK_CHAT_V3_0324,
    DEEPSEEK_CHAT_V3_0324_FREE,
    DEEPSEEK_CHAT_FREE,
    DEEPSEEK_PROVER_V2,
    DEEPSEEK_PROVER_V2_FREE,
    DEEPSEEK_R1,
    DEEPSEEK_R1_0528,
    DEEPSEEK_R1_0528_QWEN3_8B,
    DEEPSEEK_R1_0528_QWEN3_8B_FREE,
    DEEPSEEK_R1_0528_FREE,
    DEEPSEEK_R1_DISTILL_LLAMA_70B,
    DEEPSEEK_R1_DISTILL_LLAMA_70B_FREE,
    DEEPSEEK_R1_DISTILL_LLAMA_8B,
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    DEEPSEEK_R1_DISTILL_QWEN_14B,
    DEEPSEEK_R1_DISTILL_QWEN_14B_FREE,
    DEEPSEEK_R1_DISTILL_QWEN_32B,
    DEEPSEEK_R1_DISTILL_QWEN_32B_FREE,
    DEEPSEEK_R1_DISTILL_QWEN_7B,
    DEEPSEEK_R1_ZERO_FREE,
    DEEPSEEK_R1_FREE,
    DEEPSEEK_V3_BASE_FREE,
    LLEMMA_7B,
    EVA_LLAMA_3_33_70B,
    EVA_QWEN_2_5_32B,
    EVA_QWEN_2_5_72B,
    QWERKY_72B_FREE,
    GEMINI_2_0_FLASH_001,
    GEMINI_2_0_FLASH_EXP_FREE,
    GEMINI_2_0_FLASH_LITE_001,
    GEMINI_2_5_FLASH_PREVIEW,
    GEMINI_2_5_FLASH_PREVIEW_05_20,
    GEMINI_2_5_FLASH_PREVIEW_05_20_THINKING,
    GEMINI_2_5_FLASH_PREVIEW_THINKING,
    GEMINI_2_5_PRO_EXP_03_25,
    GEMINI_2_5_PRO_PREVIEW,
    GEMINI_2_5_PRO_PREVIEW_05_06,
    GEMINI_FLASH_1_5,
    GEMINI_FLASH_1_5_8B,
    GEMINI_PRO_1_5,
    GEMMA_2_27B_IT,
    GEMMA_2_9B_IT,
    GEMMA_2_9B_IT_FREE,
    GEMMA_2B_IT,
    GEMMA_3_12B_IT,
    GEMMA_3_12B_IT_FREE,
    GEMMA_3_1B_IT_FREE,
    GEMMA_3_27B_IT,
    GEMMA_3_27B_IT_FREE,
    GEMMA_3_4B_IT,
    GEMMA_3_4B_IT_FREE,
    GEMMA_3N_E4B_IT_FREE,
    MYTHOMAX_L2_13B,
    MERCURY_CODER_SMALL_BETA,
    MN_INFEROR_12B,
    INFLECTION_3_PI,
    INFLECTION_3_PRODUCTIVITY,
    LFM_3B,
    LFM_40B,
    LFM_7B,
    WEAVER,
    LLAMA_2_70B_CHAT,
    LLAMA_3_70B_INSTRUCT,
    LLAMA_3_8B_INSTRUCT,
    LLAMA_3_1_405B,
    LLAMA_3_1_405B_INSTRUCT,
    LLAMA_3_1_405B_FREE,
    LLAMA_3_1_70B_INSTRUCT,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_1_8B_INSTRUCT_FREE,
    LLAMA_3_2_11B_VISION_INSTRUCT,
    LLAMA_3_2_11B_VISION_INSTRUCT_FREE,
    LLAMA_3_2_1B_INSTRUCT,
    LLAMA_3_2_1B_INSTRUCT_FREE,
    LLAMA_3_2_3B_INSTRUCT,
    LLAMA_3_2_3B_INSTRUCT_FREE,
    LLAMA_3_2_90B_VISION_INSTRUCT,
    LLAMA_3_3_70B_INSTRUCT,
    LLAMA_3_3_70B_INSTRUCT_FREE,
    LLAMA_3_3_8B_INSTRUCT_FREE,
    LLAMA_4_MAVERICK,
    LLAMA_4_MAVERICK_FREE,
    LLAMA_4_SCOUT,
    LLAMA_4_SCOUT_FREE,
    LLAMA_GUARD_2_8B,
    LLAMA_GUARD_3_8B,
    LLAMA_GUARD_4_12B,
    MAI_DS_R1_FREE,
    PHI_3_MEDIUM_128K_INSTRUCT,
    PHI_3_MINI_128K_INSTRUCT,
    PHI_3_5_MINI_128K_INSTRUCT,
    PHI_4,
    PHI_4_MULTIMODAL_INSTRUCT,
    PHI_4_REASONING_PLUS,
    PHI_4_REASONING_PLUS_FREE,
    PHI_4_REASONING_FREE,
    WIZARDLM_2_8X22B,
    MINIMAX_01,
    CODESTRAL_2501,
    DEVSTRAL_SMALL,
    DEVSTRAL_SMALL_FREE,
    MINISTRAL_3B,
    MINISTRAL_8B,
    MISTRAL_7B_INSTRUCT,
    MISTRAL_7B_INSTRUCT_V0_1,
    MISTRAL_7B_INSTRUCT_V0_2,
    MISTRAL_7B_INSTRUCT_V0_3,
    MISTRAL_7B_INSTRUCT_FREE,
    MISTRAL_LARGE,
    MISTRAL_LARGE_2407,
    MISTRAL_LARGE_2411,
    MISTRAL_MEDIUM,
    MISTRAL_MEDIUM_3,
    MISTRAL_NEMO,
    MISTRAL_NEMO_FREE,
    MISTRAL_SABA,
    MISTRAL_SMALL,
    MISTRAL_SMALL_24B_INSTRUCT_2501,
    MISTRAL_SMALL_24B_INSTRUCT_2501_FREE,
    MISTRAL_SMALL_3_1_24B_INSTRUCT,
    MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE,
    MISTRAL_TINY,
    MIXTRAL_8X22B_INSTRUCT,
    MIXTRAL_8X7B_INSTRUCT,
    PIXTRAL_12B,
    PIXTRAL_LARGE_2411,
    KIMI_VL_A3B_THINKING_FREE,
    MOONLIGHT_16B_A3B_INSTRUCT_FREE,
    LLAMA_3_LUMIMAID_70B,
    LLAMA_3_LUMIMAID_8B,
    LLAMA_3_1_LUMIMAID_70B,
    LLAMA_3_1_LUMIMAID_8B,
    NOROMAID_20B,
    MN_CELESTE_12B,
    DEEPHERMES_3_LLAMA_3_8B_PREVIEW_FREE,
    DEEPHERMES_3_MISTRAL_24B_PREVIEW_FREE,
    HERMES_2_PRO_LLAMA_3_8B,
    HERMES_3_LLAMA_3_1_405B,
    HERMES_3_LLAMA_3_1_70B,
    NOUS_HERMES_2_MIXTRAL_8X7B_DPO,
    LLAMA_3_1_NEMOTRON_70B_INSTRUCT,
    LLAMA_3_1_NEMOTRON_ULTRA_253B_V1,
    LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE,
    LLAMA_3_3_NEMOTRON_SUPER_49B_V1,
    LLAMA_3_3_NEMOTRON_SUPER_49B_V1_FREE,
    OLYMPICCODER_32B_FREE,
    CHATGPT_4O_LATEST,
    CODEX_MINI,
    GPT_3_5_TURBO,
    GPT_3_5_TURBO_0125,
    GPT_3_5_TURBO_0613,
    GPT_3_5_TURBO_1106,
    GPT_3_5_TURBO_16K,
    GPT_3_5_TURBO_INSTRUCT,
    GPT_4,
    GPT_4_0314,
    GPT_4_1106_PREVIEW,
    GPT_4_TURBO,
    GPT_4_TURBO_PREVIEW,
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    GPT_4_5_PREVIEW,
    GPT_4O,
    GPT_4O_2024_05_13,
    GPT_4O_2024_08_06,
    GPT_4O_2024_11_20,
    GPT_4O_MINI,
    GPT_4O_MINI_2024_07_18,
    GPT_4O_MINI_SEARCH_PREVIEW,
    GPT_4O_SEARCH_PREVIEW,
    GPT_4O_EXTENDED,
    O1,
    O1_MINI,
    O1_MINI_2024_09_12,
    O1_PREVIEW,
    O1_PREVIEW_2024_09_12,
    O1_PRO,
    O3,
    O3_MINI,
    O3_MINI_HIGH,
    O4_MINI,
    O4_MINI_HIGH,
    INTERNVL3_14B_FREE,
    INTERNVL3_2B_FREE,
    AUTO,
    LLAMA_3_1_SONAR_LARGE_128K_ONLINE,
    LLAMA_3_1_SONAR_SMALL_128K_ONLINE,
    R1_1776,
    SONAR,
    SONAR_DEEP_RESEARCH,
    SONAR_PRO,
    SONAR_REASONING,
    SONAR_REASONING_PRO,
    MYTHALION_13B,
    QWEN_2_72B_INSTRUCT,
    QWEN_2_5_72B_INSTRUCT,
    QWEN_2_5_72B_INSTRUCT_FREE,
    QWEN_2_5_7B_INSTRUCT,
    QWEN_2_5_7B_INSTRUCT_FREE,
    QWEN_2_5_CODER_32B_INSTRUCT,
    QWEN_2_5_CODER_32B_INSTRUCT_FREE,
    QWEN_2_5_VL_7B_INSTRUCT,
    QWEN_2_5_VL_7B_INSTRUCT_FREE,
    QWEN_MAX,
    QWEN_PLUS,
    QWEN_TURBO,
    QWEN_VL_MAX,
    QWEN_VL_PLUS,
    QWEN2_5_VL_32B_INSTRUCT,
    QWEN2_5_VL_32B_INSTRUCT_FREE,
    QWEN2_5_VL_3B_INSTRUCT_FREE,
    QWEN2_5_VL_72B_INSTRUCT,
    QWEN2_5_VL_72B_INSTRUCT_FREE,
    QWEN3_14B,
    QWEN3_14B_FREE,
    QWEN3_235B_A22B,
    QWEN3_235B_A22B_FREE,
    QWEN3_30B_A3B,
    QWEN3_30B_A3B_FREE,
    QWEN3_32B,
    QWEN3_32B_FREE,
    QWEN3_8B,
    QWEN3_8B_FREE,
    QWQ_32B,
    QWQ_32B_PREVIEW,
    QWQ_32B_FREE,
    SORCERERLM_8X22B,
    REKA_FLASH_3_FREE,
    FIMBULVETR_11B_V2,
    L3_EURYALE_70B,
    L3_LUNARIS_8B,
    L3_1_EURYALE_70B,
    L3_3_EURYALE_70B,
    SARVAM_M,
    SARVAM_M_FREE,
    LLAMA3_1_TYPHOON2_70B_INSTRUCT,
    DOBBY_MINI_UNHINGED_PLUS_LLAMA_3_1_8B,
    SHISA_V2_LLAMA3_3_70B_FREE,
    MIDNIGHT_ROSE_70B,
    ANUBIS_PRO_105B_V1,
    ROCINANTE_12B,
    SKYFALL_36B_V2,
    UNSLOPNEMO_12B,
    VALKYRIE_49B_V1,
    GLM_4_32B,
    GLM_4_32B_FREE,
    GLM_Z1_32B,
    GLM_Z1_32B_FREE,
    GLM_Z1_RUMINATION_32B,
    DEEPSEEK_R1T_CHIMERA_FREE,
    REMM_SLERP_L2_13B,
    TOPPY_M_7B,
    GROK_2_1212,
    GROK_2_VISION_1212,
    GROK_3_BETA,
    GROK_3_MINI_BETA,
    GROK_BETA,
    GROK_VISION_BETA,
]
FREE_MODELS = [
    DEEPCODER_14B_PREVIEW_FREE,
    QWQ_32B_ARLIAI_RPR_V1_FREE,
    DOLPHIN3_0_MISTRAL_24B_FREE,
    DOLPHIN3_0_R1_MISTRAL_24B_FREE,
    DEEPSEEK_CHAT_V3_0324_FREE,
    DEEPSEEK_CHAT_FREE,
    DEEPSEEK_PROVER_V2_FREE,
    DEEPSEEK_R1_0528_QWEN3_8B_FREE,
    DEEPSEEK_R1_0528_FREE,
    DEEPSEEK_R1_DISTILL_LLAMA_70B_FREE,
    DEEPSEEK_R1_DISTILL_QWEN_14B_FREE,
    DEEPSEEK_R1_DISTILL_QWEN_32B_FREE,
    DEEPSEEK_R1_ZERO_FREE,
    DEEPSEEK_R1_FREE,
    DEEPSEEK_V3_BASE_FREE,
    QWERKY_72B_FREE,
    GEMINI_2_0_FLASH_EXP_FREE,
    GEMINI_2_5_PRO_EXP_03_25,
    GEMMA_2_9B_IT_FREE,
    GEMMA_3_12B_IT_FREE,
    GEMMA_3_1B_IT_FREE,
    GEMMA_3_27B_IT_FREE,
    GEMMA_3_4B_IT_FREE,
    GEMMA_3N_E4B_IT_FREE,
    LLAMA_3_1_405B_FREE,
    LLAMA_3_1_8B_INSTRUCT_FREE,
    LLAMA_3_2_11B_VISION_INSTRUCT_FREE,
    LLAMA_3_2_1B_INSTRUCT_FREE,
    LLAMA_3_2_3B_INSTRUCT_FREE,
    LLAMA_3_3_70B_INSTRUCT_FREE,
    LLAMA_3_3_8B_INSTRUCT_FREE,
    LLAMA_4_MAVERICK_FREE,
    LLAMA_4_SCOUT_FREE,
    MAI_DS_R1_FREE,
    PHI_4_REASONING_PLUS_FREE,
    PHI_4_REASONING_FREE,
    DEVSTRAL_SMALL_FREE,
    MISTRAL_7B_INSTRUCT_FREE,
    MISTRAL_NEMO_FREE,
    MISTRAL_SMALL_24B_INSTRUCT_2501_FREE,
    MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE,
    KIMI_VL_A3B_THINKING_FREE,
    MOONLIGHT_16B_A3B_INSTRUCT_FREE,
    DEEPHERMES_3_LLAMA_3_8B_PREVIEW_FREE,
    DEEPHERMES_3_MISTRAL_24B_PREVIEW_FREE,
    LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE,
    LLAMA_3_3_NEMOTRON_SUPER_49B_V1_FREE,
    OLYMPICCODER_32B_FREE,
    INTERNVL3_14B_FREE,
    INTERNVL3_2B_FREE,
    QWEN_2_5_72B_INSTRUCT_FREE,
    QWEN_2_5_7B_INSTRUCT_FREE,
    QWEN_2_5_CODER_32B_INSTRUCT_FREE,
    QWEN_2_5_VL_7B_INSTRUCT_FREE,
    QWEN2_5_VL_32B_INSTRUCT_FREE,
    QWEN2_5_VL_3B_INSTRUCT_FREE,
    QWEN2_5_VL_72B_INSTRUCT_FREE,
    QWEN3_14B_FREE,
    QWEN3_235B_A22B_FREE,
    QWEN3_30B_A3B_FREE,
    QWEN3_32B_FREE,
    QWEN3_8B_FREE,
    QWQ_32B_FREE,
    REKA_FLASH_3_FREE,
    SARVAM_M_FREE,
    SHISA_V2_LLAMA3_3_70B_FREE,
    GLM_4_32B_FREE,
    GLM_Z1_32B_FREE,
    DEEPSEEK_R1T_CHIMERA_FREE,
]
ALL_MODELS_DICT = {
    "YI_LARGE": YI_LARGE,
    "MN_STARCANNON_12B": MN_STARCANNON_12B,
    "DEEPCODER_14B_PREVIEW_FREE": DEEPCODER_14B_PREVIEW_FREE,
    "JAMBA_1_6_LARGE": JAMBA_1_6_LARGE,
    "JAMBA_1_6_MINI": JAMBA_1_6_MINI,
    "AION_1_0": AION_1_0,
    "AION_1_0_MINI": AION_1_0_MINI,
    "AION_RP_LLAMA_3_1_8B": AION_RP_LLAMA_3_1_8B,
    "CODELLAMA_7B_INSTRUCT_SOLIDITY": CODELLAMA_7B_INSTRUCT_SOLIDITY,
    "OPENHANDS_LM_32B_V0_1": OPENHANDS_LM_32B_V0_1,
    "GOLIATH_120B": GOLIATH_120B,
    "MAGNUM_72B": MAGNUM_72B,
    "NOVA_LITE_V1": NOVA_LITE_V1,
    "NOVA_MICRO_V1": NOVA_MICRO_V1,
    "NOVA_PRO_V1": NOVA_PRO_V1,
    "MAGNUM_V2_72B": MAGNUM_V2_72B,
    "MAGNUM_V4_72B": MAGNUM_V4_72B,
    "CLAUDE_2": CLAUDE_2,
    "CLAUDE_2_0": CLAUDE_2_0,
    "CLAUDE_2_0_BETA": CLAUDE_2_0_BETA,
    "CLAUDE_2_1": CLAUDE_2_1,
    "CLAUDE_2_1_BETA": CLAUDE_2_1_BETA,
    "CLAUDE_2_BETA": CLAUDE_2_BETA,
    "CLAUDE_3_HAIKU": CLAUDE_3_HAIKU,
    "CLAUDE_3_HAIKU_BETA": CLAUDE_3_HAIKU_BETA,
    "CLAUDE_3_OPUS": CLAUDE_3_OPUS,
    "CLAUDE_3_OPUS_BETA": CLAUDE_3_OPUS_BETA,
    "CLAUDE_3_SONNET": CLAUDE_3_SONNET,
    "CLAUDE_3_SONNET_BETA": CLAUDE_3_SONNET_BETA,
    "CLAUDE_3_5_HAIKU": CLAUDE_3_5_HAIKU,
    "CLAUDE_3_5_HAIKU_20241022": CLAUDE_3_5_HAIKU_20241022,
    "CLAUDE_3_5_HAIKU_20241022_BETA": CLAUDE_3_5_HAIKU_20241022_BETA,
    "CLAUDE_3_5_HAIKU_BETA": CLAUDE_3_5_HAIKU_BETA,
    "CLAUDE_3_5_SONNET": CLAUDE_3_5_SONNET,
    "CLAUDE_3_5_SONNET_20240620": CLAUDE_3_5_SONNET_20240620,
    "CLAUDE_3_5_SONNET_20240620_BETA": CLAUDE_3_5_SONNET_20240620_BETA,
    "CLAUDE_3_5_SONNET_BETA": CLAUDE_3_5_SONNET_BETA,
    "CLAUDE_3_7_SONNET": CLAUDE_3_7_SONNET,
    "CLAUDE_3_7_SONNET_BETA": CLAUDE_3_7_SONNET_BETA,
    "CLAUDE_3_7_SONNET_THINKING": CLAUDE_3_7_SONNET_THINKING,
    "CLAUDE_OPUS_4": CLAUDE_OPUS_4,
    "CLAUDE_SONNET_4": CLAUDE_SONNET_4,
    "ARCEE_BLITZ": ARCEE_BLITZ,
    "CALLER_LARGE": CALLER_LARGE,
    "CODER_LARGE": CODER_LARGE,
    "MAESTRO_REASONING": MAESTRO_REASONING,
    "SPOTLIGHT": SPOTLIGHT,
    "VIRTUOSO_LARGE": VIRTUOSO_LARGE,
    "VIRTUOSO_MEDIUM_V2": VIRTUOSO_MEDIUM_V2,
    "QWQ_32B_ARLIAI_RPR_V1_FREE": QWQ_32B_ARLIAI_RPR_V1_FREE,
    "DOLPHIN_MIXTRAL_8X22B": DOLPHIN_MIXTRAL_8X22B,
    "DOLPHIN3_0_MISTRAL_24B_FREE": DOLPHIN3_0_MISTRAL_24B_FREE,
    "DOLPHIN3_0_R1_MISTRAL_24B_FREE": DOLPHIN3_0_R1_MISTRAL_24B_FREE,
    "COMMAND": COMMAND,
    "COMMAND_A": COMMAND_A,
    "COMMAND_R": COMMAND_R,
    "COMMAND_R_03_2024": COMMAND_R_03_2024,
    "COMMAND_R_08_2024": COMMAND_R_08_2024,
    "COMMAND_R_PLUS": COMMAND_R_PLUS,
    "COMMAND_R_PLUS_04_2024": COMMAND_R_PLUS_04_2024,
    "COMMAND_R_PLUS_08_2024": COMMAND_R_PLUS_08_2024,
    "COMMAND_R7B_12_2024": COMMAND_R7B_12_2024,
    "DEEPSEEK_CHAT": DEEPSEEK_CHAT,
    "DEEPSEEK_CHAT_V3_0324": DEEPSEEK_CHAT_V3_0324,
    "DEEPSEEK_CHAT_V3_0324_FREE": DEEPSEEK_CHAT_V3_0324_FREE,
    "DEEPSEEK_CHAT_FREE": DEEPSEEK_CHAT_FREE,
    "DEEPSEEK_PROVER_V2": DEEPSEEK_PROVER_V2,
    "DEEPSEEK_PROVER_V2_FREE": DEEPSEEK_PROVER_V2_FREE,
    "DEEPSEEK_R1": DEEPSEEK_R1,
    "DEEPSEEK_R1_0528": DEEPSEEK_R1_0528,
    "DEEPSEEK_R1_0528_QWEN3_8B": DEEPSEEK_R1_0528_QWEN3_8B,
    "DEEPSEEK_R1_0528_QWEN3_8B_FREE": DEEPSEEK_R1_0528_QWEN3_8B_FREE,
    "DEEPSEEK_R1_0528_FREE": DEEPSEEK_R1_0528_FREE,
    "DEEPSEEK_R1_DISTILL_LLAMA_70B": DEEPSEEK_R1_DISTILL_LLAMA_70B,
    "DEEPSEEK_R1_DISTILL_LLAMA_70B_FREE": DEEPSEEK_R1_DISTILL_LLAMA_70B_FREE,
    "DEEPSEEK_R1_DISTILL_LLAMA_8B": DEEPSEEK_R1_DISTILL_LLAMA_8B,
    "DEEPSEEK_R1_DISTILL_QWEN_1_5B": DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    "DEEPSEEK_R1_DISTILL_QWEN_14B": DEEPSEEK_R1_DISTILL_QWEN_14B,
    "DEEPSEEK_R1_DISTILL_QWEN_14B_FREE": DEEPSEEK_R1_DISTILL_QWEN_14B_FREE,
    "DEEPSEEK_R1_DISTILL_QWEN_32B": DEEPSEEK_R1_DISTILL_QWEN_32B,
    "DEEPSEEK_R1_DISTILL_QWEN_32B_FREE": DEEPSEEK_R1_DISTILL_QWEN_32B_FREE,
    "DEEPSEEK_R1_DISTILL_QWEN_7B": DEEPSEEK_R1_DISTILL_QWEN_7B,
    "DEEPSEEK_R1_ZERO_FREE": DEEPSEEK_R1_ZERO_FREE,
    "DEEPSEEK_R1_FREE": DEEPSEEK_R1_FREE,
    "DEEPSEEK_V3_BASE_FREE": DEEPSEEK_V3_BASE_FREE,
    "LLEMMA_7B": LLEMMA_7B,
    "EVA_LLAMA_3_33_70B": EVA_LLAMA_3_33_70B,
    "EVA_QWEN_2_5_32B": EVA_QWEN_2_5_32B,
    "EVA_QWEN_2_5_72B": EVA_QWEN_2_5_72B,
    "QWERKY_72B_FREE": QWERKY_72B_FREE,
    "GEMINI_2_0_FLASH_001": GEMINI_2_0_FLASH_001,
    "GEMINI_2_0_FLASH_EXP_FREE": GEMINI_2_0_FLASH_EXP_FREE,
    "GEMINI_2_0_FLASH_LITE_001": GEMINI_2_0_FLASH_LITE_001,
    "GEMINI_2_5_FLASH_PREVIEW": GEMINI_2_5_FLASH_PREVIEW,
    "GEMINI_2_5_FLASH_PREVIEW_05_20": GEMINI_2_5_FLASH_PREVIEW_05_20,
    "GEMINI_2_5_FLASH_PREVIEW_05_20_THINKING": GEMINI_2_5_FLASH_PREVIEW_05_20_THINKING,
    "GEMINI_2_5_FLASH_PREVIEW_THINKING": GEMINI_2_5_FLASH_PREVIEW_THINKING,
    "GEMINI_2_5_PRO_EXP_03_25": GEMINI_2_5_PRO_EXP_03_25,
    "GEMINI_2_5_PRO_PREVIEW": GEMINI_2_5_PRO_PREVIEW,
    "GEMINI_2_5_PRO_PREVIEW_05_06": GEMINI_2_5_PRO_PREVIEW_05_06,
    "GEMINI_FLASH_1_5": GEMINI_FLASH_1_5,
    "GEMINI_FLASH_1_5_8B": GEMINI_FLASH_1_5_8B,
    "GEMINI_PRO_1_5": GEMINI_PRO_1_5,
    "GEMMA_2_27B_IT": GEMMA_2_27B_IT,
    "GEMMA_2_9B_IT": GEMMA_2_9B_IT,
    "GEMMA_2_9B_IT_FREE": GEMMA_2_9B_IT_FREE,
    "GEMMA_2B_IT": GEMMA_2B_IT,
    "GEMMA_3_12B_IT": GEMMA_3_12B_IT,
    "GEMMA_3_12B_IT_FREE": GEMMA_3_12B_IT_FREE,
    "GEMMA_3_1B_IT_FREE": GEMMA_3_1B_IT_FREE,
    "GEMMA_3_27B_IT": GEMMA_3_27B_IT,
    "GEMMA_3_27B_IT_FREE": GEMMA_3_27B_IT_FREE,
    "GEMMA_3_4B_IT": GEMMA_3_4B_IT,
    "GEMMA_3_4B_IT_FREE": GEMMA_3_4B_IT_FREE,
    "GEMMA_3N_E4B_IT_FREE": GEMMA_3N_E4B_IT_FREE,
    "MYTHOMAX_L2_13B": MYTHOMAX_L2_13B,
    "MERCURY_CODER_SMALL_BETA": MERCURY_CODER_SMALL_BETA,
    "MN_INFEROR_12B": MN_INFEROR_12B,
    "INFLECTION_3_PI": INFLECTION_3_PI,
    "INFLECTION_3_PRODUCTIVITY": INFLECTION_3_PRODUCTIVITY,
    "LFM_3B": LFM_3B,
    "LFM_40B": LFM_40B,
    "LFM_7B": LFM_7B,
    "WEAVER": WEAVER,
    "LLAMA_2_70B_CHAT": LLAMA_2_70B_CHAT,
    "LLAMA_3_70B_INSTRUCT": LLAMA_3_70B_INSTRUCT,
    "LLAMA_3_8B_INSTRUCT": LLAMA_3_8B_INSTRUCT,
    "LLAMA_3_1_405B": LLAMA_3_1_405B,
    "LLAMA_3_1_405B_INSTRUCT": LLAMA_3_1_405B_INSTRUCT,
    "LLAMA_3_1_405B_FREE": LLAMA_3_1_405B_FREE,
    "LLAMA_3_1_70B_INSTRUCT": LLAMA_3_1_70B_INSTRUCT,
    "LLAMA_3_1_8B_INSTRUCT": LLAMA_3_1_8B_INSTRUCT,
    "LLAMA_3_1_8B_INSTRUCT_FREE": LLAMA_3_1_8B_INSTRUCT_FREE,
    "LLAMA_3_2_11B_VISION_INSTRUCT": LLAMA_3_2_11B_VISION_INSTRUCT,
    "LLAMA_3_2_11B_VISION_INSTRUCT_FREE": LLAMA_3_2_11B_VISION_INSTRUCT_FREE,
    "LLAMA_3_2_1B_INSTRUCT": LLAMA_3_2_1B_INSTRUCT,
    "LLAMA_3_2_1B_INSTRUCT_FREE": LLAMA_3_2_1B_INSTRUCT_FREE,
    "LLAMA_3_2_3B_INSTRUCT": LLAMA_3_2_3B_INSTRUCT,
    "LLAMA_3_2_3B_INSTRUCT_FREE": LLAMA_3_2_3B_INSTRUCT_FREE,
    "LLAMA_3_2_90B_VISION_INSTRUCT": LLAMA_3_2_90B_VISION_INSTRUCT,
    "LLAMA_3_3_70B_INSTRUCT": LLAMA_3_3_70B_INSTRUCT,
    "LLAMA_3_3_70B_INSTRUCT_FREE": LLAMA_3_3_70B_INSTRUCT_FREE,
    "LLAMA_3_3_8B_INSTRUCT_FREE": LLAMA_3_3_8B_INSTRUCT_FREE,
    "LLAMA_4_MAVERICK": LLAMA_4_MAVERICK,
    "LLAMA_4_MAVERICK_FREE": LLAMA_4_MAVERICK_FREE,
    "LLAMA_4_SCOUT": LLAMA_4_SCOUT,
    "LLAMA_4_SCOUT_FREE": LLAMA_4_SCOUT_FREE,
    "LLAMA_GUARD_2_8B": LLAMA_GUARD_2_8B,
    "LLAMA_GUARD_3_8B": LLAMA_GUARD_3_8B,
    "LLAMA_GUARD_4_12B": LLAMA_GUARD_4_12B,
    "MAI_DS_R1_FREE": MAI_DS_R1_FREE,
    "PHI_3_MEDIUM_128K_INSTRUCT": PHI_3_MEDIUM_128K_INSTRUCT,
    "PHI_3_MINI_128K_INSTRUCT": PHI_3_MINI_128K_INSTRUCT,
    "PHI_3_5_MINI_128K_INSTRUCT": PHI_3_5_MINI_128K_INSTRUCT,
    "PHI_4": PHI_4,
    "PHI_4_MULTIMODAL_INSTRUCT": PHI_4_MULTIMODAL_INSTRUCT,
    "PHI_4_REASONING_PLUS": PHI_4_REASONING_PLUS,
    "PHI_4_REASONING_PLUS_FREE": PHI_4_REASONING_PLUS_FREE,
    "PHI_4_REASONING_FREE": PHI_4_REASONING_FREE,
    "WIZARDLM_2_8X22B": WIZARDLM_2_8X22B,
    "MINIMAX_01": MINIMAX_01,
    "CODESTRAL_2501": CODESTRAL_2501,
    "DEVSTRAL_SMALL": DEVSTRAL_SMALL,
    "DEVSTRAL_SMALL_FREE": DEVSTRAL_SMALL_FREE,
    "MINISTRAL_3B": MINISTRAL_3B,
    "MINISTRAL_8B": MINISTRAL_8B,
    "MISTRAL_7B_INSTRUCT": MISTRAL_7B_INSTRUCT,
    "MISTRAL_7B_INSTRUCT_V0_1": MISTRAL_7B_INSTRUCT_V0_1,
    "MISTRAL_7B_INSTRUCT_V0_2": MISTRAL_7B_INSTRUCT_V0_2,
    "MISTRAL_7B_INSTRUCT_V0_3": MISTRAL_7B_INSTRUCT_V0_3,
    "MISTRAL_7B_INSTRUCT_FREE": MISTRAL_7B_INSTRUCT_FREE,
    "MISTRAL_LARGE": MISTRAL_LARGE,
    "MISTRAL_LARGE_2407": MISTRAL_LARGE_2407,
    "MISTRAL_LARGE_2411": MISTRAL_LARGE_2411,
    "MISTRAL_MEDIUM": MISTRAL_MEDIUM,
    "MISTRAL_MEDIUM_3": MISTRAL_MEDIUM_3,
    "MISTRAL_NEMO": MISTRAL_NEMO,
    "MISTRAL_NEMO_FREE": MISTRAL_NEMO_FREE,
    "MISTRAL_SABA": MISTRAL_SABA,
    "MISTRAL_SMALL": MISTRAL_SMALL,
    "MISTRAL_SMALL_24B_INSTRUCT_2501": MISTRAL_SMALL_24B_INSTRUCT_2501,
    "MISTRAL_SMALL_24B_INSTRUCT_2501_FREE": MISTRAL_SMALL_24B_INSTRUCT_2501_FREE,
    "MISTRAL_SMALL_3_1_24B_INSTRUCT": MISTRAL_SMALL_3_1_24B_INSTRUCT,
    "MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE": MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE,
    "MISTRAL_TINY": MISTRAL_TINY,
    "MIXTRAL_8X22B_INSTRUCT": MIXTRAL_8X22B_INSTRUCT,
    "MIXTRAL_8X7B_INSTRUCT": MIXTRAL_8X7B_INSTRUCT,
    "PIXTRAL_12B": PIXTRAL_12B,
    "PIXTRAL_LARGE_2411": PIXTRAL_LARGE_2411,
    "KIMI_VL_A3B_THINKING_FREE": KIMI_VL_A3B_THINKING_FREE,
    "MOONLIGHT_16B_A3B_INSTRUCT_FREE": MOONLIGHT_16B_A3B_INSTRUCT_FREE,
    "LLAMA_3_LUMIMAID_70B": LLAMA_3_LUMIMAID_70B,
    "LLAMA_3_LUMIMAID_8B": LLAMA_3_LUMIMAID_8B,
    "LLAMA_3_1_LUMIMAID_70B": LLAMA_3_1_LUMIMAID_70B,
    "LLAMA_3_1_LUMIMAID_8B": LLAMA_3_1_LUMIMAID_8B,
    "NOROMAID_20B": NOROMAID_20B,
    "MN_CELESTE_12B": MN_CELESTE_12B,
    "DEEPHERMES_3_LLAMA_3_8B_PREVIEW_FREE": DEEPHERMES_3_LLAMA_3_8B_PREVIEW_FREE,
    "DEEPHERMES_3_MISTRAL_24B_PREVIEW_FREE": DEEPHERMES_3_MISTRAL_24B_PREVIEW_FREE,
    "HERMES_2_PRO_LLAMA_3_8B": HERMES_2_PRO_LLAMA_3_8B,
    "HERMES_3_LLAMA_3_1_405B": HERMES_3_LLAMA_3_1_405B,
    "HERMES_3_LLAMA_3_1_70B": HERMES_3_LLAMA_3_1_70B,
    "NOUS_HERMES_2_MIXTRAL_8X7B_DPO": NOUS_HERMES_2_MIXTRAL_8X7B_DPO,
    "LLAMA_3_1_NEMOTRON_70B_INSTRUCT": LLAMA_3_1_NEMOTRON_70B_INSTRUCT,
    "LLAMA_3_1_NEMOTRON_ULTRA_253B_V1": LLAMA_3_1_NEMOTRON_ULTRA_253B_V1,
    "LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE": LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE,
    "LLAMA_3_3_NEMOTRON_SUPER_49B_V1": LLAMA_3_3_NEMOTRON_SUPER_49B_V1,
    "LLAMA_3_3_NEMOTRON_SUPER_49B_V1_FREE": LLAMA_3_3_NEMOTRON_SUPER_49B_V1_FREE,
    "OLYMPICCODER_32B_FREE": OLYMPICCODER_32B_FREE,
    "CHATGPT_4O_LATEST": CHATGPT_4O_LATEST,
    "CODEX_MINI": CODEX_MINI,
    "GPT_3_5_TURBO": GPT_3_5_TURBO,
    "GPT_3_5_TURBO_0125": GPT_3_5_TURBO_0125,
    "GPT_3_5_TURBO_0613": GPT_3_5_TURBO_0613,
    "GPT_3_5_TURBO_1106": GPT_3_5_TURBO_1106,
    "GPT_3_5_TURBO_16K": GPT_3_5_TURBO_16K,
    "GPT_3_5_TURBO_INSTRUCT": GPT_3_5_TURBO_INSTRUCT,
    "GPT_4": GPT_4,
    "GPT_4_0314": GPT_4_0314,
    "GPT_4_1106_PREVIEW": GPT_4_1106_PREVIEW,
    "GPT_4_TURBO": GPT_4_TURBO,
    "GPT_4_TURBO_PREVIEW": GPT_4_TURBO_PREVIEW,
    "GPT_4_1": GPT_4_1,
    "GPT_4_1_MINI": GPT_4_1_MINI,
    "GPT_4_1_NANO": GPT_4_1_NANO,
    "GPT_4_5_PREVIEW": GPT_4_5_PREVIEW,
    "GPT_4O": GPT_4O,
    "GPT_4O_2024_05_13": GPT_4O_2024_05_13,
    "GPT_4O_2024_08_06": GPT_4O_2024_08_06,
    "GPT_4O_2024_11_20": GPT_4O_2024_11_20,
    "GPT_4O_MINI": GPT_4O_MINI,
    "GPT_4O_MINI_2024_07_18": GPT_4O_MINI_2024_07_18,
    "GPT_4O_MINI_SEARCH_PREVIEW": GPT_4O_MINI_SEARCH_PREVIEW,
    "GPT_4O_SEARCH_PREVIEW": GPT_4O_SEARCH_PREVIEW,
    "GPT_4O_EXTENDED": GPT_4O_EXTENDED,
    "O1": O1,
    "O1_MINI": O1_MINI,
    "O1_MINI_2024_09_12": O1_MINI_2024_09_12,
    "O1_PREVIEW": O1_PREVIEW,
    "O1_PREVIEW_2024_09_12": O1_PREVIEW_2024_09_12,
    "O1_PRO": O1_PRO,
    "O3": O3,
    "O3_MINI": O3_MINI,
    "O3_MINI_HIGH": O3_MINI_HIGH,
    "O4_MINI": O4_MINI,
    "O4_MINI_HIGH": O4_MINI_HIGH,
    "INTERNVL3_14B_FREE": INTERNVL3_14B_FREE,
    "INTERNVL3_2B_FREE": INTERNVL3_2B_FREE,
    "AUTO": AUTO,
    "LLAMA_3_1_SONAR_LARGE_128K_ONLINE": LLAMA_3_1_SONAR_LARGE_128K_ONLINE,
    "LLAMA_3_1_SONAR_SMALL_128K_ONLINE": LLAMA_3_1_SONAR_SMALL_128K_ONLINE,
    "R1_1776": R1_1776,
    "SONAR": SONAR,
    "SONAR_DEEP_RESEARCH": SONAR_DEEP_RESEARCH,
    "SONAR_PRO": SONAR_PRO,
    "SONAR_REASONING": SONAR_REASONING,
    "SONAR_REASONING_PRO": SONAR_REASONING_PRO,
    "MYTHALION_13B": MYTHALION_13B,
    "QWEN_2_72B_INSTRUCT": QWEN_2_72B_INSTRUCT,
    "QWEN_2_5_72B_INSTRUCT": QWEN_2_5_72B_INSTRUCT,
    "QWEN_2_5_72B_INSTRUCT_FREE": QWEN_2_5_72B_INSTRUCT_FREE,
    "QWEN_2_5_7B_INSTRUCT": QWEN_2_5_7B_INSTRUCT,
    "QWEN_2_5_7B_INSTRUCT_FREE": QWEN_2_5_7B_INSTRUCT_FREE,
    "QWEN_2_5_CODER_32B_INSTRUCT": QWEN_2_5_CODER_32B_INSTRUCT,
    "QWEN_2_5_CODER_32B_INSTRUCT_FREE": QWEN_2_5_CODER_32B_INSTRUCT_FREE,
    "QWEN_2_5_VL_7B_INSTRUCT": QWEN_2_5_VL_7B_INSTRUCT,
    "QWEN_2_5_VL_7B_INSTRUCT_FREE": QWEN_2_5_VL_7B_INSTRUCT_FREE,
    "QWEN_MAX": QWEN_MAX,
    "QWEN_PLUS": QWEN_PLUS,
    "QWEN_TURBO": QWEN_TURBO,
    "QWEN_VL_MAX": QWEN_VL_MAX,
    "QWEN_VL_PLUS": QWEN_VL_PLUS,
    "QWEN2_5_VL_32B_INSTRUCT": QWEN2_5_VL_32B_INSTRUCT,
    "QWEN2_5_VL_32B_INSTRUCT_FREE": QWEN2_5_VL_32B_INSTRUCT_FREE,
    "QWEN2_5_VL_3B_INSTRUCT_FREE": QWEN2_5_VL_3B_INSTRUCT_FREE,
    "QWEN2_5_VL_72B_INSTRUCT": QWEN2_5_VL_72B_INSTRUCT,
    "QWEN2_5_VL_72B_INSTRUCT_FREE": QWEN2_5_VL_72B_INSTRUCT_FREE,
    "QWEN3_14B": QWEN3_14B,
    "QWEN3_14B_FREE": QWEN3_14B_FREE,
    "QWEN3_235B_A22B": QWEN3_235B_A22B,
    "QWEN3_235B_A22B_FREE": QWEN3_235B_A22B_FREE,
    "QWEN3_30B_A3B": QWEN3_30B_A3B,
    "QWEN3_30B_A3B_FREE": QWEN3_30B_A3B_FREE,
    "QWEN3_32B": QWEN3_32B,
    "QWEN3_32B_FREE": QWEN3_32B_FREE,
    "QWEN3_8B": QWEN3_8B,
    "QWEN3_8B_FREE": QWEN3_8B_FREE,
    "QWQ_32B": QWQ_32B,
    "QWQ_32B_PREVIEW": QWQ_32B_PREVIEW,
    "QWQ_32B_FREE": QWQ_32B_FREE,
    "SORCERERLM_8X22B": SORCERERLM_8X22B,
    "REKA_FLASH_3_FREE": REKA_FLASH_3_FREE,
    "FIMBULVETR_11B_V2": FIMBULVETR_11B_V2,
    "L3_EURYALE_70B": L3_EURYALE_70B,
    "L3_LUNARIS_8B": L3_LUNARIS_8B,
    "L3_1_EURYALE_70B": L3_1_EURYALE_70B,
    "L3_3_EURYALE_70B": L3_3_EURYALE_70B,
    "SARVAM_M": SARVAM_M,
    "SARVAM_M_FREE": SARVAM_M_FREE,
    "LLAMA3_1_TYPHOON2_70B_INSTRUCT": LLAMA3_1_TYPHOON2_70B_INSTRUCT,
    "DOBBY_MINI_UNHINGED_PLUS_LLAMA_3_1_8B": DOBBY_MINI_UNHINGED_PLUS_LLAMA_3_1_8B,
    "SHISA_V2_LLAMA3_3_70B_FREE": SHISA_V2_LLAMA3_3_70B_FREE,
    "MIDNIGHT_ROSE_70B": MIDNIGHT_ROSE_70B,
    "ANUBIS_PRO_105B_V1": ANUBIS_PRO_105B_V1,
    "ROCINANTE_12B": ROCINANTE_12B,
    "SKYFALL_36B_V2": SKYFALL_36B_V2,
    "UNSLOPNEMO_12B": UNSLOPNEMO_12B,
    "VALKYRIE_49B_V1": VALKYRIE_49B_V1,
    "GLM_4_32B": GLM_4_32B,
    "GLM_4_32B_FREE": GLM_4_32B_FREE,
    "GLM_Z1_32B": GLM_Z1_32B,
    "GLM_Z1_32B_FREE": GLM_Z1_32B_FREE,
    "GLM_Z1_RUMINATION_32B": GLM_Z1_RUMINATION_32B,
    "DEEPSEEK_R1T_CHIMERA_FREE": DEEPSEEK_R1T_CHIMERA_FREE,
    "REMM_SLERP_L2_13B": REMM_SLERP_L2_13B,
    "TOPPY_M_7B": TOPPY_M_7B,
    "GROK_2_1212": GROK_2_1212,
    "GROK_2_VISION_1212": GROK_2_VISION_1212,
    "GROK_3_BETA": GROK_3_BETA,
    "GROK_3_MINI_BETA": GROK_3_MINI_BETA,
    "GROK_BETA": GROK_BETA,
    "GROK_VISION_BETA": GROK_VISION_BETA,
}
