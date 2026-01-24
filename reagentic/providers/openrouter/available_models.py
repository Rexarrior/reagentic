from ..common import ModelInfo


JAMBA_LARGE_1_7 = ModelInfo(
    str_identifier="ai21/jamba-large-1.7",
    price_in=2e-06,
    price_out=8e-06,
    creator="ai21",
    description='''Jamba Large 1.7 is the latest model in the Jamba open family, offering improvements in grounding, instruction-following, and overall efficiency. Built on a hybrid SSM-Transformer architecture with a 256K context window, it delivers more accurate, contextually grounded responses and better steerability than previous versions.,
    created=1754669020'''
)

JAMBA_MINI_1_7 = ModelInfo(
    str_identifier="ai21/jamba-mini-1.7",
    price_in=2e-07,
    price_out=4e-07,
    creator="ai21",
    description='''Jamba Mini 1.7 is a compact and efficient member of the Jamba open model family, incorporating key improvements in grounding and instruction-following while maintaining the benefits of the SSM-Transformer hybrid architecture and 256K context window. Despite its compact size, it delivers accurate, contextually grounded responses and improved steerability.,
    created=1754670601'''
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
    price_in=8e-07,
    price_out=1.6e-06,
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

TONGYI_DEEPRESEARCH_30B_A3B = ModelInfo(
    str_identifier="alibaba/tongyi-deepresearch-30b-a3b",
    price_in=9e-08,
    price_out=4e-07,
    creator="alibaba",
    description='''Tongyi DeepResearch is an agentic large language model developed by Tongyi Lab, with 30 billion total parameters activating only 3 billion per token. It's optimized for long-horizon, deep information-seeking tasks and delivers state-of-the-art performance on benchmarks like Humanity's Last Exam, BrowserComp, BrowserComp-ZH, WebWalkerQA, GAIA, xbench-DeepSearch, and FRAMES. This makes it superior for complex agentic search, reasoning, and multi-step problem-solving compared to prior models.

The model includes a fully automated synthetic data pipeline for scalable pre-training, fine-tuning, and reinforcement learning. It uses large-scale continual pre-training on diverse agentic data to boost reasoning and stay fresh. It also features end-to-end on-policy RL with a customized Group Relative Policy Optimization, including token-level gradients and negative sample filtering for stable training. The model supports ReAct for core ability checks and an IterResearch-based 'Heavy' mode for max performance through test-time scaling. It's ideal for advanced research agents, tool use, and heavy inference workflows.,
    created=1758210804'''
)

MOLMO_2_8B_FREE = ModelInfo(
    str_identifier="allenai/molmo-2-8b:free",
    price_in=0.0,
    price_out=0.0,
    creator="allenai",
    description='''Molmo2-8B is an open vision-language model developed by the Allen Institute for AI (Ai2) as part of the Molmo2 family, supporting image, video, and multi-image understanding and grounding. It is based on Qwen3-8B and uses SigLIP 2 as its vision backbone, outperforming other open-weight, open-data models on short videos, counting, and captioning, while remaining competitive on long-video tasks.,
    created=1767996672'''
)

OLMO_2_0325_32B_INSTRUCT = ModelInfo(
    str_identifier="allenai/olmo-2-0325-32b-instruct",
    price_in=5e-08,
    price_out=2e-07,
    creator="allenai",
    description='''OLMo-2 32B Instruct is a supervised instruction-finetuned variant of the OLMo-2 32B March 2025 base model. It excels in complex reasoning and instruction-following tasks across diverse benchmarks such as GSM8K, MATH, IFEval, and general NLP evaluation. Developed by AI2, OLMo-2 32B is part of an open, research-oriented initiative, trained primarily on English-language datasets to advance the understanding and development of open-source language models.,
    created=1741988556'''
)

OLMO_3_32B_THINK = ModelInfo(
    str_identifier="allenai/olmo-3-32b-think",
    price_in=1.5e-07,
    price_out=5e-07,
    creator="allenai",
    description='''Olmo 3 32B Think is a large-scale, 32-billion-parameter model purpose-built for deep reasoning, complex logic chains and advanced instruction-following scenarios. Its capacity enables strong performance on demanding evaluation tasks and highly nuanced conversational reasoning. Developed by Ai2 under the Apache 2.0 license, Olmo 3 32B Think embodies the Olmo initiative’s commitment to openness, offering full transparency across weights, code and training methodology.,
    created=1763758276'''
)

OLMO_3_7B_INSTRUCT = ModelInfo(
    str_identifier="allenai/olmo-3-7b-instruct",
    price_in=1e-07,
    price_out=2e-07,
    creator="allenai",
    description='''Olmo 3 7B Instruct is a supervised instruction-fine-tuned variant of the Olmo 3 7B base model, optimized for instruction-following, question-answering, and natural conversational dialogue. By leveraging high-quality instruction data and an open training pipeline, it delivers strong performance across everyday NLP tasks while remaining accessible and easy to integrate. Developed by Ai2 under the Apache 2.0 license, the model offers a transparent, community-friendly option for instruction-driven applications.,
    created=1763758273'''
)

OLMO_3_7B_THINK = ModelInfo(
    str_identifier="allenai/olmo-3-7b-think",
    price_in=1.2e-07,
    price_out=2e-07,
    creator="allenai",
    description='''Olmo 3 7B Think is a research-oriented language model in the Olmo family designed for advanced reasoning and instruction-driven tasks. It excels at multi-step problem solving, logical inference, and maintaining coherent conversational context. Developed by Ai2 under the Apache 2.0 license, Olmo 3 7B Think supports transparent, fully open experimentation and provides a lightweight yet capable foundation for academic research and practical NLP workflows.,
    created=1763758270'''
)

OLMO_3_1_32B_INSTRUCT = ModelInfo(
    str_identifier="allenai/olmo-3.1-32b-instruct",
    price_in=2e-07,
    price_out=6e-07,
    creator="allenai",
    description='''Olmo 3.1 32B Instruct is a large-scale, 32-billion-parameter instruction-tuned language model engineered for high-performance conversational AI, multi-turn dialogue, and practical instruction following. As part of the Olmo 3.1 family, this variant emphasizes responsiveness to complex user directions and robust chat interactions while retaining strong capabilities on reasoning and coding benchmarks. Developed by Ai2 under the Apache 2.0 license, Olmo 3.1 32B Instruct reflects the Olmo initiative’s commitment to openness and transparency.,
    created=1767728554'''
)

OLMO_3_1_32B_THINK = ModelInfo(
    str_identifier="allenai/olmo-3.1-32b-think",
    price_in=1.5e-07,
    price_out=5e-07,
    creator="allenai",
    description='''Olmo 3.1 32B Think is a large-scale, 32-billion-parameter model designed for deep reasoning, complex multi-step logic, and advanced instruction following. Building on the Olmo 3 series, version 3.1 delivers refined reasoning behavior and stronger performance across demanding evaluations and nuanced conversational tasks. Developed by Ai2 under the Apache 2.0 license, Olmo 3.1 32B Think continues the Olmo initiative’s commitment to openness, providing full transparency across model weights, code, and training methodology.,
    created=1765907719'''
)

GOLIATH_120B = ModelInfo(
    str_identifier="alpindale/goliath-120b",
    price_in=6e-06,
    price_out=8e-06,
    creator="alpindale",
    description='''A large LLM created by combining two fine-tuned Llama 70B models into one 120B model. Combines Xwin and Euryale.

Credits to
- [@chargoddard](https://huggingface.co/chargoddard) for developing the framework used to merge the model - [mergekit](https://github.com/cg123/mergekit).
- [@Undi95](https://huggingface.co/Undi95) for helping with the merge ratios.

#merge,
    created=1699574400'''
)

NOVA_2_LITE_V1 = ModelInfo(
    str_identifier="amazon/nova-2-lite-v1",
    price_in=3e-07,
    price_out=2.5e-06,
    creator="amazon",
    description='''Nova 2 Lite is a fast, cost-effective reasoning model for everyday workloads that can process text, images, and videos to generate text. 

Nova 2 Lite demonstrates standout capabilities in processing documents, extracting information from videos, generating code, providing accurate grounded answers, and automating multi-step agentic workflows.,
    created=1764696672'''
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

NOVA_PREMIER_V1 = ModelInfo(
    str_identifier="amazon/nova-premier-v1",
    price_in=2.5e-06,
    price_out=1.25e-05,
    creator="amazon",
    description='''Amazon Nova Premier is the most capable of Amazon’s multimodal models for complex reasoning tasks and for use as the best teacher for distilling custom models.,
    created=1761950332'''
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

MAGNUM_V4_72B = ModelInfo(
    str_identifier="anthracite-org/magnum-v4-72b",
    price_in=3e-06,
    price_out=5e-06,
    creator="anthracite-org",
    description='''This is a series of models designed to replicate the prose quality of the Claude 3 models, specifically Sonnet(https://openrouter.ai/anthropic/claude-3.5-sonnet) and Opus(https://openrouter.ai/anthropic/claude-3-opus).

The model is fine-tuned on top of [Qwen2.5 72B](https://openrouter.ai/qwen/qwen-2.5-72b-instruct).,
    created=1729555200'''
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

CLAUDE_3_5_SONNET = ModelInfo(
    str_identifier="anthropic/claude-3.5-sonnet",
    price_in=6e-06,
    price_out=3e-05,
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

CLAUDE_HAIKU_4_5 = ModelInfo(
    str_identifier="anthropic/claude-haiku-4.5",
    price_in=1e-06,
    price_out=5e-06,
    creator="anthropic",
    description='''Claude Haiku 4.5 is Anthropic’s fastest and most efficient model, delivering near-frontier intelligence at a fraction of the cost and latency of larger Claude models. Matching Claude Sonnet 4’s performance across reasoning, coding, and computer-use tasks, Haiku 4.5 brings frontier-level capability to real-time and high-volume applications.

It introduces extended thinking to the Haiku line; enabling controllable reasoning depth, summarized or interleaved thought output, and tool-assisted workflows with full support for coding, bash, web search, and computer-use tools. Scoring >73% on SWE-bench Verified, Haiku 4.5 ranks among the world’s best coding models while maintaining exceptional responsiveness for sub-agents, parallelized execution, and scaled deployment.,
    created=1760547638'''
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

CLAUDE_OPUS_4_1 = ModelInfo(
    str_identifier="anthropic/claude-opus-4.1",
    price_in=1.5e-05,
    price_out=7.5e-05,
    creator="anthropic",
    description='''Claude Opus 4.1 is an updated version of Anthropic’s flagship model, offering improved performance in coding, reasoning, and agentic tasks. It achieves 74.5% on SWE-bench Verified and shows notable gains in multi-file code refactoring, debugging precision, and detail-oriented reasoning. The model supports extended thinking up to 64K tokens and is optimized for tasks involving research, data analysis, and tool-assisted reasoning.,
    created=1754411591'''
)

CLAUDE_OPUS_4_5 = ModelInfo(
    str_identifier="anthropic/claude-opus-4.5",
    price_in=5e-06,
    price_out=2.5e-05,
    creator="anthropic",
    description='''Claude Opus 4.5 is Anthropic’s frontier reasoning model optimized for complex software engineering, agentic workflows, and long-horizon computer use. It offers strong multimodal capabilities, competitive performance across real-world coding and reasoning benchmarks, and improved robustness to prompt injection. The model is designed to operate efficiently across varied effort levels, enabling developers to trade off speed, depth, and token usage depending on task requirements. It comes with a new parameter to control token efficiency, which can be accessed using the OpenRouter Verbosity parameter with low, medium, or high.

Opus 4.5 supports advanced tool use, extended context management, and coordinated multi-agent setups, making it well-suited for autonomous research, debugging, multi-step planning, and spreadsheet/browser manipulation. It delivers substantial gains in structured reasoning, execution reliability, and alignment compared to prior Opus generations, while reducing token overhead and improving performance on long-running tasks.,
    created=1764010580'''
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

CLAUDE_SONNET_4_5 = ModelInfo(
    str_identifier="anthropic/claude-sonnet-4.5",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="anthropic",
    description='''Claude Sonnet 4.5 is Anthropic’s most advanced Sonnet model to date, optimized for real-world agents and coding workflows. It delivers state-of-the-art performance on coding benchmarks such as SWE-bench Verified, with improvements across system design, code security, and specification adherence. The model is designed for extended autonomous operation, maintaining task continuity across sessions and providing fact-based progress tracking.

Sonnet 4.5 also introduces stronger agentic capabilities, including improved tool orchestration, speculative parallel execution, and more efficient context and memory management. With enhanced context tracking and awareness of token usage across tool calls, it is particularly well-suited for multi-context and long-running workflows. Use cases span software engineering, cybersecurity, financial analysis, research agents, and other domains requiring sustained reasoning and tool use.,
    created=1759161676'''
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

TRINITY_MINI = ModelInfo(
    str_identifier="arcee-ai/trinity-mini",
    price_in=4.5e-08,
    price_out=1.5e-07,
    creator="arcee-ai",
    description='''Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language model featuring 128 experts with 8 active per token. Engineered for efficient reasoning over long contexts (131k) with robust function calling and multi-step agent workflows.,
    created=1764601720'''
)

TRINITY_MINI_FREE = ModelInfo(
    str_identifier="arcee-ai/trinity-mini:free",
    price_in=0.0,
    price_out=0.0,
    creator="arcee-ai",
    description='''Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language model featuring 128 experts with 8 active per token. Engineered for efficient reasoning over long contexts (131k) with robust function calling and multi-step agent workflows.,
    created=1764601720'''
)

VIRTUOSO_LARGE = ModelInfo(
    str_identifier="arcee-ai/virtuoso-large",
    price_in=7.5e-07,
    price_out=1.2e-06,
    creator="arcee-ai",
    description='''Virtuoso‑Large is Arcee's top‑tier general‑purpose LLM at 72 B parameters, tuned to tackle cross‑domain reasoning, creative writing and enterprise QA. Unlike many 70 B peers, it retains the 128 k context inherited from Qwen 2.5, letting it ingest books, codebases or financial filings wholesale. Training blended DeepSeek R1 distillation, multi‑epoch supervised fine‑tuning and a final DPO/RLHF alignment stage, yielding strong performance on BIG‑Bench‑Hard, GSM‑8K and long‑context Needle‑In‑Haystack tests. Enterprises use Virtuoso‑Large as the "fallback" brain in Conductor pipelines when other SLMs flag low confidence. Despite its size, aggressive KV‑cache optimizations keep first‑token latency in the low‑second range on 8× H100 nodes, making it a practical production‑grade powerhouse.,
    created=1746478885'''
)

ERNIE_4_5_21B_A3B = ModelInfo(
    str_identifier="baidu/ernie-4.5-21b-a3b",
    price_in=7e-08,
    price_out=2.8e-07,
    creator="baidu",
    description='''A sophisticated text-based Mixture-of-Experts (MoE) model featuring 21B total parameters with 3B activated per token, delivering exceptional multimodal understanding and generation through heterogeneous MoE structures and modality-isolated routing. Supporting an extensive 131K token context length, the model achieves efficient inference via multi-expert parallel collaboration and quantization, while advanced post-training techniques including SFT, DPO, and UPO ensure optimized performance across diverse applications with specialized routing and balancing losses for superior task handling.,
    created=1755034167'''
)

ERNIE_4_5_21B_A3B_THINKING = ModelInfo(
    str_identifier="baidu/ernie-4.5-21b-a3b-thinking",
    price_in=7e-08,
    price_out=2.8e-07,
    creator="baidu",
    description='''ERNIE-4.5-21B-A3B-Thinking is Baidu's upgraded lightweight MoE model, refined to boost reasoning depth and quality for top-tier performance in logical puzzles, math, science, coding, text generation, and expert-level academic benchmarks.,
    created=1760048887'''
)

ERNIE_4_5_300B_A47B = ModelInfo(
    str_identifier="baidu/ernie-4.5-300b-a47b",
    price_in=2.8e-07,
    price_out=1.1e-06,
    creator="baidu",
    description='''ERNIE-4.5-300B-A47B is a 300B parameter Mixture-of-Experts (MoE) language model developed by Baidu as part of the ERNIE 4.5 series. It activates 47B parameters per token and supports text generation in both English and Chinese. Optimized for high-throughput inference and efficient scaling, it uses a heterogeneous MoE structure with advanced routing and quantization strategies, including FP8 and 2-bit formats. This version is fine-tuned for language-only tasks and supports reasoning, tool parameters, and extended context lengths up to 131k tokens. Suitable for general-purpose LLM applications with high reasoning and throughput demands.,
    created=1751300139'''
)

ERNIE_4_5_VL_28B_A3B = ModelInfo(
    str_identifier="baidu/ernie-4.5-vl-28b-a3b",
    price_in=1.4e-07,
    price_out=5.6e-07,
    creator="baidu",
    description='''A powerful multimodal Mixture-of-Experts chat model featuring 28B total parameters with 3B activated per token, delivering exceptional text and vision understanding through its innovative heterogeneous MoE structure with modality-isolated routing. Built with scaling-efficient infrastructure for high-throughput training and inference, the model leverages advanced post-training techniques including SFT, DPO, and UPO for optimized performance, while supporting an impressive 131K context length and RLVR alignment for superior cross-modal reasoning and generation capabilities.,
    created=1755032836'''
)

ERNIE_4_5_VL_424B_A47B = ModelInfo(
    str_identifier="baidu/ernie-4.5-vl-424b-a47b",
    price_in=4.2e-07,
    price_out=1.25e-06,
    creator="baidu",
    description='''ERNIE-4.5-VL-424B-A47B is a multimodal Mixture-of-Experts (MoE) model from Baidu’s ERNIE 4.5 series, featuring 424B total parameters with 47B active per token. It is trained jointly on text and image data using a heterogeneous MoE architecture and modality-isolated routing to enable high-fidelity cross-modal reasoning, image understanding, and long-context generation (up to 131k tokens). Fine-tuned with techniques like SFT, DPO, UPO, and RLVR, this model supports both “thinking” and non-thinking inference modes. Designed for vision-language tasks in English and Chinese, it is optimized for efficient scaling and can operate under 4-bit/8-bit quantization.,
    created=1751300903'''
)

SEED_1_6 = ModelInfo(
    str_identifier="bytedance-seed/seed-1.6",
    price_in=2.5e-07,
    price_out=2e-06,
    creator="bytedance-seed",
    description='''Seed 1.6 is a general-purpose model released by the ByteDance Seed team. It incorporates multimodal capabilities and adaptive deep thinking with a 256K context window.,
    created=1766504997'''
)

SEED_1_6_FLASH = ModelInfo(
    str_identifier="bytedance-seed/seed-1.6-flash",
    price_in=7.5e-08,
    price_out=3e-07,
    creator="bytedance-seed",
    description='''Seed 1.6 Flash is an ultra-fast multimodal deep thinking model by ByteDance Seed, supporting both text and visual understanding. It features a 256k context window and can generate outputs of up to 16k tokens.,
    created=1766505011'''
)

UI_TARS_1_5_7B = ModelInfo(
    str_identifier="bytedance/ui-tars-1.5-7b",
    price_in=1e-07,
    price_out=2e-07,
    creator="bytedance",
    description='''UI-TARS-1.5 is a multimodal vision-language agent optimized for GUI-based environments, including desktop interfaces, web browsers, mobile systems, and games. Built by ByteDance, it builds upon the UI-TARS framework with reinforcement learning-based reasoning, enabling robust action planning and execution across virtual interfaces.

This model achieves state-of-the-art results on a range of interactive and grounding benchmarks, including OSworld, WebVoyager, AndroidWorld, and ScreenSpot. It also demonstrates perfect task completion across diverse Poki games and outperforms prior models in Minecraft agent tasks. UI-TARS-1.5 supports thought decomposition during inference and shows strong scaling across variants, with the 1.5 version notably exceeding the performance of earlier 72B and 7B checkpoints.,
    created=1753205056'''
)

DOLPHIN_MISTRAL_24B_VENICE_EDITION_FREE = ModelInfo(
    str_identifier="cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    price_in=0.0,
    price_out=0.0,
    creator="cognitivecomputations",
    description='''Venice Uncensored Dolphin Mistral 24B Venice Edition is a fine-tuned variant of Mistral-Small-24B-Instruct-2501, developed by dphn.ai in collaboration with Venice.ai. This model is designed as an “uncensored” instruct-tuned LLM, preserving user control over alignment, system prompts, and behavior. Intended for advanced and unrestricted use cases, Venice Uncensored emphasizes steerability and transparent behavior, removing default safety and alignment layers typically found in mainstream assistant models.,
    created=1752094966'''
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

COGITO_V2_PREVIEW_LLAMA_109B_MOE = ModelInfo(
    str_identifier="deepcogito/cogito-v2-preview-llama-109b-moe",
    price_in=1.8e-07,
    price_out=5.9e-07,
    creator="deepcogito",
    description='''An instruction-tuned, hybrid-reasoning Mixture-of-Experts model built on Llama-4-Scout-17B-16E. Cogito v2 can answer directly or engage an extended “thinking” phase, with alignment guided by Iterated Distillation & Amplification (IDA). It targets coding, STEM, instruction following, and general helpfulness, with stronger multilingual, tool-calling, and reasoning performance than size-equivalent baselines. The model supports long-context use (up to 10M tokens) and standard Transformers workflows. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config),
    created=1756831568'''
)

COGITO_V2_PREVIEW_LLAMA_405B = ModelInfo(
    str_identifier="deepcogito/cogito-v2-preview-llama-405b",
    price_in=3.5e-06,
    price_out=3.5e-06,
    creator="deepcogito",
    description='''Cogito v2 405B is a dense hybrid reasoning model that combines direct answering capabilities with advanced self-reflection. It represents a significant step toward frontier intelligence with dense architecture delivering performance competitive with leading closed models. This advanced reasoning system combines policy improvement with massive scale for exceptional capabilities.
,
    created=1760709933'''
)

COGITO_V2_PREVIEW_LLAMA_70B = ModelInfo(
    str_identifier="deepcogito/cogito-v2-preview-llama-70b",
    price_in=8.8e-07,
    price_out=8.8e-07,
    creator="deepcogito",
    description='''Cogito v2 70B is a dense hybrid reasoning model that combines direct answering capabilities with advanced self-reflection. Built with iterative policy improvement, it delivers strong performance across reasoning tasks while maintaining efficiency through shorter reasoning chains and improved intuition.,
    created=1756831784'''
)

COGITO_V2_1_671B = ModelInfo(
    str_identifier="deepcogito/cogito-v2.1-671b",
    price_in=1.25e-06,
    price_out=1.25e-06,
    creator="deepcogito",
    description='''Cogito v2.1 671B MoE represents one of the strongest open models globally, matching performance of frontier closed and open models. This model is trained using self play with reinforcement learning to reach state-of-the-art performance on multiple categories (instruction following, coding, longer queries and creative writing). This advanced system demonstrates significant progress toward scalable superintelligence through policy improvement.,
    created=1763071233'''
)

DEEPSEEK_CHAT = ModelInfo(
    str_identifier="deepseek/deepseek-chat",
    price_in=3e-07,
    price_out=1.2e-06,
    creator="deepseek",
    description='''DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported evaluations reveal that the model outperforms other open-source models and rivals leading closed-source models.

For model details, please visit [the DeepSeek-V3 repo](https://github.com/deepseek-ai/DeepSeek-V3) for more information, or see the [launch announcement](https://api-docs.deepseek.com/news/news1226).,
    created=1735241320'''
)

DEEPSEEK_CHAT_V3_0324 = ModelInfo(
    str_identifier="deepseek/deepseek-chat-v3-0324",
    price_in=1.9e-07,
    price_out=8.7e-07,
    creator="deepseek",
    description='''DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.

It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) model and performs really well on a variety of tasks.,
    created=1742824755'''
)

DEEPSEEK_CHAT_V3_1 = ModelInfo(
    str_identifier="deepseek/deepseek-chat-v3.1",
    price_in=1.5e-07,
    price_out=7.5e-07,
    creator="deepseek",
    description='''DeepSeek-V3.1 is a large hybrid reasoning model (671B parameters, 37B active) that supports both thinking and non-thinking modes via prompt templates. It extends the DeepSeek-V3 base with a two-phase long-context training process, reaching up to 128K tokens, and uses FP8 microscaling for efficient inference. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)

The model improves tool use, code generation, and reasoning efficiency, achieving performance comparable to DeepSeek-R1 on difficult benchmarks while responding more quickly. It supports structured tool calling, code agents, and search agents, making it suitable for research, coding, and agentic workflows. 

It succeeds the [DeepSeek V3-0324](/deepseek/deepseek-chat-v3-0324) model and performs well on a variety of tasks.,
    created=1755779628'''
)

DEEPSEEK_R1 = ModelInfo(
    str_identifier="deepseek/deepseek-r1",
    price_in=7e-07,
    price_out=2.5e-06,
    creator="deepseek",
    description='''DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.

Fully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).

MIT licensed: Distill & commercialize freely!,
    created=1737381095'''
)

DEEPSEEK_R1_0528 = ModelInfo(
    str_identifier="deepseek/deepseek-r1-0528",
    price_in=4e-07,
    price_out=1.75e-06,
    creator="deepseek",
    description='''May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.

Fully open-source model.,
    created=1748455170'''
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
    price_in=3e-08,
    price_out=1.1e-07,
    creator="deepseek",
    description='''DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3.3-70b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:

- AIME 2024 pass@1: 70.0
- MATH-500 pass@1: 94.5
- CodeForces Rating: 1633

The model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1737663169'''
)

DEEPSEEK_R1_DISTILL_QWEN_32B = ModelInfo(
    str_identifier="deepseek/deepseek-r1-distill-qwen-32b",
    price_in=2.9e-07,
    price_out=2.9e-07,
    creator="deepseek",
    description='''DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 72.6\n- MATH-500 pass@1: 94.3\n- CodeForces Rating: 1691\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.,
    created=1738194830'''
)

DEEPSEEK_V3_1_TERMINUS = ModelInfo(
    str_identifier="deepseek/deepseek-v3.1-terminus",
    price_in=2.1e-07,
    price_out=7.9e-07,
    creator="deepseek",
    description='''DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v3.1) that maintains the model's original capabilities while addressing issues reported by users, including language consistency and agent capabilities, further optimizing the model's performance in coding and search agents. It is a large hybrid reasoning model (671B parameters, 37B active) that supports both thinking and non-thinking modes. It extends the DeepSeek-V3 base with a two-phase long-context training process, reaching up to 128K tokens, and uses FP8 microscaling for efficient inference. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)

The model improves tool use, code generation, and reasoning efficiency, achieving performance comparable to DeepSeek-R1 on difficult benchmarks while responding more quickly. It supports structured tool calling, code agents, and search agents, making it suitable for research, coding, and agentic workflows. ,
    created=1758548275'''
)

DEEPSEEK_V3_1_TERMINUS_EXACTO = ModelInfo(
    str_identifier="deepseek/deepseek-v3.1-terminus:exacto",
    price_in=2.1e-07,
    price_out=7.9e-07,
    creator="deepseek",
    description='''DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v3.1) that maintains the model's original capabilities while addressing issues reported by users, including language consistency and agent capabilities, further optimizing the model's performance in coding and search agents. It is a large hybrid reasoning model (671B parameters, 37B active) that supports both thinking and non-thinking modes. It extends the DeepSeek-V3 base with a two-phase long-context training process, reaching up to 128K tokens, and uses FP8 microscaling for efficient inference. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)

The model improves tool use, code generation, and reasoning efficiency, achieving performance comparable to DeepSeek-R1 on difficult benchmarks while responding more quickly. It supports structured tool calling, code agents, and search agents, making it suitable for research, coding, and agentic workflows. ,
    created=1758548275'''
)

DEEPSEEK_V3_2 = ModelInfo(
    str_identifier="deepseek/deepseek-v3.2",
    price_in=2.5e-07,
    price_out=3.8e-07,
    creator="deepseek",
    description='''DeepSeek-V3.2 is a large language model designed to harmonize high computational efficiency with strong reasoning and agentic tool-use performance. It introduces DeepSeek Sparse Attention (DSA), a fine-grained sparse attention mechanism that reduces training and inference cost while preserving quality in long-context scenarios. A scalable reinforcement learning post-training framework further improves reasoning, with reported performance in the GPT-5 class, and the model has demonstrated gold-medal results on the 2025 IMO and IOI. V3.2 also uses a large-scale agentic task synthesis pipeline to better integrate reasoning into tool-use settings, boosting compliance and generalization in interactive environments.

Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config),
    created=1764594642'''
)

DEEPSEEK_V3_2_EXP = ModelInfo(
    str_identifier="deepseek/deepseek-v3.2-exp",
    price_in=2.1e-07,
    price_out=3.2e-07,
    creator="deepseek",
    description='''DeepSeek-V3.2-Exp is an experimental large language model released by DeepSeek as an intermediate step between V3.1 and future architectures. It introduces DeepSeek Sparse Attention (DSA), a fine-grained sparse attention mechanism designed to improve training and inference efficiency in long-context scenarios while maintaining output quality. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)

The model was trained under conditions aligned with V3.1-Terminus to enable direct comparison. Benchmarking shows performance roughly on par with V3.1 across reasoning, coding, and agentic tool-use tasks, with minor tradeoffs and gains depending on the domain. This release focuses on validating architectural optimizations for extended context lengths rather than advancing raw task accuracy, making it primarily a research-oriented model for exploring efficient transformer designs.,
    created=1759150481'''
)

DEEPSEEK_V3_2_SPECIALE = ModelInfo(
    str_identifier="deepseek/deepseek-v3.2-speciale",
    price_in=2.7e-07,
    price_out=4.1e-07,
    creator="deepseek",
    description='''DeepSeek-V3.2-Speciale is a high-compute variant of DeepSeek-V3.2 optimized for maximum reasoning and agentic performance. It builds on DeepSeek Sparse Attention (DSA) for efficient long-context processing, then scales post-training reinforcement learning to push capability beyond the base model. Reported evaluations place Speciale ahead of GPT-5 on difficult reasoning workloads, with proficiency comparable to Gemini-3.0-Pro, while retaining strong coding and tool-use reliability. Like V3.2, it benefits from a large-scale agentic task synthesis pipeline that improves compliance and generalization in interactive environments.,
    created=1764594837'''
)

LLEMMA_7B = ModelInfo(
    str_identifier="eleutherai/llemma_7b",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="eleutherai",
    description='''Llemma 7B is a language model for mathematics. It was initialized with Code Llama 7B weights, and trained on the Proof-Pile-2 for 200B tokens. Llemma models are particularly strong at chain-of-thought mathematical reasoning and using computational tools for mathematics, such as Python and formal theorem provers.,
    created=1744643225'''
)

RNJ_1_INSTRUCT = ModelInfo(
    str_identifier="essentialai/rnj-1-instruct",
    price_in=1.5e-07,
    price_out=1.5e-07,
    creator="essentialai",
    description='''Rnj-1 is an 8B-parameter, dense, open-weight model family developed by Essential AI and trained from scratch with a focus on programming, math, and scientific reasoning. The model demonstrates strong performance across multiple programming languages, tool-use workflows, and agentic execution environments (e.g., mini-SWE-agent). ,
    created=1765094847'''
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

GEMINI_2_5_FLASH = ModelInfo(
    str_identifier="google/gemini-2.5-flash",
    price_in=3e-07,
    price_out=2.5e-06,
    creator="google",
    description='''Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling. 

Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).,
    created=1750172488'''
)

GEMINI_2_5_FLASH_IMAGE = ModelInfo(
    str_identifier="google/gemini-2.5-flash-image",
    price_in=3e-07,
    price_out=2.5e-06,
    creator="google",
    description='''Gemini 2.5 Flash Image, a.k.a. "Nano Banana," is now generally available. It is a state of the art image generation model with contextual understanding. It is capable of image generation, edits, and multi-turn conversations. Aspect ratios can be controlled with the [image_config API Parameter](https://openrouter.ai/docs/features/multimodal/image-generation#image-aspect-ratio-configuration),
    created=1759870431'''
)

GEMINI_2_5_FLASH_LITE = ModelInfo(
    str_identifier="google/gemini-2.5-flash-lite",
    price_in=1e-07,
    price_out=4e-07,
    creator="google",
    description='''Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra-low latency and cost efficiency. It offers improved throughput, faster token generation, and better performance across common benchmarks compared to earlier Flash models. By default, "thinking" (i.e. multi-pass reasoning) is disabled to prioritize speed, but developers can enable it via the [Reasoning API parameter](https://openrouter.ai/docs/use-cases/reasoning-tokens) to selectively trade off cost for intelligence. ,
    created=1753200276'''
)

GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025 = ModelInfo(
    str_identifier="google/gemini-2.5-flash-lite-preview-09-2025",
    price_in=1e-07,
    price_out=4e-07,
    creator="google",
    description='''Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra-low latency and cost efficiency. It offers improved throughput, faster token generation, and better performance across common benchmarks compared to earlier Flash models. By default, "thinking" (i.e. multi-pass reasoning) is disabled to prioritize speed, but developers can enable it via the [Reasoning API parameter](https://openrouter.ai/docs/use-cases/reasoning-tokens) to selectively trade off cost for intelligence. ,
    created=1758819686'''
)

GEMINI_2_5_FLASH_PREVIEW_09_2025 = ModelInfo(
    str_identifier="google/gemini-2.5-flash-preview-09-2025",
    price_in=3e-07,
    price_out=2.5e-06,
    creator="google",
    description='''Gemini 2.5 Flash Preview September 2025 Checkpoint is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks. It includes built-in "thinking" capabilities, enabling it to provide responses with greater accuracy and nuanced context handling. 

Additionally, Gemini 2.5 Flash is configurable through the "max tokens for reasoning" parameter, as described in the documentation (https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning).,
    created=1758820178'''
)

GEMINI_2_5_PRO = ModelInfo(
    str_identifier="google/gemini-2.5-pro",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="google",
    description='''Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities.,
    created=1750169544'''
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

GEMINI_3_FLASH_PREVIEW = ModelInfo(
    str_identifier="google/gemini-3-flash-preview",
    price_in=5e-07,
    price_out=3e-06,
    creator="google",
    description='''Gemini 3 Flash Preview is a high speed, high value thinking model designed for agentic workflows, multi turn chat, and coding assistance. It delivers near Pro level reasoning and tool use performance with substantially lower latency than larger Gemini variants, making it well suited for interactive development, long running agent loops, and collaborative coding tasks. Compared to Gemini 2.5 Flash, it provides broad quality improvements across reasoning, multimodal understanding, and reliability.

The model supports a 1M token context window and multimodal inputs including text, images, audio, video, and PDFs, with text output. It includes configurable reasoning via thinking levels (minimal, low, medium, high), structured output, tool use, and automatic context caching. Gemini 3 Flash Preview is optimized for users who want strong reasoning and agentic behavior without the cost or latency of full scale frontier models.,
    created=1765987078'''
)

GEMINI_3_PRO_IMAGE_PREVIEW = ModelInfo(
    str_identifier="google/gemini-3-pro-image-preview",
    price_in=2e-06,
    price_out=1.2e-05,
    creator="google",
    description='''Nano Banana Pro is Google’s most advanced image-generation and editing model, built on Gemini 3 Pro. It extends the original Nano Banana with significantly improved multimodal reasoning, real-world grounding, and high-fidelity visual synthesis. The model generates context-rich graphics, from infographics and diagrams to cinematic composites, and can incorporate real-time information via Search grounding.

It offers industry-leading text rendering in images (including long passages and multilingual layouts), consistent multi-image blending, and accurate identity preservation across up to five subjects. Nano Banana Pro adds fine-grained creative controls such as localized edits, lighting and focus adjustments, camera transformations, and support for 2K/4K outputs and flexible aspect ratios. It is designed for professional-grade design, product visualization, storyboarding, and complex multi-element compositions while remaining efficient for general image creation workflows.,
    created=1763653797'''
)

GEMINI_3_PRO_PREVIEW = ModelInfo(
    str_identifier="google/gemini-3-pro-preview",
    price_in=2e-06,
    price_out=1.2e-05,
    creator="google",
    description='''Gemini 3 Pro is Google’s flagship frontier model for high-precision multimodal reasoning, combining strong performance across text, image, video, audio, and code with a 1M-token context window. Reasoning Details must be preserved when using multi-turn tool calling, see our docs here: https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks. It delivers state-of-the-art benchmark results in general reasoning, STEM problem solving, factual QA, and multimodal understanding, including leading scores on LMArena, GPQA Diamond, MathArena Apex, MMMU-Pro, and Video-MMMU. Interactions emphasize depth and interpretability: the model is designed to infer intent with minimal prompting and produce direct, insight-focused responses.

Built for advanced development and agentic workflows, Gemini 3 Pro provides robust tool-calling, long-horizon planning stability, and strong zero-shot generation for complex UI, visualization, and coding tasks. It excels at agentic coding (SWE-Bench Verified, Terminal-Bench 2.0), multimodal analysis, and structured long-form tasks such as research synthesis, planning, and interactive learning experiences. Suitable applications include autonomous agents, coding assistants, multimodal analytics, scientific reasoning, and high-context information processing.,
    created=1763474668'''
)

GEMMA_2_27B_IT = ModelInfo(
    str_identifier="google/gemma-2-27b-it",
    price_in=6.5e-07,
    price_out=6.5e-07,
    creator="google",
    description='''Gemma 2 27B by Google is an open model built from the same research and technology used to create the [Gemini models](/models?q=gemini).

Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning.

See the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).,
    created=1720828800'''
)

GEMMA_2_9B_IT = ModelInfo(
    str_identifier="google/gemma-2-9b-it",
    price_in=3e-08,
    price_out=9e-08,
    creator="google",
    description='''Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.

Designed for a wide variety of tasks, it empowers developers and researchers to build innovative applications, while maintaining accessibility, safety, and cost-effectiveness.

See the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).,
    created=1719532800'''
)

GEMMA_3_12B_IT = ModelInfo(
    str_identifier="google/gemma-3-12b-it",
    price_in=3e-08,
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

GEMMA_3_27B_IT = ModelInfo(
    str_identifier="google/gemma-3-27b-it",
    price_in=4e-08,
    price_out=1.5e-07,
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
    price_in=1.703012e-08,
    price_out=6.81536e-08,
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

GEMMA_3N_E2B_IT_FREE = ModelInfo(
    str_identifier="google/gemma-3n-e2b-it:free",
    price_in=0.0,
    price_out=0.0,
    creator="google",
    description='''Gemma 3n E2B IT is a multimodal, instruction-tuned model developed by Google DeepMind, designed to operate efficiently at an effective parameter size of 2B while leveraging a 6B architecture. Based on the MatFormer architecture, it supports nested submodels and modular composition via the Mix-and-Match framework. Gemma 3n models are optimized for low-resource deployment, offering 32K context length and strong multilingual and reasoning performance across common benchmarks. This variant is trained on a diverse corpus including code, math, web, and multimodal data.,
    created=1752074904'''
)

GEMMA_3N_E4B_IT = ModelInfo(
    str_identifier="google/gemma-3n-e4b-it",
    price_in=2e-08,
    price_out=4e-08,
    creator="google",
    description='''Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource devices, such as phones, laptops, and tablets. It supports multimodal inputs—including text, visual data, and audio—enabling diverse tasks such as text generation, speech recognition, translation, and image analysis. Leveraging innovations like Per-Layer Embedding (PLE) caching and the MatFormer architecture, Gemma 3n dynamically manages memory usage and computational load by selectively activating model parameters, significantly reducing runtime resource requirements.

This model supports a wide linguistic range (trained in over 140 languages) and features a flexible 32K token context window. Gemma 3n can selectively load parameters, optimizing memory and computational efficiency based on the task or device capabilities, making it well-suited for privacy-focused, offline-capable applications and on-device AI solutions. [Read more in the blog post](https://developers.googleblog.com/en/introducing-gemma-3n/),
    created=1747776824'''
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
    price_in=6e-08,
    price_out=6e-08,
    creator="gryphe",
    description='''One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge,
    created=1688256000'''
)

GRANITE_4_0_H_MICRO = ModelInfo(
    str_identifier="ibm-granite/granite-4.0-h-micro",
    price_in=1.7e-08,
    price_out=1.1e-07,
    creator="ibm-granite",
    description='''Granite-4.0-H-Micro is a 3B parameter from the Granite 4 family of models. These models are the latest in a series of models released by IBM. They are fine-tuned for long context tool calling. ,
    created=1760927695'''
)

MERCURY = ModelInfo(
    str_identifier="inception/mercury",
    price_in=2.5e-07,
    price_out=1e-06,
    creator="inception",
    description='''Mercury is the first diffusion large language model (dLLM). Applying a breakthrough discrete diffusion approach, the model runs 5-10x faster than even speed optimized models like GPT-4.1 Nano and Claude 3.5 Haiku while matching their performance. Mercury's speed enables developers to provide responsive user experiences, including with voice agents, search interfaces, and chatbots. Read more in the [blog post]
(https://www.inceptionlabs.ai/blog/introducing-mercury) here. ,
    created=1750973026'''
)

MERCURY_CODER = ModelInfo(
    str_identifier="inception/mercury-coder",
    price_in=2.5e-07,
    price_out=1e-06,
    creator="inception",
    description='''Mercury Coder is the first diffusion large language model (dLLM). Applying a breakthrough discrete diffusion approach, the model runs 5-10x faster than even speed optimized models like Claude 3.5 Haiku and GPT-4o Mini while matching their performance. Mercury Coder's speed means that developers can stay in the flow while coding, enjoying rapid chat-based iteration and responsive code completion suggestions. On Copilot Arena, Mercury Coder ranks 1st in speed and ties for 2nd in quality. Read more in the [blog post here](https://www.inceptionlabs.ai/blog/introducing-mercury).,
    created=1746033880'''
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

KAT_CODER_PRO = ModelInfo(
    str_identifier="kwaipilot/kat-coder-pro",
    price_in=2.07e-07,
    price_out=8.28e-07,
    creator="kwaipilot",
    description='''KAT-Coder-Pro V1 is KwaiKAT's most advanced agentic coding model in the KAT-Coder series. Designed specifically for agentic coding tasks, it excels in real-world software engineering scenarios, achieving 73.4% solve rate on the SWE-Bench Verified benchmark. 

The model has been optimized for tool-use capability, multi-turn interaction, instruction following, generalization, and comprehensive capabilities through a multi-stage training process, including mid-training, supervised fine-tuning (SFT), reinforcement fine-tuning (RFT), and scalable agentic RL.,
    created=1762745912'''
)

LFM_2_2_6B = ModelInfo(
    str_identifier="liquid/lfm-2.2-6b",
    price_in=1e-08,
    price_out=2e-08,
    creator="liquid",
    description='''LFM2 is a new generation of hybrid models developed by Liquid AI, specifically designed for edge AI and on-device deployment. It sets a new standard in terms of quality, speed, and memory efficiency.,
    created=1760970889'''
)

LFM_2_5_1_2B_INSTRUCT_FREE = ModelInfo(
    str_identifier="liquid/lfm-2.5-1.2b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="liquid",
    description='''LFM2.5-1.2B-Instruct is a compact, high-performance instruction-tuned model built for fast on-device AI. It delivers strong chat quality in a 1.2B parameter footprint, with efficient edge inference and broad runtime support.,
    created=1768927521'''
)

LFM_2_5_1_2B_THINKING_FREE = ModelInfo(
    str_identifier="liquid/lfm-2.5-1.2b-thinking:free",
    price_in=0.0,
    price_out=0.0,
    creator="liquid",
    description='''LFM2.5-1.2B-Thinking is a lightweight reasoning-focused model optimized for agentic tasks, data extraction, and RAG—while still running comfortably on edge devices. It supports long context (up to 32K tokens) and is designed to provide higher-quality “thinking” responses in a small 1.2B model.,
    created=1768927527'''
)

LFM2_8B_A1B = ModelInfo(
    str_identifier="liquid/lfm2-8b-a1b",
    price_in=1e-08,
    price_out=2e-08,
    creator="liquid",
    description='''LFM2-8B-A1B is an efficient on-device Mixture-of-Experts (MoE) model from Liquid AI’s LFM2 family, built for fast, high-quality inference on edge hardware. It uses 8.3B total parameters with only ~1.5B active per token, delivering strong performance while keeping compute and memory usage low—making it ideal for phones, tablets, and laptops.,
    created=1760970984'''
)

WEAVER = ModelInfo(
    str_identifier="mancer/weaver",
    price_in=7.5e-07,
    price_out=1e-06,
    creator="mancer",
    description='''An attempt to recreate Claude-style verbosity, but don't expect the same level of coherence or memory. Meant for use in roleplay/narrative situations.,
    created=1690934400'''
)

LONGCAT_FLASH_CHAT = ModelInfo(
    str_identifier="meituan/longcat-flash-chat",
    price_in=2e-07,
    price_out=8e-07,
    creator="meituan",
    description='''LongCat-Flash-Chat is a large-scale Mixture-of-Experts (MoE) model with 560B total parameters, of which 18.6B–31.3B (≈27B on average) are dynamically activated per input. It introduces a shortcut-connected MoE design to reduce communication overhead and achieve high throughput while maintaining training stability through advanced scaling strategies such as hyperparameter transfer, deterministic computation, and multi-stage optimization.

This release, LongCat-Flash-Chat, is a non-thinking foundation model optimized for conversational and agentic tasks. It supports long context windows up to 128K tokens and shows competitive performance across reasoning, coding, instruction following, and domain benchmarks, with particular strengths in tool use and complex multi-step interactions.,
    created=1757427658'''
)

LLAMA_3_70B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3-70b-instruct",
    price_in=4e-07,
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
    price_in=4e-06,
    price_out=4e-06,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This is the base 405B pre-trained version.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1722556800'''
)

LLAMA_3_1_405B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.1-405b-instruct",
    price_in=3.5e-06,
    price_out=3.5e-06,
    creator="meta-llama",
    description='''The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.

Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 405B instruct-tuned version is optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models including GPT-4o and Claude 3.5 Sonnet in evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_1_405B_INSTRUCT_FREE = ModelInfo(
    str_identifier="meta-llama/llama-3.1-405b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="meta-llama",
    description='''The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.

Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 405B instruct-tuned version is optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models including GPT-4o and Claude 3.5 Sonnet in evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_1_70B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.1-70b-instruct",
    price_in=4e-07,
    price_out=4e-07,
    creator="meta-llama",
    description='''Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instruct-tuned version is optimized for high quality dialogue usecases.

It has demonstrated strong performance compared to leading closed-source models in human evaluations.

To read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1721692800'''
)

LLAMA_3_1_8B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.1-8b-instruct",
    price_in=2e-08,
    price_out=5e-08,
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

LLAMA_3_2_1B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.2-1b-instruct",
    price_in=2.7e-08,
    price_out=2e-07,
    creator="meta-llama",
    description='''Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural language tasks, such as summarization, dialogue, and multilingual text analysis. Its smaller size allows it to operate efficiently in low-resource environments while maintaining strong task performance.

Supporting eight core languages and fine-tunable for more, Llama 1.3B is ideal for businesses or developers seeking lightweight yet powerful AI solutions that can operate in diverse multilingual settings without the high computational demand of larger models.

Click here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).

Usage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/).,
    created=1727222400'''
)

LLAMA_3_2_3B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.2-3b-instruct",
    price_in=2e-08,
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

LLAMA_3_3_70B_INSTRUCT = ModelInfo(
    str_identifier="meta-llama/llama-3.3-70b-instruct",
    price_in=1e-07,
    price_out=3.2e-07,
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

LLAMA_4_MAVERICK = ModelInfo(
    str_identifier="meta-llama/llama-4-maverick",
    price_in=1.5e-07,
    price_out=6e-07,
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
    price_in=1.8e-07,
    price_out=1.8e-07,
    creator="meta-llama",
    description='''Llama Guard 4 is a Llama 4 Scout-derived multimodal pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM—generating text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 4 was aligned to safeguard against the standardized MLCommons hazards taxonomy and designed to support multimodal Llama 4 capabilities. Specifically, it combines features from previous Llama Guard models, providing content moderation for English and multiple supported languages, along with enhanced capabilities to handle mixed text-and-image prompts, including multiple images. Additionally, Llama Guard 4 is integrated into the Llama Moderations API, extending robust safety classification to text and images.,
    created=1745975193'''
)

PHI_4 = ModelInfo(
    str_identifier="microsoft/phi-4",
    price_in=6e-08,
    price_out=1.4e-07,
    creator="microsoft",
    description='''[Microsoft Research](/microsoft) Phi-4 is designed to perform well in complex reasoning tasks and can operate efficiently in situations with limited memory or where quick responses are needed. 

At 14 billion parameters, it was trained on a mix of high-quality synthetic datasets, data from curated websites, and academic materials. It has undergone careful improvement to follow instructions accurately and maintain strong safety standards. It works best with English language inputs.

For more information, please see [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905)
,
    created=1736489872'''
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

MINIMAX_M1 = ModelInfo(
    str_identifier="minimax/minimax-m1",
    price_in=4e-07,
    price_out=2.2e-06,
    creator="minimax",
    description='''MiniMax-M1 is a large-scale, open-weight reasoning model designed for extended context and high-efficiency inference. It leverages a hybrid Mixture-of-Experts (MoE) architecture paired with a custom "lightning attention" mechanism, allowing it to process long sequences—up to 1 million tokens—while maintaining competitive FLOP efficiency. With 456 billion total parameters and 45.9B active per token, this variant is optimized for complex, multi-step reasoning tasks.

Trained via a custom reinforcement learning pipeline (CISPO), M1 excels in long-context understanding, software engineering, agentic tool use, and mathematical reasoning. Benchmarks show strong performance across FullStackBench, SWE-bench, MATH, GPQA, and TAU-Bench, often outperforming other open models like DeepSeek R1 and Qwen3-235B.,
    created=1750200414'''
)

MINIMAX_M2 = ModelInfo(
    str_identifier="minimax/minimax-m2",
    price_in=2e-07,
    price_out=1e-06,
    creator="minimax",
    description='''MiniMax-M2 is a compact, high-efficiency large language model optimized for end-to-end coding and agentic workflows. With 10 billion activated parameters (230 billion total), it delivers near-frontier intelligence across general reasoning, tool use, and multi-step task execution while maintaining low latency and deployment efficiency.

The model excels in code generation, multi-file editing, compile-run-fix loops, and test-validated repair, showing strong results on SWE-Bench Verified, Multi-SWE-Bench, and Terminal-Bench. It also performs competitively in agentic evaluations such as BrowseComp and GAIA, effectively handling long-horizon planning, retrieval, and recovery from execution errors.

Benchmarked by [Artificial Analysis](https://artificialanalysis.ai/models/minimax-m2), MiniMax-M2 ranks among the top open-source models for composite intelligence, spanning mathematics, science, and instruction-following. Its small activation footprint enables fast inference, high concurrency, and improved unit economics, making it well-suited for large-scale agents, developer assistants, and reasoning-driven applications that require responsiveness and cost efficiency.

To avoid degrading this model's performance, MiniMax highly recommends preserving reasoning between turns. Learn more about using reasoning_details to pass back reasoning in our [docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks).,
    created=1761252093'''
)

MINIMAX_M2_HER = ModelInfo(
    str_identifier="minimax/minimax-m2-her",
    price_in=3e-07,
    price_out=1.2e-06,
    creator="minimax",
    description='''MiniMax M2-her is a dialogue-first large language model built for immersive roleplay, character-driven chat, and expressive multi-turn conversations. Designed to stay consistent in tone and personality, it supports rich message roles (user_system, group, sample_message_user, sample_message_ai) and can learn from example dialogue to better match the style and pacing of your scenario, making it a strong choice for storytelling, companions, and conversational experiences where natural flow and vivid interaction matter most.,
    created=1769177239'''
)

MINIMAX_M2_1 = ModelInfo(
    str_identifier="minimax/minimax-m2.1",
    price_in=2.7e-07,
    price_out=1.1e-06,
    creator="minimax",
    description='''MiniMax-M2.1 is a lightweight, state-of-the-art large language model optimized for coding, agentic workflows, and modern application development. With only 10 billion activated parameters, it delivers a major jump in real-world capability while maintaining exceptional latency, scalability, and cost efficiency.

Compared to its predecessor, M2.1 delivers cleaner, more concise outputs and faster perceived response times. It shows leading multilingual coding performance across major systems and application languages, achieving 49.4% on Multi-SWE-Bench and 72.5% on SWE-Bench Multilingual, and serves as a versatile agent “brain” for IDEs, coding tools, and general-purpose assistance.

To avoid degrading this model's performance, MiniMax highly recommends preserving reasoning between turns. Learn more about using reasoning_details to pass back reasoning in our [docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks).,
    created=1766454997'''
)

CODESTRAL_2508 = ModelInfo(
    str_identifier="mistralai/codestral-2508",
    price_in=3e-07,
    price_out=9e-07,
    creator="mistralai",
    description='''Mistral's cutting-edge language model for coding released end of July 2025. Codestral specializes in low-latency, high-frequency tasks such as fill-in-the-middle (FIM), code correction and test generation.

[Blog Post](https://mistral.ai/news/codestral-25-08),
    created=1754079630'''
)

DEVSTRAL_2512 = ModelInfo(
    str_identifier="mistralai/devstral-2512",
    price_in=5e-08,
    price_out=2.2e-07,
    creator="mistralai",
    description='''Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in agentic coding. It is a 123B-parameter dense transformer model supporting a 256K context window.

Devstral 2 supports exploring codebases and orchestrating changes across multiple files while maintaining architecture-level context. It tracks framework dependencies, detects failures, and retries with corrections—solving challenges like bug fixing and modernizing legacy systems. The model can be fine-tuned to prioritize specific languages or optimize for large enterprise codebases. It is available under a modified MIT license.,
    created=1765285419'''
)

DEVSTRAL_2512_FREE = ModelInfo(
    str_identifier="mistralai/devstral-2512:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in agentic coding. It is a 123B-parameter dense transformer model supporting a 256K context window.

Devstral 2 supports exploring codebases and orchestrating changes across multiple files while maintaining architecture-level context. It tracks framework dependencies, detects failures, and retries with corrections—solving challenges like bug fixing and modernizing legacy systems. The model can be fine-tuned to prioritize specific languages or optimize for large enterprise codebases. It is available under a modified MIT license.,
    created=1765285419'''
)

DEVSTRAL_MEDIUM = ModelInfo(
    str_identifier="mistralai/devstral-medium",
    price_in=4e-07,
    price_out=2e-06,
    creator="mistralai",
    description='''Devstral Medium is a high-performance code generation and agentic reasoning model developed jointly by Mistral AI and All Hands AI. Positioned as a step up from Devstral Small, it achieves 61.6% on SWE-Bench Verified, placing it ahead of Gemini 2.5 Pro and GPT-4.1 in code-related tasks, at a fraction of the cost. It is designed for generalization across prompt styles and tool use in code agents and frameworks.

Devstral Medium is available via API only (not open-weight), and supports enterprise deployment on private infrastructure, with optional fine-tuning capabilities.,
    created=1752161321'''
)

DEVSTRAL_SMALL = ModelInfo(
    str_identifier="mistralai/devstral-small",
    price_in=1e-07,
    price_out=3e-07,
    creator="mistralai",
    description='''Devstral Small 1.1 is a 24B parameter open-weight language model for software engineering agents, developed by Mistral AI in collaboration with All Hands AI. Finetuned from Mistral Small 3.1 and released under the Apache 2.0 license, it features a 128k token context window and supports both Mistral-style function calling and XML output formats.

Designed for agentic coding workflows, Devstral Small 1.1 is optimized for tasks such as codebase exploration, multi-file edits, and integration into autonomous development agents like OpenHands and Cline. It achieves 53.6% on SWE-Bench Verified, surpassing all other open models on this benchmark, while remaining lightweight enough to run on a single 4090 GPU or Apple silicon machine. The model uses a Tekken tokenizer with a 131k vocabulary and is deployable via vLLM, Transformers, Ollama, LM Studio, and other OpenAI-compatible runtimes.
,
    created=1752160751'''
)

MINISTRAL_14B_2512 = ModelInfo(
    str_identifier="mistralai/ministral-14b-2512",
    price_in=2e-07,
    price_out=2e-07,
    creator="mistralai",
    description='''The largest model in the Ministral 3 family, Ministral 3 14B offers frontier capabilities and performance comparable to its larger Mistral Small 3.2 24B counterpart. A powerful and efficient language model with vision capabilities.,
    created=1764681735'''
)

MINISTRAL_3B = ModelInfo(
    str_identifier="mistralai/ministral-3b",
    price_in=4e-08,
    price_out=4e-08,
    creator="mistralai",
    description='''Ministral 3B is a 3B parameter model optimized for on-device and edge computing. It excels in knowledge, commonsense reasoning, and function-calling, outperforming larger models like Mistral 7B on most benchmarks. Supporting up to 128k context length, it’s ideal for orchestrating agentic workflows and specialist tasks with efficient inference.,
    created=1729123200'''
)

MINISTRAL_3B_2512 = ModelInfo(
    str_identifier="mistralai/ministral-3b-2512",
    price_in=1e-07,
    price_out=1e-07,
    creator="mistralai",
    description='''The smallest model in the Ministral 3 family, Ministral 3 3B is a powerful, efficient tiny language model with vision capabilities.,
    created=1764681560'''
)

MINISTRAL_8B = ModelInfo(
    str_identifier="mistralai/ministral-8b",
    price_in=1e-07,
    price_out=1e-07,
    creator="mistralai",
    description='''Ministral 8B is an 8B parameter model featuring a unique interleaved sliding-window attention pattern for faster, memory-efficient inference. Designed for edge use cases, it supports up to 128k context length and excels in knowledge and reasoning tasks. It outperforms peers in the sub-10B category, making it perfect for low-latency, privacy-first applications.,
    created=1729123200'''
)

MINISTRAL_8B_2512 = ModelInfo(
    str_identifier="mistralai/ministral-8b-2512",
    price_in=1.5e-07,
    price_out=1.5e-07,
    creator="mistralai",
    description='''A balanced model in the Ministral 3 family, Ministral 3 8B is a powerful, efficient tiny language model with vision capabilities.,
    created=1764681654'''
)

MISTRAL_7B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mistral-7b-instruct",
    price_in=2e-07,
    price_out=2e-07,
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
    price_in=2e-07,
    price_out=2e-07,
    creator="mistralai",
    description='''A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.

An improved version of [Mistral 7B Instruct v0.2](/models/mistralai/mistral-7b-instruct-v0.2), with the following changes:

- Extended vocabulary to 32768
- Supports v3 Tokenizer
- Supports function calling

NOTE: Support for function calling depends on the provider.,
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

MISTRAL_LARGE_2512 = ModelInfo(
    str_identifier="mistralai/mistral-large-2512",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="mistralai",
    description='''Mistral Large 3 2512 is Mistral’s most capable model to date, featuring a sparse mixture-of-experts architecture with 41B active parameters (675B total), and released under the Apache 2.0 license.,
    created=1764624472'''
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

MISTRAL_MEDIUM_3_1 = ModelInfo(
    str_identifier="mistralai/mistral-medium-3.1",
    price_in=4e-07,
    price_out=2e-06,
    creator="mistralai",
    description='''Mistral Medium 3.1 is an updated version of Mistral Medium 3, which is a high-performance enterprise-grade language model designed to deliver frontier-level capabilities at significantly reduced operational cost. It balances state-of-the-art reasoning and multimodal performance with 8× lower cost compared to traditional large models, making it suitable for scalable deployments across professional and industrial use cases.

The model excels in domains such as coding, STEM reasoning, and enterprise adaptation. It supports hybrid, on-prem, and in-VPC deployments and is optimized for integration into custom workflows. Mistral Medium 3.1 offers competitive accuracy relative to larger models like Claude Sonnet 3.5/3.7, Llama 4 Maverick, and Command R+, while maintaining broad compatibility across cloud environments.,
    created=1755095639'''
)

MISTRAL_NEMO = ModelInfo(
    str_identifier="mistralai/mistral-nemo",
    price_in=2e-08,
    price_out=4e-08,
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

MISTRAL_SMALL_24B_INSTRUCT_2501 = ModelInfo(
    str_identifier="mistralai/mistral-small-24b-instruct-2501",
    price_in=3e-08,
    price_out=1.1e-07,
    creator="mistralai",
    description='''Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained and instruction-tuned versions designed for efficient local deployment.

The model achieves 81% accuracy on the MMLU benchmark and performs competitively with larger models like Llama 3.3 70B and Qwen 32B, while operating at three times the speed on equivalent hardware. [Read the blog post about the model here.](https://mistral.ai/news/mistral-small-3/),
    created=1738255409'''
)

MISTRAL_SMALL_3_1_24B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mistral-small-3.1-24b-instruct",
    price_in=3e-08,
    price_out=1.1e-07,
    creator="mistralai",
    description='''Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational agents, function calling, long-document comprehension, and privacy-sensitive deployments. The updated version is [Mistral Small 3.2](mistralai/mistral-small-3.2-24b-instruct),
    created=1742238937'''
)

MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE = ModelInfo(
    str_identifier="mistralai/mistral-small-3.1-24b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="mistralai",
    description='''Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational agents, function calling, long-document comprehension, and privacy-sensitive deployments. The updated version is [Mistral Small 3.2](mistralai/mistral-small-3.2-24b-instruct),
    created=1742238937'''
)

MISTRAL_SMALL_3_2_24B_INSTRUCT = ModelInfo(
    str_identifier="mistralai/mistral-small-3.2-24b-instruct",
    price_in=6e-08,
    price_out=1.8e-07,
    creator="mistralai",
    description='''Mistral-Small-3.2-24B-Instruct-2506 is an updated 24B parameter model from Mistral optimized for instruction following, repetition reduction, and improved function calling. Compared to the 3.1 release, version 3.2 significantly improves accuracy on WildBench and Arena Hard, reduces infinite generations, and delivers gains in tool use and structured output tasks.

It supports image and text inputs with structured outputs, function/tool calling, and strong performance across coding (HumanEval+, MBPP), STEM (MMLU, MATH, GPQA), and vision benchmarks (ChartQA, DocVQA).,
    created=1750443016'''
)

MISTRAL_SMALL_CREATIVE = ModelInfo(
    str_identifier="mistralai/mistral-small-creative",
    price_in=1e-07,
    price_out=3e-07,
    creator="mistralai",
    description='''Mistral Small Creative is an experimental small model designed for creative writing, narrative generation, roleplay and character-driven dialogue, general-purpose instruction following, and conversational agents.,
    created=1765908653'''
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
    price_in=2e-06,
    price_out=6e-06,
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
    price_in=5.4e-07,
    price_out=5.4e-07,
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

VOXTRAL_SMALL_24B_2507 = ModelInfo(
    str_identifier="mistralai/voxtral-small-24b-2507",
    price_in=1e-07,
    price_out=3e-07,
    creator="mistralai",
    description='''Voxtral Small is an enhancement of Mistral Small 3, incorporating state-of-the-art audio input capabilities while retaining best-in-class text performance. It excels at speech transcription, translation and audio understanding. Input audio is priced at $100 per million seconds.,
    created=1761835144'''
)

KIMI_DEV_72B = ModelInfo(
    str_identifier="moonshotai/kimi-dev-72b",
    price_in=2.9e-07,
    price_out=1.15e-06,
    creator="moonshotai",
    description='''Kimi-Dev-72B is an open-source large language model fine-tuned for software engineering and issue resolution tasks. Based on Qwen2.5-72B, it is optimized using large-scale reinforcement learning that applies code patches in real repositories and validates them via full test suite execution—rewarding only correct, robust completions. The model achieves 60.4% on SWE-bench Verified, setting a new benchmark among open-source models for software bug fixing and code reasoning.,
    created=1750115909'''
)

KIMI_K2 = ModelInfo(
    str_identifier="moonshotai/kimi-k2",
    price_in=5e-07,
    price_out=2.4e-06,
    creator="moonshotai",
    description='''Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It is optimized for agentic capabilities, including advanced tool use, reasoning, and code synthesis. Kimi K2 excels across a broad range of benchmarks, particularly in coding (LiveCodeBench, SWE-bench), reasoning (ZebraLogic, GPQA), and tool-use (Tau2, AceBench) tasks. It supports long-context inference up to 128K tokens and is designed with a novel training stack that includes the MuonClip optimizer for stable large-scale MoE training.,
    created=1752263252'''
)

KIMI_K2_0905 = ModelInfo(
    str_identifier="moonshotai/kimi-k2-0905",
    price_in=3.9e-07,
    price_out=1.9e-06,
    creator="moonshotai",
    description='''Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It supports long-context inference up to 256k tokens, extended from the previous 128k.

This update improves agentic coding with higher accuracy and better generalization across scaffolds, and enhances frontend coding with more aesthetic and functional outputs for web, 3D, and related tasks. Kimi K2 is optimized for agentic capabilities, including advanced tool use, reasoning, and code synthesis. It excels across coding (LiveCodeBench, SWE-bench), reasoning (ZebraLogic, GPQA), and tool-use (Tau2, AceBench) benchmarks. The model is trained with a novel stack incorporating the MuonClip optimizer for stable large-scale MoE training.,
    created=1757021147'''
)

KIMI_K2_0905_EXACTO = ModelInfo(
    str_identifier="moonshotai/kimi-k2-0905:exacto",
    price_in=6e-07,
    price_out=2.5e-06,
    creator="moonshotai",
    description='''Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It supports long-context inference up to 256k tokens, extended from the previous 128k.

This update improves agentic coding with higher accuracy and better generalization across scaffolds, and enhances frontend coding with more aesthetic and functional outputs for web, 3D, and related tasks. Kimi K2 is optimized for agentic capabilities, including advanced tool use, reasoning, and code synthesis. It excels across coding (LiveCodeBench, SWE-bench), reasoning (ZebraLogic, GPQA), and tool-use (Tau2, AceBench) benchmarks. The model is trained with a novel stack incorporating the MuonClip optimizer for stable large-scale MoE training.,
    created=1757021147'''
)

KIMI_K2_THINKING = ModelInfo(
    str_identifier="moonshotai/kimi-k2-thinking",
    price_in=4e-07,
    price_out=1.75e-06,
    creator="moonshotai",
    description='''Kimi K2 Thinking is Moonshot AI’s most advanced open reasoning model to date, extending the K2 series into agentic, long-horizon reasoning. Built on the trillion-parameter Mixture-of-Experts (MoE) architecture introduced in Kimi K2, it activates 32 billion parameters per forward pass and supports 256 k-token context windows. The model is optimized for persistent step-by-step thought, dynamic tool invocation, and complex reasoning workflows that span hundreds of turns. It interleaves step-by-step reasoning with tool use, enabling autonomous research, coding, and writing that can persist for hundreds of sequential actions without drift.

It sets new open-source benchmarks on HLE, BrowseComp, SWE-Multilingual, and LiveCodeBench, while maintaining stable multi-agent behavior through 200–300 tool calls. Built on a large-scale MoE architecture with MuonClip optimization, it combines strong reasoning depth with high inference efficiency for demanding agentic and analytical tasks.,
    created=1762440622'''
)

KIMI_K2_FREE = ModelInfo(
    str_identifier="moonshotai/kimi-k2:free",
    price_in=0.0,
    price_out=0.0,
    creator="moonshotai",
    description='''Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It is optimized for agentic capabilities, including advanced tool use, reasoning, and code synthesis. Kimi K2 excels across a broad range of benchmarks, particularly in coding (LiveCodeBench, SWE-bench), reasoning (ZebraLogic, GPQA), and tool-use (Tau2, AceBench) tasks. It supports long-context inference up to 128K tokens and is designed with a novel training stack that includes the MuonClip optimizer for stable large-scale MoE training.,
    created=1752263252'''
)

MORPH_V3_FAST = ModelInfo(
    str_identifier="morph/morph-v3-fast",
    price_in=8e-07,
    price_out=1.2e-06,
    creator="morph",
    description='''Morph's fastest apply model for code edits. ~10,500 tokens/sec with 96% accuracy for rapid code transformations.

The model requires the prompt to be in the following format: 
<instruction>{instruction}</instruction>
<code>{initial_code}</code>
<update>{edit_snippet}</update>

Zero Data Retention is enabled for Morph. Learn more about this model in their [documentation](https://docs.morphllm.com/quickstart),
    created=1751910002'''
)

MORPH_V3_LARGE = ModelInfo(
    str_identifier="morph/morph-v3-large",
    price_in=9e-07,
    price_out=1.9e-06,
    creator="morph",
    description='''Morph's high-accuracy apply model for complex code edits. ~4,500 tokens/sec with 98% accuracy for precise code transformations.

The model requires the prompt to be in the following format: 
<instruction>{instruction}</instruction>
<code>{initial_code}</code>
<update>{edit_snippet}</update>

Zero Data Retention is enabled for Morph. Learn more about this model in their [documentation](https://docs.morphllm.com/quickstart),
    created=1751910858'''
)

LLAMA_3_1_LUMIMAID_8B = ModelInfo(
    str_identifier="neversleep/llama-3.1-lumimaid-8b",
    price_in=9e-08,
    price_out=6e-07,
    creator="neversleep",
    description='''Lumimaid v0.2 8B is a finetune of [Llama 3.1 8B](/models/meta-llama/llama-3.1-8b-instruct) with a "HUGE step up dataset wise" compared to Lumimaid v0.1. Sloppy chats output were purged.

Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).,
    created=1726358400'''
)

NOROMAID_20B = ModelInfo(
    str_identifier="neversleep/noromaid-20b",
    price_in=1e-06,
    price_out=1.75e-06,
    creator="neversleep",
    description='''A collab between IkariDev and Undi. This merge is suitable for RP, ERP, and general knowledge.

#merge #uncensored,
    created=1700956800'''
)

DEEPSEEK_V3_1_NEX_N1 = ModelInfo(
    str_identifier="nex-agi/deepseek-v3.1-nex-n1",
    price_in=2.7e-07,
    price_out=1e-06,
    creator="nex-agi",
    description='''DeepSeek V3.1 Nex-N1 is the flagship release of the Nex-N1 series — a post-trained model designed to highlight agent autonomy, tool use, and real-world productivity. 

Nex-N1 demonstrates competitive performance across all evaluation scenarios, showing particularly strong results in practical coding and HTML generation tasks.,
    created=1765204393'''
)

DEEPHERMES_3_MISTRAL_24B_PREVIEW = ModelInfo(
    str_identifier="nousresearch/deephermes-3-mistral-24b-preview",
    price_in=2e-08,
    price_out=1e-07,
    creator="nousresearch",
    description='''DeepHermes 3 (Mistral 24B Preview) is an instruction-tuned language model by Nous Research based on Mistral-Small-24B, designed for chat, function calling, and advanced multi-turn reasoning. It introduces a dual-mode system that toggles between intuitive chat responses and structured “deep reasoning” mode using special system prompts. Fine-tuned via distillation from R1, it supports structured output (JSON mode) and function call syntax for agent-based applications.

DeepHermes 3 supports a **reasoning toggle via system prompt**, allowing users to switch between fast, intuitive responses and deliberate, multi-step reasoning. When activated with the following specific system instruction, the model enters a *"deep thinking"* mode—generating extended chains of thought wrapped in `<think></think>` tags before delivering a final answer. 

System Prompt: You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
,
    created=1746830904'''
)

HERMES_2_PRO_LLAMA_3_8B = ModelInfo(
    str_identifier="nousresearch/hermes-2-pro-llama-3-8b",
    price_in=1.4e-07,
    price_out=1.4e-07,
    creator="nousresearch",
    description='''Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house.,
    created=1716768000'''
)

HERMES_3_LLAMA_3_1_405B = ModelInfo(
    str_identifier="nousresearch/hermes-3-llama-3.1-405b",
    price_in=1e-06,
    price_out=1e-06,
    creator="nousresearch",
    description='''Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.

Hermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.

The Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills.

Hermes 3 is competitive, if not superior, to Llama-3.1 Instruct models at general capabilities, with varying strengths and weaknesses attributable between the two.,
    created=1723766400'''
)

HERMES_3_LLAMA_3_1_405B_FREE = ModelInfo(
    str_identifier="nousresearch/hermes-3-llama-3.1-405b:free",
    price_in=0.0,
    price_out=0.0,
    creator="nousresearch",
    description='''Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.

Hermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.

The Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills.

Hermes 3 is competitive, if not superior, to Llama-3.1 Instruct models at general capabilities, with varying strengths and weaknesses attributable between the two.,
    created=1723766400'''
)

HERMES_3_LLAMA_3_1_70B = ModelInfo(
    str_identifier="nousresearch/hermes-3-llama-3.1-70b",
    price_in=3e-07,
    price_out=3e-07,
    creator="nousresearch",
    description='''Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.

Hermes 3 70B is a competitive, if not superior finetune of the [Llama-3.1 70B foundation model](/models/meta-llama/llama-3.1-70b-instruct), focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.

The Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills.,
    created=1723939200'''
)

HERMES_4_405B = ModelInfo(
    str_identifier="nousresearch/hermes-4-405b",
    price_in=1e-06,
    price_out=3e-06,
    creator="nousresearch",
    description='''Hermes 4 is a large-scale reasoning model built on Meta-Llama-3.1-405B and released by Nous Research. It introduces a hybrid reasoning mode, where the model can choose to deliberate internally with <think>...</think> traces or respond directly, offering flexibility between speed and depth. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)

The model is instruction-tuned with an expanded post-training corpus (~60B tokens) emphasizing reasoning traces, improving performance in math, code, STEM, and logical reasoning, while retaining broad assistant utility. It also supports structured outputs, including JSON mode, schema adherence, function calling, and tool use. Hermes 4 is trained for steerability, lower refusal rates, and alignment toward neutral, user-directed behavior.,
    created=1756235463'''
)

HERMES_4_70B = ModelInfo(
    str_identifier="nousresearch/hermes-4-70b",
    price_in=1.1e-07,
    price_out=3.8e-07,
    creator="nousresearch",
    description='''Hermes 4 70B is a hybrid reasoning model from Nous Research, built on Meta-Llama-3.1-70B. It introduces the same hybrid mode as the larger 405B release, allowing the model to either respond directly or generate explicit <think>...</think> reasoning traces before answering. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config)

This 70B variant is trained with the expanded post-training corpus (~60B tokens) emphasizing verified reasoning data, leading to improvements in mathematics, coding, STEM, logic, and structured outputs while maintaining general assistant performance. It supports JSON mode, schema adherence, function calling, and tool use, and is designed for greater steerability with reduced refusal rates.,
    created=1756236182'''
)

LLAMA_3_1_NEMOTRON_70B_INSTRUCT = ModelInfo(
    str_identifier="nvidia/llama-3.1-nemotron-70b-instruct",
    price_in=1.2e-06,
    price_out=1.2e-06,
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

LLAMA_3_3_NEMOTRON_SUPER_49B_V1_5 = ModelInfo(
    str_identifier="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    price_in=1e-07,
    price_out=4e-07,
    creator="nvidia",
    description='''Llama-3.3-Nemotron-Super-49B-v1.5 is a 49B-parameter, English-centric reasoning/chat model derived from Meta’s Llama-3.3-70B-Instruct with a 128K context. It’s post-trained for agentic workflows (RAG, tool calling) via SFT across math, code, science, and multi-turn chat, followed by multiple RL stages; Reward-aware Preference Optimization (RPO) for alignment, RL with Verifiable Rewards (RLVR) for step-wise reasoning, and iterative DPO to refine tool-use behavior. A distillation-driven Neural Architecture Search (“Puzzle”) replaces some attention blocks and varies FFN widths to shrink memory footprint and improve throughput, enabling single-GPU (H100/H200) deployment while preserving instruction following and CoT quality.

In internal evaluations (NeMo-Skills, up to 16 runs, temp = 0.6, top_p = 0.95), the model reports strong reasoning/coding results, e.g., MATH500 pass@1 = 97.4, AIME-2024 = 87.5, AIME-2025 = 82.71, GPQA = 71.97, LiveCodeBench (24.10–25.02) = 73.58, and MMLU-Pro (CoT) = 79.53. The model targets practical inference efficiency (high tokens/s, reduced VRAM) with Transformers/vLLM support and explicit “reasoning on/off” modes (chat-first defaults, greedy recommended when disabled). Suitable for building agents, assistants, and long-context retrieval systems where balanced accuracy-to-cost and reliable tool use matter.
,
    created=1760101395'''
)

NEMOTRON_3_NANO_30B_A3B = ModelInfo(
    str_identifier="nvidia/nemotron-3-nano-30b-a3b",
    price_in=6e-08,
    price_out=2.4e-07,
    creator="nvidia",
    description='''NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest compute efficiency and accuracy for developers to build specialized agentic AI systems.

The model is fully open with open-weights, datasets and recipes so developers can easily
customize, optimize, and deploy the model on their infrastructure for maximum privacy and
security.

Note: For the free endpoint, all prompts and output are logged to improve the provider's model and its product and services. Please do not upload any personal, confidential, or otherwise sensitive information. This is a trial use only. Do not use for production or business-critical systems.,
    created=1765731275'''
)

NEMOTRON_3_NANO_30B_A3B_FREE = ModelInfo(
    str_identifier="nvidia/nemotron-3-nano-30b-a3b:free",
    price_in=0.0,
    price_out=0.0,
    creator="nvidia",
    description='''NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest compute efficiency and accuracy for developers to build specialized agentic AI systems.

The model is fully open with open-weights, datasets and recipes so developers can easily
customize, optimize, and deploy the model on their infrastructure for maximum privacy and
security.

Note: For the free endpoint, all prompts and output are logged to improve the provider's model and its product and services. Please do not upload any personal, confidential, or otherwise sensitive information. This is a trial use only. Do not use for production or business-critical systems.,
    created=1765731275'''
)

NEMOTRON_NANO_12B_V2_VL = ModelInfo(
    str_identifier="nvidia/nemotron-nano-12b-v2-vl",
    price_in=2e-07,
    price_out=6e-07,
    creator="nvidia",
    description='''NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning model designed for video understanding and document intelligence. It introduces a hybrid Transformer-Mamba architecture, combining transformer-level accuracy with Mamba’s memory-efficient sequence modeling for significantly higher throughput and lower latency.

The model supports inputs of text and multi-image documents, producing natural-language outputs. It is trained on high-quality NVIDIA-curated synthetic datasets optimized for optical-character recognition, chart reasoning, and multimodal comprehension.

Nemotron Nano 2 VL achieves leading results on OCRBench v2 and scores ≈ 74 average across MMMU, MathVista, AI2D, OCRBench, OCR-Reasoning, ChartQA, DocVQA, and Video-MME—surpassing prior open VL baselines. With Efficient Video Sampling (EVS), it handles long-form videos while reducing inference cost.

Open-weights, training data, and fine-tuning recipes are released under a permissive NVIDIA open license, with deployment supported across NeMo, NIM, and major inference runtimes.,
    created=1761675565'''
)

NEMOTRON_NANO_12B_V2_VL_FREE = ModelInfo(
    str_identifier="nvidia/nemotron-nano-12b-v2-vl:free",
    price_in=0.0,
    price_out=0.0,
    creator="nvidia",
    description='''NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning model designed for video understanding and document intelligence. It introduces a hybrid Transformer-Mamba architecture, combining transformer-level accuracy with Mamba’s memory-efficient sequence modeling for significantly higher throughput and lower latency.

The model supports inputs of text and multi-image documents, producing natural-language outputs. It is trained on high-quality NVIDIA-curated synthetic datasets optimized for optical-character recognition, chart reasoning, and multimodal comprehension.

Nemotron Nano 2 VL achieves leading results on OCRBench v2 and scores ≈ 74 average across MMMU, MathVista, AI2D, OCRBench, OCR-Reasoning, ChartQA, DocVQA, and Video-MME—surpassing prior open VL baselines. With Efficient Video Sampling (EVS), it handles long-form videos while reducing inference cost.

Open-weights, training data, and fine-tuning recipes are released under a permissive NVIDIA open license, with deployment supported across NeMo, NIM, and major inference runtimes.,
    created=1761675565'''
)

NEMOTRON_NANO_9B_V2 = ModelInfo(
    str_identifier="nvidia/nemotron-nano-9b-v2",
    price_in=4e-08,
    price_out=1.6e-07,
    creator="nvidia",
    description='''NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response. 

The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so.,
    created=1757106807'''
)

NEMOTRON_NANO_9B_V2_FREE = ModelInfo(
    str_identifier="nvidia/nemotron-nano-9b-v2:free",
    price_in=0.0,
    price_out=0.0,
    creator="nvidia",
    description='''NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response. 

The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so.,
    created=1757106807'''
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

GPT_3_5_TURBO = ModelInfo(
    str_identifier="openai/gpt-3.5-turbo",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="openai",
    description='''GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.

Training data up to Sep 2021.,
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

GPT_4O_AUDIO_PREVIEW = ModelInfo(
    str_identifier="openai/gpt-4o-audio-preview",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="openai",
    description='''The gpt-4o-audio-preview model adds support for audio inputs as prompts. This enhancement allows the model to detect nuances within audio recordings and add depth to generated user experiences. Audio outputs are currently not supported. Audio tokens are priced at $40 per million input and $80 per million output audio tokens.,
    created=1755233061'''
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

GPT_5 = ModelInfo(
    str_identifier="openai/gpt-5",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5 is OpenAI’s most advanced model, offering major improvements in reasoning, code quality, and user experience. It is optimized for complex tasks that require step-by-step reasoning, instruction following, and accuracy in high-stakes use cases. It supports test-time routing features and advanced prompt understanding, including user-specified intent like "think hard about this." Improvements include reductions in hallucination, sycophancy, and better performance in coding, writing, and health-related tasks.,
    created=1754587413'''
)

GPT_5_CHAT = ModelInfo(
    str_identifier="openai/gpt-5-chat",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5 Chat is designed for advanced, natural, multimodal, and context-aware conversations for enterprise applications.,
    created=1754587837'''
)

GPT_5_CODEX = ModelInfo(
    str_identifier="openai/gpt-5-codex",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5-Codex is a specialized version of GPT-5 optimized for software engineering and coding workflows. It is designed for both interactive development sessions and long, independent execution of complex engineering tasks. The model supports building projects from scratch, feature development, debugging, large-scale refactoring, and code review. Compared to GPT-5, Codex is more steerable, adheres closely to developer instructions, and produces cleaner, higher-quality code outputs. Reasoning effort can be adjusted with the `reasoning.effort` parameter. Read the [docs here](https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level)

Codex integrates into developer environments including the CLI, IDE extensions, GitHub, and cloud tasks. It adapts reasoning effort dynamically—providing fast responses for small tasks while sustaining extended multi-hour runs for large projects. The model is trained to perform structured code reviews, catching critical flaws by reasoning over dependencies and validating behavior against tests. It also supports multimodal inputs such as images or screenshots for UI development and integrates tool use for search, dependency installation, and environment setup. Codex is intended specifically for agentic coding applications.,
    created=1758643403'''
)

GPT_5_IMAGE = ModelInfo(
    str_identifier="openai/gpt-5-image",
    price_in=1e-05,
    price_out=1e-05,
    creator="openai",
    description='''[GPT-5](https://openrouter.ai/openai/gpt-5) Image combines OpenAI's GPT-5 model with state-of-the-art image generation capabilities. It offers major improvements in reasoning, code quality, and user experience while incorporating GPT Image 1's superior instruction following, text rendering, and detailed image editing.,
    created=1760447986'''
)

GPT_5_IMAGE_MINI = ModelInfo(
    str_identifier="openai/gpt-5-image-mini",
    price_in=2.5e-06,
    price_out=2e-06,
    creator="openai",
    description='''GPT-5 Image Mini combines OpenAI's advanced language capabilities, powered by [GPT-5 Mini](https://openrouter.ai/openai/gpt-5-mini), with GPT Image 1 Mini for efficient image generation. This natively multimodal model features superior instruction following, text rendering, and detailed image editing with reduced latency and cost. It excels at high-quality visual creation while maintaining strong text understanding, making it ideal for applications that require both efficient image generation and text processing at scale.,
    created=1760624583'''
)

GPT_5_MINI = ModelInfo(
    str_identifier="openai/gpt-5-mini",
    price_in=2.5e-07,
    price_out=2e-06,
    creator="openai",
    description='''GPT-5 Mini is a compact version of GPT-5, designed to handle lighter-weight reasoning tasks. It provides the same instruction-following and safety-tuning benefits as GPT-5, but with reduced latency and cost. GPT-5 Mini is the successor to OpenAI's o4-mini model.,
    created=1754587407'''
)

GPT_5_NANO = ModelInfo(
    str_identifier="openai/gpt-5-nano",
    price_in=5e-08,
    price_out=4e-07,
    creator="openai",
    description='''GPT-5-Nano is the smallest and fastest variant in the GPT-5 system, optimized for developer tools, rapid interactions, and ultra-low latency environments. While limited in reasoning depth compared to its larger counterparts, it retains key instruction-following and safety features. It is the successor to GPT-4.1-nano and offers a lightweight option for cost-sensitive or real-time applications.,
    created=1754587402'''
)

GPT_5_PRO = ModelInfo(
    str_identifier="openai/gpt-5-pro",
    price_in=1.5e-05,
    price_out=0.00012,
    creator="openai",
    description='''GPT-5 Pro is OpenAI’s most advanced model, offering major improvements in reasoning, code quality, and user experience. It is optimized for complex tasks that require step-by-step reasoning, instruction following, and accuracy in high-stakes use cases. It supports test-time routing features and advanced prompt understanding, including user-specified intent like "think hard about this." Improvements include reductions in hallucination, sycophancy, and better performance in coding, writing, and health-related tasks.,
    created=1759776663'''
)

GPT_5_1 = ModelInfo(
    str_identifier="openai/gpt-5.1",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5.1 is the latest frontier-grade model in the GPT-5 series, offering stronger general-purpose reasoning, improved instruction adherence, and a more natural conversational style compared to GPT-5. It uses adaptive reasoning to allocate computation dynamically, responding quickly to simple queries while spending more depth on complex tasks. The model produces clearer, more grounded explanations with reduced jargon, making it easier to follow even on technical or multi-step problems.

Built for broad task coverage, GPT-5.1 delivers consistent gains across math, coding, and structured analysis workloads, with more coherent long-form answers and improved tool-use reliability. It also features refined conversational alignment, enabling warmer, more intuitive responses without compromising precision. GPT-5.1 serves as the primary full-capability successor to GPT-5,
    created=1763060305'''
)

GPT_5_1_CHAT = ModelInfo(
    str_identifier="openai/gpt-5.1-chat",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5.1 Chat (AKA Instant is the fast, lightweight member of the 5.1 family, optimized for low-latency chat while retaining strong general intelligence. It uses adaptive reasoning to selectively “think” on harder queries, improving accuracy on math, coding, and multi-step tasks without slowing down typical conversations. The model is warmer and more conversational by default, with better instruction following and more stable short-form reasoning. GPT-5.1 Chat is designed for high-throughput, interactive workloads where responsiveness and consistency matter more than deep deliberation.
,
    created=1763060302'''
)

GPT_5_1_CODEX = ModelInfo(
    str_identifier="openai/gpt-5.1-codex",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5.1-Codex is a specialized version of GPT-5.1 optimized for software engineering and coding workflows. It is designed for both interactive development sessions and long, independent execution of complex engineering tasks. The model supports building projects from scratch, feature development, debugging, large-scale refactoring, and code review. Compared to GPT-5.1, Codex is more steerable, adheres closely to developer instructions, and produces cleaner, higher-quality code outputs. Reasoning effort can be adjusted with the `reasoning.effort` parameter. Read the [docs here](https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level)

Codex integrates into developer environments including the CLI, IDE extensions, GitHub, and cloud tasks. It adapts reasoning effort dynamically—providing fast responses for small tasks while sustaining extended multi-hour runs for large projects. The model is trained to perform structured code reviews, catching critical flaws by reasoning over dependencies and validating behavior against tests. It also supports multimodal inputs such as images or screenshots for UI development and integrates tool use for search, dependency installation, and environment setup. Codex is intended specifically for agentic coding applications.,
    created=1763060298'''
)

GPT_5_1_CODEX_MAX = ModelInfo(
    str_identifier="openai/gpt-5.1-codex-max",
    price_in=1.25e-06,
    price_out=1e-05,
    creator="openai",
    description='''GPT-5.1-Codex-Max is OpenAI’s latest agentic coding model, designed for long-running, high-context software development tasks. It is based on an updated version of the 5.1 reasoning stack and trained on agentic workflows spanning software engineering, mathematics, and research. 
GPT-5.1-Codex-Max delivers faster performance, improved reasoning, and higher token efficiency across the development lifecycle. ,
    created=1764878934'''
)

GPT_5_1_CODEX_MINI = ModelInfo(
    str_identifier="openai/gpt-5.1-codex-mini",
    price_in=2.5e-07,
    price_out=2e-06,
    creator="openai",
    description='''GPT-5.1-Codex-Mini is a smaller and faster version of GPT-5.1-Codex,
    created=1763057820'''
)

GPT_5_2 = ModelInfo(
    str_identifier="openai/gpt-5.2",
    price_in=1.75e-06,
    price_out=1.4e-05,
    creator="openai",
    description='''GPT-5.2 is the latest frontier-grade model in the GPT-5 series, offering stronger agentic and long context perfomance compared to GPT-5.1. It uses adaptive reasoning to allocate computation dynamically, responding quickly to simple queries while spending more depth on complex tasks.

Built for broad task coverage, GPT-5.2 delivers consistent gains across math, coding, sciende, and tool calling workloads, with more coherent long-form answers and improved tool-use reliability.,
    created=1765389775'''
)

GPT_5_2_CHAT = ModelInfo(
    str_identifier="openai/gpt-5.2-chat",
    price_in=1.75e-06,
    price_out=1.4e-05,
    creator="openai",
    description='''GPT-5.2 Chat (AKA Instant) is the fast, lightweight member of the 5.2 family, optimized for low-latency chat while retaining strong general intelligence. It uses adaptive reasoning to selectively “think” on harder queries, improving accuracy on math, coding, and multi-step tasks without slowing down typical conversations. The model is warmer and more conversational by default, with better instruction following and more stable short-form reasoning. GPT-5.2 Chat is designed for high-throughput, interactive workloads where responsiveness and consistency matter more than deep deliberation.,
    created=1765389783'''
)

GPT_5_2_CODEX = ModelInfo(
    str_identifier="openai/gpt-5.2-codex",
    price_in=1.75e-06,
    price_out=1.4e-05,
    creator="openai",
    description='''GPT-5.2-Codex is an upgraded version of GPT-5.1-Codex optimized for software engineering and coding workflows. It is designed for both interactive development sessions and long, independent execution of complex engineering tasks. The model supports building projects from scratch, feature development, debugging, large-scale refactoring, and code review. Compared to GPT-5.1-Codex, 5.2-Codex is more steerable, adheres closely to developer instructions, and produces cleaner, higher-quality code outputs. Reasoning effort can be adjusted with the `reasoning.effort` parameter. Read the [docs here](https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level)

Codex integrates into developer environments including the CLI, IDE extensions, GitHub, and cloud tasks. It adapts reasoning effort dynamically—providing fast responses for small tasks while sustaining extended multi-hour runs for large projects. The model is trained to perform structured code reviews, catching critical flaws by reasoning over dependencies and validating behavior against tests. It also supports multimodal inputs such as images or screenshots for UI development and integrates tool use for search, dependency installation, and environment setup. Codex is intended specifically for agentic coding applications.,
    created=1768409315'''
)

GPT_5_2_PRO = ModelInfo(
    str_identifier="openai/gpt-5.2-pro",
    price_in=2.1e-05,
    price_out=0.000168,
    creator="openai",
    description='''GPT-5.2 Pro is OpenAI’s most advanced model, offering major improvements in agentic coding and long context performance over GPT-5 Pro. It is optimized for complex tasks that require step-by-step reasoning, instruction following, and accuracy in high-stakes use cases. It supports test-time routing features and advanced prompt understanding, including user-specified intent like "think hard about this." Improvements include reductions in hallucination, sycophancy, and better performance in coding, writing, and health-related tasks.,
    created=1765389780'''
)

GPT_AUDIO = ModelInfo(
    str_identifier="openai/gpt-audio",
    price_in=2.5e-06,
    price_out=1e-05,
    creator="openai",
    description='''The gpt-audio model is OpenAI's first generally available audio model. The new snapshot features an upgraded decoder for more natural sounding voices and maintains better voice consistency. Audio is priced at $32 per million input tokens and $64 per million output tokens.,
    created=1768862569'''
)

GPT_AUDIO_MINI = ModelInfo(
    str_identifier="openai/gpt-audio-mini",
    price_in=6e-07,
    price_out=2.4e-06,
    creator="openai",
    description='''A cost-efficient version of GPT Audio. The new snapshot features an upgraded decoder for more natural sounding voices and maintains better voice consistency. Input is priced at $0.60 per million tokens and output is priced at $2.40 per million tokens.,
    created=1768859419'''
)

GPT_OSS_120B = ModelInfo(
    str_identifier="openai/gpt-oss-120b",
    price_in=3.9e-08,
    price_out=1.9e-07,
    creator="openai",
    description='''gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI designed for high-reasoning, agentic, and general-purpose production use cases. It activates 5.1B parameters per forward pass and is optimized to run on a single H100 GPU with native MXFP4 quantization. The model supports configurable reasoning depth, full chain-of-thought access, and native tool use, including function calling, browsing, and structured output generation.,
    created=1754414231'''
)

GPT_OSS_120B_EXACTO = ModelInfo(
    str_identifier="openai/gpt-oss-120b:exacto",
    price_in=3.9e-08,
    price_out=1.9e-07,
    creator="openai",
    description='''gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI designed for high-reasoning, agentic, and general-purpose production use cases. It activates 5.1B parameters per forward pass and is optimized to run on a single H100 GPU with native MXFP4 quantization. The model supports configurable reasoning depth, full chain-of-thought access, and native tool use, including function calling, browsing, and structured output generation.,
    created=1754414231'''
)

GPT_OSS_120B_FREE = ModelInfo(
    str_identifier="openai/gpt-oss-120b:free",
    price_in=0.0,
    price_out=0.0,
    creator="openai",
    description='''gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI designed for high-reasoning, agentic, and general-purpose production use cases. It activates 5.1B parameters per forward pass and is optimized to run on a single H100 GPU with native MXFP4 quantization. The model supports configurable reasoning depth, full chain-of-thought access, and native tool use, including function calling, browsing, and structured output generation.,
    created=1754414231'''
)

GPT_OSS_20B = ModelInfo(
    str_identifier="openai/gpt-oss-20b",
    price_in=2e-08,
    price_out=1e-07,
    creator="openai",
    description='''gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. It uses a Mixture-of-Experts (MoE) architecture with 3.6B active parameters per forward pass, optimized for lower-latency inference and deployability on consumer or single-GPU hardware. The model is trained in OpenAI’s Harmony response format and supports reasoning level configuration, fine-tuning, and agentic capabilities including function calling, tool use, and structured outputs.,
    created=1754414229'''
)

GPT_OSS_20B_FREE = ModelInfo(
    str_identifier="openai/gpt-oss-20b:free",
    price_in=0.0,
    price_out=0.0,
    creator="openai",
    description='''gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. It uses a Mixture-of-Experts (MoE) architecture with 3.6B active parameters per forward pass, optimized for lower-latency inference and deployability on consumer or single-GPU hardware. The model is trained in OpenAI’s Harmony response format and supports reasoning level configuration, fine-tuning, and agentic capabilities including function calling, tool use, and structured outputs.,
    created=1754414229'''
)

GPT_OSS_SAFEGUARD_20B = ModelInfo(
    str_identifier="openai/gpt-oss-safeguard-20b",
    price_in=7.5e-08,
    price_out=3e-07,
    creator="openai",
    description='''gpt-oss-safeguard-20b is a safety reasoning model from OpenAI built upon gpt-oss-20b. This open-weight, 21B-parameter Mixture-of-Experts (MoE) model offers lower latency for safety tasks like content classification, LLM filtering, and trust & safety labeling.

Learn more about this model in OpenAI's gpt-oss-safeguard [user guide](https://cookbook.openai.com/articles/gpt-oss-safeguard-guide).,
    created=1761752836'''
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
    price_in=2e-06,
    price_out=8e-06,
    creator="openai",
    description='''o3 is a well-rounded and powerful model across domains. It sets a new standard for math, science, coding, and visual reasoning tasks. It also excels at technical writing and instruction-following. Use it to think through multi-step problems that involve analysis across text, code, and images. ,
    created=1744823457'''
)

O3_DEEP_RESEARCH = ModelInfo(
    str_identifier="openai/o3-deep-research",
    price_in=1e-05,
    price_out=4e-05,
    creator="openai",
    description='''o3-deep-research is OpenAI's advanced model for deep research, designed to tackle complex, multi-step research tasks.

Note: This model always uses the 'web_search' tool which adds additional cost.,
    created=1760129661'''
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

O3_PRO = ModelInfo(
    str_identifier="openai/o3-pro",
    price_in=2e-05,
    price_out=8e-05,
    creator="openai",
    description='''The o-series of models are trained with reinforcement learning to think before they answer and perform complex reasoning. The o3-pro model uses more compute to think harder and provide consistently better answers.

Note that BYOK is required for this model. Set up here: https://openrouter.ai/settings/integrations,
    created=1749598352'''
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

O4_MINI_DEEP_RESEARCH = ModelInfo(
    str_identifier="openai/o4-mini-deep-research",
    price_in=2e-06,
    price_out=8e-06,
    creator="openai",
    description='''o4-mini-deep-research is OpenAI's faster, more affordable deep research model—ideal for tackling complex, multi-step research tasks.

Note: This model always uses the 'web_search' tool which adds additional cost.,
    created=1760129642'''
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

INTERNVL3_78B = ModelInfo(
    str_identifier="opengvlab/internvl3-78b",
    price_in=1e-07,
    price_out=3.9e-07,
    creator="opengvlab",
    description='''The InternVL3 series is an advanced multimodal large language model (MLLM). Compared to InternVL 2.5, InternVL3 demonstrates stronger multimodal perception and reasoning capabilities. 

In addition, InternVL3 is benchmarked against the Qwen2.5 Chat models, whose pre-trained base models serve as the initialization for its language component. Benefiting from Native Multimodal Pre-Training, the InternVL3 series surpasses the Qwen2.5 series in overall text performance.,
    created=1757962555'''
)

AUTO = ModelInfo(
    str_identifier="openrouter/auto",
    price_in=-1.0,
    price_out=-1.0,
    creator="openrouter",
    description='''Your prompt will be processed by a meta-model and routed to one of dozens of models (see below), optimizing for the best possible output.

To see which model was used, visit [Activity](/activity), or read the `model` attribute of the response. Your response will be priced at the same rate as the routed model.

Learn more, including how to customize the models for routing, in our [docs](/docs/guides/routing/routers/auto-router).

Requests will be routed to the following models:
- [openai/gpt-5.1](/openai/gpt-5.1)
- [openai/gpt-5](/openai/gpt-5)
- [openai/gpt-5-mini](/openai/gpt-5-mini)
- [openai/gpt-5-nano](/openai/gpt-5-nano)
- [openai/gpt-4.1](/openai/gpt-4.1)
- [openai/gpt-4.1-mini](/openai/gpt-4.1-mini)
- [openai/gpt-4.1-nano](/openai/gpt-4.1-nano)
- [openai/gpt-4o](/openai/gpt-4o)
- [openai/gpt-4o-2024-05-13](/openai/gpt-4o-2024-05-13)
- [openai/gpt-4o-2024-08-06](/openai/gpt-4o-2024-08-06)
- [openai/gpt-4o-2024-11-20](/openai/gpt-4o-2024-11-20)
- [openai/gpt-4o-mini](/openai/gpt-4o-mini)
- [openai/gpt-4o-mini-2024-07-18](/openai/gpt-4o-mini-2024-07-18)
- [openai/gpt-4-turbo](/openai/gpt-4-turbo)
- [openai/gpt-4-turbo-preview](/openai/gpt-4-turbo-preview)
- [openai/gpt-4-1106-preview](/openai/gpt-4-1106-preview)
- [openai/gpt-4](/openai/gpt-4)
- [openai/gpt-3.5-turbo](/openai/gpt-3.5-turbo)
- [openai/gpt-oss-120b](/openai/gpt-oss-120b)
- [anthropic/claude-opus-4.5](/anthropic/claude-opus-4.5)
- [anthropic/claude-opus-4.1](/anthropic/claude-opus-4.1)
- [anthropic/claude-opus-4](/anthropic/claude-opus-4)
- [anthropic/claude-sonnet-4.5](/anthropic/claude-sonnet-4.5)
- [anthropic/claude-sonnet-4](/anthropic/claude-sonnet-4)
- [anthropic/claude-3.7-sonnet](/anthropic/claude-3.7-sonnet)
- [anthropic/claude-haiku-4.5](/anthropic/claude-haiku-4.5)
- [anthropic/claude-3.5-haiku](/anthropic/claude-3.5-haiku)
- [anthropic/claude-3-haiku](/anthropic/claude-3-haiku)
- [google/gemini-3-pro-preview](/google/gemini-3-pro-preview)
- [google/gemini-2.5-pro](/google/gemini-2.5-pro)
- [google/gemini-2.0-flash-001](/google/gemini-2.0-flash-001)
- [google/gemini-2.5-flash](/google/gemini-2.5-flash)
- [mistralai/mistral-large](/mistralai/mistral-large)
- [mistralai/mistral-large-2407](/mistralai/mistral-large-2407)
- [mistralai/mistral-large-2411](/mistralai/mistral-large-2411)
- [mistralai/mistral-medium-3.1](/mistralai/mistral-medium-3.1)
- [mistralai/mistral-nemo](/mistralai/mistral-nemo)
- [mistralai/mistral-7b-instruct](/mistralai/mistral-7b-instruct)
- [mistralai/mixtral-8x7b-instruct](/mistralai/mixtral-8x7b-instruct)
- [mistralai/mixtral-8x22b-instruct](/mistralai/mixtral-8x22b-instruct)
- [mistralai/codestral-2508](/mistralai/codestral-2508)
- [x-ai/grok-4](/x-ai/grok-4)
- [x-ai/grok-3](/x-ai/grok-3)
- [x-ai/grok-3-mini](/x-ai/grok-3-mini)
- [deepseek/deepseek-r1](/deepseek/deepseek-r1)
- [meta-llama/llama-3.3-70b-instruct](/meta-llama/llama-3.3-70b-instruct)
- [meta-llama/llama-3.1-405b-instruct](/meta-llama/llama-3.1-405b-instruct)
- [meta-llama/llama-3.1-70b-instruct](/meta-llama/llama-3.1-70b-instruct)
- [meta-llama/llama-3.1-8b-instruct](/meta-llama/llama-3.1-8b-instruct)
- [meta-llama/llama-3-70b-instruct](/meta-llama/llama-3-70b-instruct)
- [meta-llama/llama-3-8b-instruct](/meta-llama/llama-3-8b-instruct)
- [qwen/qwen3-235b-a22b](/qwen/qwen3-235b-a22b)
- [qwen/qwen3-32b](/qwen/qwen3-32b)
- [qwen/qwen3-14b](/qwen/qwen3-14b)
- [cohere/command-r-plus-08-2024](/cohere/command-r-plus-08-2024)
- [cohere/command-r-08-2024](/cohere/command-r-08-2024)
- [moonshotai/kimi-k2-thinking](/moonshotai/kimi-k2-thinking)
- [perplexity/sonar](/perplexity/sonar),
    created=1699401600'''
)

BODYBUILDER = ModelInfo(
    str_identifier="openrouter/bodybuilder",
    price_in=-1.0,
    price_out=-1.0,
    creator="openrouter",
    description='''Transform your natural language requests into structured OpenRouter API request objects. Describe what you want to accomplish with AI models, and Body Builder will construct the appropriate API calls. Example: "count to 10 using gemini and opus."

This is useful for creating multi-model requests, custom model routers, or programmatic generation of API calls from human descriptions.

**BETA NOTICE**: Body Builder is in beta, and currently free. Pricing and functionality may change in the future.,
    created=1764903653'''
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

SONAR_PRO_SEARCH = ModelInfo(
    str_identifier="perplexity/sonar-pro-search",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="perplexity",
    description='''Exclusively available on the OpenRouter API, Sonar Pro's new Pro Search mode is Perplexity's most advanced agentic search system. It is designed for deeper reasoning and analysis. Pricing is based on tokens plus $18 per thousand requests. This model powers the Pro Search mode on the Perplexity platform.

Sonar Pro Search adds autonomous, multi-step reasoning to Sonar Pro. So, instead of just one query + synthesis, it plans and executes entire research workflows using tools.,
    created=1761854366'''
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

INTELLECT_3 = ModelInfo(
    str_identifier="prime-intellect/intellect-3",
    price_in=2e-07,
    price_out=1.1e-06,
    creator="prime-intellect",
    description='''INTELLECT-3 is a 106B-parameter Mixture-of-Experts model (12B active) post-trained from GLM-4.5-Air-Base using supervised fine-tuning (SFT) followed by large-scale reinforcement learning (RL). It offers state-of-the-art performance for its size across math, code, science, and general reasoning, consistently outperforming many larger frontier models. Designed for strong multi-step problem solving, it maintains high accuracy on structured tasks while remaining efficient at inference thanks to its MoE architecture.,
    created=1764212534'''
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

QWEN_2_5_CODER_32B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen-2.5-coder-32b-instruct",
    price_in=3e-08,
    price_out=1.1e-07,
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

QWEN_PLUS_2025_07_28 = ModelInfo(
    str_identifier="qwen/qwen-plus-2025-07-28",
    price_in=4e-07,
    price_out=1.2e-06,
    creator="qwen",
    description='''Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybrid reasoning model with a balanced performance, speed, and cost combination.,
    created=1757347599'''
)

QWEN_PLUS_2025_07_28_THINKING = ModelInfo(
    str_identifier="qwen/qwen-plus-2025-07-28:thinking",
    price_in=4e-07,
    price_out=4e-06,
    creator="qwen",
    description='''Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybrid reasoning model with a balanced performance, speed, and cost combination.,
    created=1757347599'''
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

QWEN2_5_CODER_7B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen2.5-coder-7b-instruct",
    price_in=3e-08,
    price_out=9e-08,
    creator="qwen",
    description='''Qwen2.5-Coder-7B-Instruct is a 7B parameter instruction-tuned language model optimized for code-related tasks such as code generation, reasoning, and bug fixing. Based on the Qwen2.5 architecture, it incorporates enhancements like RoPE, SwiGLU, RMSNorm, and GQA attention with support for up to 128K tokens using YaRN-based extrapolation. It is trained on a large corpus of source code, synthetic data, and text-code grounding, providing robust performance across programming languages and agentic coding workflows.

This model is part of the Qwen2.5-Coder family and offers strong compatibility with tools like vLLM for efficient deployment. Released under the Apache 2.0 license.,
    created=1744734887'''
)

QWEN2_5_VL_32B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-32b-instruct",
    price_in=5e-08,
    price_out=2.2e-07,
    creator="qwen",
    description='''Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solving capabilities. It excels at visual analysis tasks, including object recognition, textual interpretation within images, and precise event localization in extended videos. Qwen2.5-VL-32B demonstrates state-of-the-art performance across multimodal benchmarks such as MMMU, MathVista, and VideoMME, while maintaining strong reasoning and clarity in text-based tasks like MMLU, mathematical problem-solving, and code generation.,
    created=1742839838'''
)

QWEN2_5_VL_72B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen2.5-vl-72b-instruct",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="qwen",
    description='''Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within images.,
    created=1738410311'''
)

QWEN3_14B = ModelInfo(
    str_identifier="qwen/qwen3-14b",
    price_in=5e-08,
    price_out=2.2e-07,
    creator="qwen",
    description='''Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series, designed for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, programming, and logical inference, and a "non-thinking" mode for general-purpose conversation. The model is fine-tuned for instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling.,
    created=1745876478'''
)

QWEN3_235B_A22B = ModelInfo(
    str_identifier="qwen/qwen3-235b-a22b",
    price_in=2e-07,
    price_out=6e-07,
    creator="qwen",
    description='''Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by Qwen, activating 22B parameters per forward pass. It supports seamless switching between a "thinking" mode for complex reasoning, math, and code tasks, and a "non-thinking" mode for general conversational efficiency. The model demonstrates strong reasoning ability, multilingual support (100+ languages and dialects), advanced instruction-following, and agent tool-calling capabilities. It natively handles a 32K token context window and extends up to 131K tokens using YaRN-based scaling.,
    created=1745875757'''
)

QWEN3_235B_A22B_2507 = ModelInfo(
    str_identifier="qwen/qwen3-235b-a22b-2507",
    price_in=7.1e-08,
    price_out=4.63e-07,
    creator="qwen",
    description='''Qwen3-235B-A22B-Instruct-2507 is a multilingual, instruction-tuned mixture-of-experts language model based on the Qwen3-235B architecture, with 22B active parameters per forward pass. It is optimized for general-purpose text generation, including instruction following, logical reasoning, math, code, and tool usage. The model supports a native 262K context length and does not implement "thinking mode" (<think> blocks).

Compared to its base variant, this version delivers significant gains in knowledge coverage, long-context reasoning, coding benchmarks, and alignment with open-ended tasks. It is particularly strong on multilingual understanding, math reasoning (e.g., AIME, HMMT), and alignment evaluations like Arena-Hard and WritingBench.,
    created=1753119555'''
)

QWEN3_235B_A22B_THINKING_2507 = ModelInfo(
    str_identifier="qwen/qwen3-235b-a22b-thinking-2507",
    price_in=1.1e-07,
    price_out=6e-07,
    creator="qwen",
    description='''Qwen3-235B-A22B-Thinking-2507 is a high-performance, open-weight Mixture-of-Experts (MoE) language model optimized for complex reasoning tasks. It activates 22B of its 235B parameters per forward pass and natively supports up to 262,144 tokens of context. This "thinking-only" variant enhances structured logical reasoning, mathematics, science, and long-form generation, showing strong benchmark performance across AIME, SuperGPQA, LiveCodeBench, and MMLU-Redux. It enforces a special reasoning mode (</think>) and is designed for high-token outputs (up to 81,920 tokens) in challenging domains.

The model is instruction-tuned and excels at step-by-step reasoning, tool use, agentic workflows, and multilingual tasks. This release represents the most capable open-source variant in the Qwen3-235B series, surpassing many closed models in structured reasoning use cases.,
    created=1753449557'''
)

QWEN3_30B_A3B = ModelInfo(
    str_identifier="qwen/qwen3-30b-a3b",
    price_in=6e-08,
    price_out=2.2e-07,
    creator="qwen",
    description='''Qwen3, the latest generation in the Qwen large language model series, features both dense and mixture-of-experts (MoE) architectures to excel in reasoning, multilingual support, and advanced agent tasks. Its unique ability to switch seamlessly between a thinking mode for complex reasoning and a non-thinking mode for efficient dialogue ensures versatile, high-quality performance.

Significantly outperforming prior models like QwQ and Qwen2.5, Qwen3 delivers superior mathematics, coding, commonsense reasoning, creative writing, and interactive dialogue capabilities. The Qwen3-30B-A3B variant includes 30.5 billion parameters (3.3 billion activated), 48 layers, 128 experts (8 activated per task), and supports up to 131K token contexts with YaRN, setting a new standard among open-source models.,
    created=1745878604'''
)

QWEN3_30B_A3B_INSTRUCT_2507 = ModelInfo(
    str_identifier="qwen/qwen3-30b-a3b-instruct-2507",
    price_in=8e-08,
    price_out=3.3e-07,
    creator="qwen",
    description='''Qwen3-30B-A3B-Instruct-2507 is a 30.5B-parameter mixture-of-experts language model from Qwen, with 3.3B active parameters per inference. It operates in non-thinking mode and is designed for high-quality instruction following, multilingual understanding, and agentic tool use. Post-trained on instruction data, it demonstrates competitive performance across reasoning (AIME, ZebraLogic), coding (MultiPL-E, LiveCodeBench), and alignment (IFEval, WritingBench) benchmarks. It outperforms its non-instruct variant on subjective and open-ended tasks while retaining strong factual and coding performance.,
    created=1753806965'''
)

QWEN3_30B_A3B_THINKING_2507 = ModelInfo(
    str_identifier="qwen/qwen3-30b-a3b-thinking-2507",
    price_in=5.1e-08,
    price_out=3.4e-07,
    creator="qwen",
    description='''Qwen3-30B-A3B-Thinking-2507 is a 30B parameter Mixture-of-Experts reasoning model optimized for complex tasks requiring extended multi-step thinking. The model is designed specifically for “thinking mode,” where internal reasoning traces are separated from final answers.

Compared to earlier Qwen3-30B releases, this version improves performance across logical reasoning, mathematics, science, coding, and multilingual benchmarks. It also demonstrates stronger instruction following, tool use, and alignment with human preferences. With higher reasoning efficiency and extended output budgets, it is best suited for advanced research, competitive problem solving, and agentic applications requiring structured long-context reasoning.,
    created=1756399192'''
)

QWEN3_32B = ModelInfo(
    str_identifier="qwen/qwen3-32b",
    price_in=8e-08,
    price_out=2.4e-07,
    creator="qwen",
    description='''Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series, optimized for both complex reasoning and efficient dialogue. It supports seamless switching between a "thinking" mode for tasks like math, coding, and logical inference, and a "non-thinking" mode for faster, general-purpose conversation. The model demonstrates strong performance in instruction-following, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling. ,
    created=1745875945'''
)

QWEN3_4B_FREE = ModelInfo(
    str_identifier="qwen/qwen3-4b:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-4B is a 4 billion parameter dense language model from the Qwen3 series, designed to support both general-purpose and reasoning-intensive tasks. It introduces a dual-mode architecture—thinking and non-thinking—allowing dynamic switching between high-precision logical reasoning and efficient dialogue generation. This makes it well-suited for multi-turn chat, instruction following, and complex agent workflows.,
    created=1746031104'''
)

QWEN3_8B = ModelInfo(
    str_identifier="qwen/qwen3-8b",
    price_in=5e-08,
    price_out=2.5e-07,
    creator="qwen",
    description='''Qwen3-8B is a dense 8.2B parameter causal language model from the Qwen3 series, designed for both reasoning-heavy tasks and efficient dialogue. It supports seamless switching between "thinking" mode for math, coding, and logical inference, and "non-thinking" mode for general conversation. The model is fine-tuned for instruction-following, agent integration, creative writing, and multilingual use across 100+ languages and dialects. It natively supports a 32K token context window and can extend to 131K tokens with YaRN scaling.,
    created=1745876632'''
)

QWEN3_CODER = ModelInfo(
    str_identifier="qwen/qwen3-coder",
    price_in=2.2e-07,
    price_out=9.5e-07,
    creator="qwen",
    description='''Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the Qwen team. It is optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning over repositories. The model features 480 billion total parameters, with 35 billion active per forward pass (8 out of 160 experts).

Pricing for the Alibaba endpoints varies by context length. Once a request is greater than 128k input tokens, the higher pricing is used.,
    created=1753230546'''
)

QWEN3_CODER_30B_A3B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen3-coder-30b-a3b-instruct",
    price_in=7e-08,
    price_out=2.7e-07,
    creator="qwen",
    description='''Qwen3-Coder-30B-A3B-Instruct is a 30.5B parameter Mixture-of-Experts (MoE) model with 128 experts (8 active per forward pass), designed for advanced code generation, repository-scale understanding, and agentic tool use. Built on the Qwen3 architecture, it supports a native context length of 256K tokens (extendable to 1M with Yarn) and performs strongly in tasks involving function calls, browser use, and structured code completion.

This model is optimized for instruction-following without “thinking mode”, and integrates well with OpenAI-compatible tool-use formats. ,
    created=1753972379'''
)

QWEN3_CODER_FLASH = ModelInfo(
    str_identifier="qwen/qwen3-coder-flash",
    price_in=3e-07,
    price_out=1.5e-06,
    creator="qwen",
    description='''Qwen3 Coder Flash is Alibaba's fast and cost efficient version of their proprietary Qwen3 Coder Plus. It is a powerful coding agent model specializing in autonomous programming via tool calling and environment interaction, combining coding proficiency with versatile general-purpose abilities.,
    created=1758115536'''
)

QWEN3_CODER_PLUS = ModelInfo(
    str_identifier="qwen/qwen3-coder-plus",
    price_in=1e-06,
    price_out=5e-06,
    creator="qwen",
    description='''Qwen3 Coder Plus is Alibaba's proprietary version of the Open Source Qwen3 Coder 480B A35B. It is a powerful coding agent model specializing in autonomous programming via tool calling and environment interaction, combining coding proficiency with versatile general-purpose abilities.,
    created=1758662707'''
)

QWEN3_CODER_EXACTO = ModelInfo(
    str_identifier="qwen/qwen3-coder:exacto",
    price_in=2.2e-07,
    price_out=1.8e-06,
    creator="qwen",
    description='''Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the Qwen team. It is optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning over repositories. The model features 480 billion total parameters, with 35 billion active per forward pass (8 out of 160 experts).

Pricing for the Alibaba endpoints varies by context length. Once a request is greater than 128k input tokens, the higher pricing is used.,
    created=1753230546'''
)

QWEN3_CODER_FREE = ModelInfo(
    str_identifier="qwen/qwen3-coder:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the Qwen team. It is optimized for agentic coding tasks such as function calling, tool use, and long-context reasoning over repositories. The model features 480 billion total parameters, with 35 billion active per forward pass (8 out of 160 experts).

Pricing for the Alibaba endpoints varies by context length. Once a request is greater than 128k input tokens, the higher pricing is used.,
    created=1753230546'''
)

QWEN3_MAX = ModelInfo(
    str_identifier="qwen/qwen3-max",
    price_in=1.2e-06,
    price_out=6e-06,
    creator="qwen",
    description='''Qwen3-Max is an updated release built on the Qwen3 series, offering major improvements in reasoning, instruction following, multilingual support, and long-tail knowledge coverage compared to the January 2025 version. It delivers higher accuracy in math, coding, logic, and science tasks, follows complex instructions in Chinese and English more reliably, reduces hallucinations, and produces higher-quality responses for open-ended Q&A, writing, and conversation. The model supports over 100 languages with stronger translation and commonsense reasoning, and is optimized for retrieval-augmented generation (RAG) and tool calling, though it does not include a dedicated “thinking” mode.,
    created=1758662808'''
)

QWEN3_NEXT_80B_A3B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen3-next-80b-a3b-instruct",
    price_in=9e-08,
    price_out=1.1e-06,
    creator="qwen",
    description='''Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next series optimized for fast, stable responses without “thinking” traces. It targets complex tasks across reasoning, code generation, knowledge QA, and multilingual use, while remaining robust on alignment and formatting. Compared with prior Qwen3 instruct variants, it focuses on higher throughput and stability on ultra-long inputs and multi-turn dialogues, making it well-suited for RAG, tool use, and agentic workflows that require consistent final answers rather than visible chain-of-thought.

The model employs scaling-efficient training and decoding to improve parameter efficiency and inference speed, and has been validated on a broad set of public benchmarks where it reaches or approaches larger Qwen3 systems in several categories while outperforming earlier mid-sized baselines. It is best used as a general assistant, code helper, and long-context task solver in production settings where deterministic, instruction-following outputs are preferred.,
    created=1757612213'''
)

QWEN3_NEXT_80B_A3B_INSTRUCT_FREE = ModelInfo(
    str_identifier="qwen/qwen3-next-80b-a3b-instruct:free",
    price_in=0.0,
    price_out=0.0,
    creator="qwen",
    description='''Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next series optimized for fast, stable responses without “thinking” traces. It targets complex tasks across reasoning, code generation, knowledge QA, and multilingual use, while remaining robust on alignment and formatting. Compared with prior Qwen3 instruct variants, it focuses on higher throughput and stability on ultra-long inputs and multi-turn dialogues, making it well-suited for RAG, tool use, and agentic workflows that require consistent final answers rather than visible chain-of-thought.

The model employs scaling-efficient training and decoding to improve parameter efficiency and inference speed, and has been validated on a broad set of public benchmarks where it reaches or approaches larger Qwen3 systems in several categories while outperforming earlier mid-sized baselines. It is best used as a general assistant, code helper, and long-context task solver in production settings where deterministic, instruction-following outputs are preferred.,
    created=1757612213'''
)

QWEN3_NEXT_80B_A3B_THINKING = ModelInfo(
    str_identifier="qwen/qwen3-next-80b-a3b-thinking",
    price_in=1.5e-07,
    price_out=1.2e-06,
    creator="qwen",
    description='''Qwen3-Next-80B-A3B-Thinking is a reasoning-first chat model in the Qwen3-Next line that outputs structured “thinking” traces by default. It’s designed for hard multi-step problems; math proofs, code synthesis/debugging, logic, and agentic planning, and reports strong results across knowledge, reasoning, coding, alignment, and multilingual evaluations. Compared with prior Qwen3 variants, it emphasizes stability under long chains of thought and efficient scaling during inference, and it is tuned to follow complex instructions while reducing repetitive or off-task behavior.

The model is suitable for agent frameworks and tool use (function calling), retrieval-heavy workflows, and standardized benchmarking where step-by-step solutions are required. It supports long, detailed completions and leverages throughput-oriented techniques (e.g., multi-token prediction) for faster generation. Note that it operates in thinking-only mode.,
    created=1757612284'''
)

QWEN3_VL_235B_A22B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen3-vl-235b-a22b-instruct",
    price_in=2e-07,
    price_out=1.2e-06,
    creator="qwen",
    description='''Qwen3-VL-235B-A22B Instruct is an open-weight multimodal model that unifies strong text generation with visual understanding across images and video. The Instruct model targets general vision-language use (VQA, document parsing, chart/table extraction, multilingual OCR). The series emphasizes robust perception (recognition of diverse real-world and synthetic categories), spatial understanding (2D/3D grounding), and long-form visual comprehension, with competitive results on public multimodal benchmarks for both perception and reasoning.

Beyond analysis, Qwen3-VL supports agentic interaction and tool use: it can follow complex instructions over multi-image, multi-turn dialogues; align text to video timelines for precise temporal queries; and operate GUI elements for automation tasks. The models also enable visual coding workflows—turning sketches or mockups into code and assisting with UI debugging—while maintaining strong text-only performance comparable to the flagship Qwen3 language models. This makes Qwen3-VL suitable for production scenarios spanning document AI, multilingual OCR, software/UI assistance, spatial/embodied tasks, and research on vision-language agents.,
    created=1758668687'''
)

QWEN3_VL_235B_A22B_THINKING = ModelInfo(
    str_identifier="qwen/qwen3-vl-235b-a22b-thinking",
    price_in=4.5e-07,
    price_out=3.5e-06,
    creator="qwen",
    description='''Qwen3-VL-235B-A22B Thinking is a multimodal model that unifies strong text generation with visual understanding across images and video. The Thinking model is optimized for multimodal reasoning in STEM and math. The series emphasizes robust perception (recognition of diverse real-world and synthetic categories), spatial understanding (2D/3D grounding), and long-form visual comprehension, with competitive results on public multimodal benchmarks for both perception and reasoning.

Beyond analysis, Qwen3-VL supports agentic interaction and tool use: it can follow complex instructions over multi-image, multi-turn dialogues; align text to video timelines for precise temporal queries; and operate GUI elements for automation tasks. The models also enable visual coding workflows, turning sketches or mockups into code and assisting with UI debugging, while maintaining strong text-only performance comparable to the flagship Qwen3 language models. This makes Qwen3-VL suitable for production scenarios spanning document AI, multilingual OCR, software/UI assistance, spatial/embodied tasks, and research on vision-language agents.,
    created=1758668690'''
)

QWEN3_VL_30B_A3B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen3-vl-30b-a3b-instruct",
    price_in=1.5e-07,
    price_out=6e-07,
    creator="qwen",
    description='''Qwen3-VL-30B-A3B-Instruct is a multimodal model that unifies strong text generation with visual understanding for images and videos. Its Instruct variant optimizes instruction-following for general multimodal tasks. It excels in perception of real-world/synthetic categories, 2D/3D spatial grounding, and long-form visual comprehension, achieving competitive multimodal benchmark results. For agentic use, it handles multi-image multi-turn instructions, video timeline alignments, GUI automation, and visual coding from sketches to debugged UI. Text performance matches flagship Qwen3 models, suiting document AI, OCR, UI assistance, spatial tasks, and agent research.,
    created=1759794476'''
)

QWEN3_VL_30B_A3B_THINKING = ModelInfo(
    str_identifier="qwen/qwen3-vl-30b-a3b-thinking",
    price_in=2e-07,
    price_out=1e-06,
    creator="qwen",
    description='''Qwen3-VL-30B-A3B-Thinking is a multimodal model that unifies strong text generation with visual understanding for images and videos. Its Thinking variant enhances reasoning in STEM, math, and complex tasks. It excels in perception of real-world/synthetic categories, 2D/3D spatial grounding, and long-form visual comprehension, achieving competitive multimodal benchmark results. For agentic use, it handles multi-image multi-turn instructions, video timeline alignments, GUI automation, and visual coding from sketches to debugged UI. Text performance matches flagship Qwen3 models, suiting document AI, OCR, UI assistance, spatial tasks, and agent research.,
    created=1759794479'''
)

QWEN3_VL_32B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen3-vl-32b-instruct",
    price_in=5e-07,
    price_out=1.5e-06,
    creator="qwen",
    description='''Qwen3-VL-32B-Instruct is a large-scale multimodal vision-language model designed for high-precision understanding and reasoning across text, images, and video. With 32 billion parameters, it combines deep visual perception with advanced text comprehension, enabling fine-grained spatial reasoning, document and scene analysis, and long-horizon video understanding.Robust OCR in 32 languages, and enhanced multimodal fusion through Interleaved-MRoPE and DeepStack architectures. Optimized for agentic interaction and visual tool use, Qwen3-VL-32B delivers state-of-the-art performance for complex real-world multimodal tasks.,
    created=1761231332'''
)

QWEN3_VL_8B_INSTRUCT = ModelInfo(
    str_identifier="qwen/qwen3-vl-8b-instruct",
    price_in=8e-08,
    price_out=5e-07,
    creator="qwen",
    description='''Qwen3-VL-8B-Instruct is a multimodal vision-language model from the Qwen3-VL series, built for high-fidelity understanding and reasoning across text, images, and video. It features improved multimodal fusion with Interleaved-MRoPE for long-horizon temporal reasoning, DeepStack for fine-grained visual-text alignment, and text-timestamp alignment for precise event localization.

The model supports a native 256K-token context window, extensible to 1M tokens, and handles both static and dynamic media inputs for tasks like document parsing, visual question answering, spatial reasoning, and GUI control. It achieves text understanding comparable to leading LLMs while expanding OCR coverage to 32 languages and enhancing robustness under varied visual conditions.,
    created=1760463308'''
)

QWEN3_VL_8B_THINKING = ModelInfo(
    str_identifier="qwen/qwen3-vl-8b-thinking",
    price_in=1.8e-07,
    price_out=2.1e-06,
    creator="qwen",
    description='''Qwen3-VL-8B-Thinking is the reasoning-optimized variant of the Qwen3-VL-8B multimodal model, designed for advanced visual and textual reasoning across complex scenes, documents, and temporal sequences. It integrates enhanced multimodal alignment and long-context processing (native 256K, expandable to 1M tokens) for tasks such as scientific visual analysis, causal inference, and mathematical reasoning over image or video inputs.

Compared to the Instruct edition, the Thinking version introduces deeper visual-language fusion and deliberate reasoning pathways that improve performance on long-chain logic tasks, STEM problem-solving, and multi-step video understanding. It achieves stronger temporal grounding via Interleaved-MRoPE and timestamp-aware embeddings, while maintaining robust OCR, multilingual comprehension, and text generation on par with large text-only LLMs.,
    created=1760463746'''
)

QWQ_32B = ModelInfo(
    str_identifier="qwen/qwq-32b",
    price_in=1.5e-07,
    price_out=4e-07,
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

RELACE_APPLY_3 = ModelInfo(
    str_identifier="relace/relace-apply-3",
    price_in=8.5e-07,
    price_out=1.25e-06,
    creator="relace",
    description='''Relace Apply 3 is a specialized code-patching LLM that merges AI-suggested edits straight into your source files. It can apply updates from GPT-4o, Claude, and others into your files at 10,000 tokens/sec on average.

The model requires the prompt to be in the following format: 
<instruction>{instruction}</instruction>
<code>{initial_code}</code>
<update>{edit_snippet}</update>

Zero Data Retention is enabled for Relace. Learn more about this model in their [documentation](https://docs.relace.ai/api-reference/instant-apply/apply),
    created=1758891572'''
)

RELACE_SEARCH = ModelInfo(
    str_identifier="relace/relace-search",
    price_in=1e-06,
    price_out=3e-06,
    creator="relace",
    description='''The relace-search model uses 4-12 `view_file` and `grep` tools in parallel to explore a codebase and return relevant files to the user request. 

In contrast to RAG, relace-search performs agentic multi-step reasoning to produce highly precise results 4x faster than any frontier model. It's designed to serve as a subagent that passes its findings to an "oracle" coding agent, who orchestrates/performs the rest of the coding task.

To use relace-search you need to build an appropriate agent harness, and parse the response for relevant information to hand off to the oracle. Read more about it in the [Relace documentation](https://docs.relace.ai/docs/fast-agentic-search/agent).,
    created=1765213560'''
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
    price_in=4e-08,
    price_out=5e-08,
    creator="sao10k",
    description='''Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It's a strategic merge of multiple models, designed to balance creativity with improved logic and general knowledge.

Created by [Sao10k](https://huggingface.co/Sao10k), this model aims to offer an improved experience over Stheno v3.2, with enhanced creativity and logical reasoning.

For best results, use with Llama 3 Instruct context template, temperature 1.4, and min_p 0.1.,
    created=1723507200'''
)

L3_1_70B_HANAMI_X1 = ModelInfo(
    str_identifier="sao10k/l3.1-70b-hanami-x1",
    price_in=3e-06,
    price_out=3e-06,
    creator="sao10k",
    description='''This is [Sao10K](/sao10k)'s experiment over [Euryale v2.2](/sao10k/l3.1-euryale-70b).,
    created=1736302854'''
)

L3_1_EURYALE_70B = ModelInfo(
    str_identifier="sao10k/l3.1-euryale-70b",
    price_in=6.5e-07,
    price_out=7.5e-07,
    creator="sao10k",
    description='''Euryale L3.1 70B v2.2 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.1](/models/sao10k/l3-euryale-70b).,
    created=1724803200'''
)

L3_3_EURYALE_70B = ModelInfo(
    str_identifier="sao10k/l3.3-euryale-70b",
    price_in=6.5e-07,
    price_out=7.5e-07,
    creator="sao10k",
    description='''Euryale L3.3 70B is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.2](/models/sao10k/l3-euryale-70b).,
    created=1734535928'''
)

STEP3 = ModelInfo(
    str_identifier="stepfun-ai/step3",
    price_in=5.7e-07,
    price_out=1.42e-06,
    creator="stepfun-ai",
    description='''Step3 is a cutting-edge multimodal reasoning model—built on a Mixture-of-Experts architecture with 321B total parameters and 38B active. It is designed end-to-end to minimize decoding costs while delivering top-tier performance in vision–language reasoning. Through the co-design of Multi-Matrix Factorization Attention (MFA) and Attention-FFN Disaggregation (AFD), Step3 maintains exceptional efficiency across both flagship and low-end accelerators.,
    created=1756415375'''
)

ROUTER = ModelInfo(
    str_identifier="switchpoint/router",
    price_in=8.5e-07,
    price_out=3.4e-06,
    creator="switchpoint",
    description='''Switchpoint AI's router instantly analyzes your request and directs it to the optimal AI from an ever-evolving library. 

As the world of LLMs advances, our router gets smarter, ensuring you always benefit from the industry's newest models without changing your workflow.

This model is configured for a simple, flat rate per response here on OpenRouter. It's powered by the full routing engine from [Switchpoint AI](https://www.switchpoint.dev).,
    created=1752272899'''
)

HUNYUAN_A13B_INSTRUCT = ModelInfo(
    str_identifier="tencent/hunyuan-a13b-instruct",
    price_in=1.4e-07,
    price_out=5.7e-07,
    creator="tencent",
    description='''Hunyuan-A13B is a 13B active parameter Mixture-of-Experts (MoE) language model developed by Tencent, with a total parameter count of 80B and support for reasoning via Chain-of-Thought. It offers competitive benchmark performance across mathematics, science, coding, and multi-turn reasoning tasks, while maintaining high inference efficiency via Grouped Query Attention (GQA) and quantization support (FP8, GPTQ, etc.).,
    created=1751987664'''
)

CYDONIA_24B_V4_1 = ModelInfo(
    str_identifier="thedrummer/cydonia-24b-v4.1",
    price_in=3e-07,
    price_out=5e-07,
    creator="thedrummer",
    description='''Uncensored and creative writing model based on Mistral Small 3.2 24B with good recall, prompt adherence, and intelligence.,
    created=1758931878'''
)

ROCINANTE_12B = ModelInfo(
    str_identifier="thedrummer/rocinante-12b",
    price_in=1.7e-07,
    price_out=4.3e-07,
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
    price_in=5.5e-07,
    price_out=8e-07,
    creator="thedrummer",
    description='''Skyfall 36B v2 is an enhanced iteration of Mistral Small 2501, specifically fine-tuned for improved creativity, nuanced writing, role-playing, and coherent storytelling.,
    created=1741636566'''
)

UNSLOPNEMO_12B = ModelInfo(
    str_identifier="thedrummer/unslopnemo-12b",
    price_in=4e-07,
    price_out=4e-07,
    creator="thedrummer",
    description='''UnslopNemo v4.1 is the latest addition from the creator of Rocinante, designed for adventure writing and role-play scenarios.,
    created=1731103448'''
)

DEEPSEEK_R1T_CHIMERA = ModelInfo(
    str_identifier="tngtech/deepseek-r1t-chimera",
    price_in=3e-07,
    price_out=1.2e-06,
    creator="tngtech",
    description='''DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), combining the reasoning capabilities of R1 with the token efficiency improvements of V3. It is based on a DeepSeek-MoE Transformer architecture and is optimized for general text generation tasks.

The model merges pretrained weights from both source models to balance performance across reasoning, efficiency, and instruction-following tasks. It is released under the MIT license and intended for research and commercial use.,
    created=1745760875'''
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

DEEPSEEK_R1T2_CHIMERA = ModelInfo(
    str_identifier="tngtech/deepseek-r1t2-chimera",
    price_in=2.5e-07,
    price_out=8.5e-07,
    creator="tngtech",
    description='''DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech. It is a 671 B-parameter mixture-of-experts text-generation model assembled from DeepSeek-AI’s R1-0528, R1, and V3-0324 checkpoints with an Assembly-of-Experts merge. The tri-parent design yields strong reasoning performance while running roughly 20 % faster than the original R1 and more than 2× faster than R1-0528 under vLLM, giving a favorable cost-to-intelligence trade-off. The checkpoint supports contexts up to 60 k tokens in standard use (tested to ~130 k) and maintains consistent <think> token behaviour, making it suitable for long-context analysis, dialogue and other open-ended generation tasks.,
    created=1751986985'''
)

DEEPSEEK_R1T2_CHIMERA_FREE = ModelInfo(
    str_identifier="tngtech/deepseek-r1t2-chimera:free",
    price_in=0.0,
    price_out=0.0,
    creator="tngtech",
    description='''DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech. It is a 671 B-parameter mixture-of-experts text-generation model assembled from DeepSeek-AI’s R1-0528, R1, and V3-0324 checkpoints with an Assembly-of-Experts merge. The tri-parent design yields strong reasoning performance while running roughly 20 % faster than the original R1 and more than 2× faster than R1-0528 under vLLM, giving a favorable cost-to-intelligence trade-off. The checkpoint supports contexts up to 60 k tokens in standard use (tested to ~130 k) and maintains consistent <think> token behaviour, making it suitable for long-context analysis, dialogue and other open-ended generation tasks.,
    created=1751986985'''
)

TNG_R1T_CHIMERA = ModelInfo(
    str_identifier="tngtech/tng-r1t-chimera",
    price_in=2.5e-07,
    price_out=8.5e-07,
    creator="tngtech",
    description='''TNG-R1T-Chimera is an experimental LLM with a faible for creative storytelling and character interaction. It is a derivate of the original TNG/DeepSeek-R1T-Chimera released in April 2025 and is available exclusively via Chutes and OpenRouter.

Characteristics and improvements include:

We think that it has a creative and pleasant personality.
It has a preliminary EQ-Bench3 value of about 1305.
It is quite a bit more intelligent than the original, albeit a slightly slower.
It is much more think-token consistent, i.e. reasoning and answer blocks are properly delineated.
Tool calling is much improved.

TNG Tech, the model authors, ask that users follow the careful guidelines that Microsoft has created for their "MAI-DS-R1" DeepSeek-based model. These guidelines are available on Hugging Face (https://huggingface.co/microsoft/MAI-DS-R1).,
    created=1764184161'''
)

TNG_R1T_CHIMERA_FREE = ModelInfo(
    str_identifier="tngtech/tng-r1t-chimera:free",
    price_in=0.0,
    price_out=0.0,
    creator="tngtech",
    description='''TNG-R1T-Chimera is an experimental LLM with a faible for creative storytelling and character interaction. It is a derivate of the original TNG/DeepSeek-R1T-Chimera released in April 2025 and is available exclusively via Chutes and OpenRouter.

Characteristics and improvements include:

We think that it has a creative and pleasant personality.
It has a preliminary EQ-Bench3 value of about 1305.
It is quite a bit more intelligent than the original, albeit a slightly slower.
It is much more think-token consistent, i.e. reasoning and answer blocks are properly delineated.
Tool calling is much improved.

TNG Tech, the model authors, ask that users follow the careful guidelines that Microsoft has created for their "MAI-DS-R1" DeepSeek-based model. These guidelines are available on Hugging Face (https://huggingface.co/microsoft/MAI-DS-R1).,
    created=1764184161'''
)

REMM_SLERP_L2_13B = ModelInfo(
    str_identifier="undi95/remm-slerp-l2-13b",
    price_in=4.5e-07,
    price_out=6.5e-07,
    creator="undi95",
    description='''A recreation trial of the original MythoMax-L2-B13 but with updated models. #merge,
    created=1689984000'''
)

PALMYRA_X5 = ModelInfo(
    str_identifier="writer/palmyra-x5",
    price_in=6e-07,
    price_out=6e-06,
    creator="writer",
    description='''Palmyra X5 is Writer's most advanced model, purpose-built for building and scaling AI agents across the enterprise. It delivers industry-leading speed and efficiency on context windows up to 1 million tokens, powered by a novel transformer architecture and hybrid attention mechanisms. This enables faster inference and expanded memory for processing large volumes of enterprise data, critical for scaling AI agents.,
    created=1769003823'''
)

GROK_3 = ModelInfo(
    str_identifier="x-ai/grok-3",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="x-ai",
    description='''Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases like data extraction, coding, and text summarization. Possesses deep domain knowledge in finance, healthcare, law, and science.

,
    created=1749582908'''
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

GROK_3_MINI = ModelInfo(
    str_identifier="x-ai/grok-3-mini",
    price_in=3e-07,
    price_out=5e-07,
    creator="x-ai",
    description='''A lightweight model that thinks before responding. Fast, smart, and great for logic-based tasks that do not require deep domain knowledge. The raw thinking traces are accessible.,
    created=1749583245'''
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

GROK_4 = ModelInfo(
    str_identifier="x-ai/grok-4",
    price_in=3e-06,
    price_out=1.5e-05,
    creator="x-ai",
    description='''Grok 4 is xAI's latest reasoning model with a 256k context window. It supports parallel tool calling, structured outputs, and both image and text inputs. Note that reasoning is not exposed, reasoning cannot be disabled, and the reasoning effort cannot be specified. Pricing increases once the total tokens in a given request is greater than 128k tokens. See more details on the [xAI docs](https://docs.x.ai/docs/models/grok-4-0709),
    created=1752087689'''
)

GROK_4_FAST = ModelInfo(
    str_identifier="x-ai/grok-4-fast",
    price_in=2e-07,
    price_out=5e-07,
    creator="x-ai",
    description='''Grok 4 Fast is xAI's latest multimodal model with SOTA cost-efficiency and a 2M token context window. It comes in two flavors: non-reasoning and reasoning. Read more about the model on xAI's [news post](http://x.ai/news/grok-4-fast).

Reasoning can be enabled/disabled using the `reasoning` `enabled` parameter in the API. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens),
    created=1758240090'''
)

GROK_4_1_FAST = ModelInfo(
    str_identifier="x-ai/grok-4.1-fast",
    price_in=2e-07,
    price_out=5e-07,
    creator="x-ai",
    description='''Grok 4.1 Fast is xAI's best agentic tool calling model that shines in real-world use cases like customer support and deep research. 2M context window.

Reasoning can be enabled/disabled using the `reasoning` `enabled` parameter in the API. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens),
    created=1763587502'''
)

GROK_CODE_FAST_1 = ModelInfo(
    str_identifier="x-ai/grok-code-fast-1",
    price_in=2e-07,
    price_out=1.5e-06,
    creator="x-ai",
    description='''Grok Code Fast 1 is a speedy and economical reasoning model that excels at agentic coding. With reasoning traces visible in the response, developers can steer Grok Code for high-quality work flows.,
    created=1756238927'''
)

MIMO_V2_FLASH = ModelInfo(
    str_identifier="xiaomi/mimo-v2-flash",
    price_in=9e-08,
    price_out=2.9e-07,
    creator="xiaomi",
    description='''MiMo-V2-Flash is an open-source foundation language model developed by Xiaomi. It is a Mixture-of-Experts model with 309B total parameters and 15B active parameters, adopting hybrid attention architecture. MiMo-V2-Flash supports a hybrid-thinking toggle and a 256K context window, and excels at reasoning, coding, and agent scenarios. On SWE-bench Verified and SWE-bench Multilingual, MiMo-V2-Flash ranks as the top #1 open-source model globally, delivering performance comparable to Claude Sonnet 4.5 while costing only about 3.5% as much.

Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config).,
    created=1765731308'''
)

MIMO_V2_FLASH_FREE = ModelInfo(
    str_identifier="xiaomi/mimo-v2-flash:free",
    price_in=0.0,
    price_out=0.0,
    creator="xiaomi",
    description='''MiMo-V2-Flash is an open-source foundation language model developed by Xiaomi. It is a Mixture-of-Experts model with 309B total parameters and 15B active parameters, adopting hybrid attention architecture. MiMo-V2-Flash supports a hybrid-thinking toggle and a 256K context window, and excels at reasoning, coding, and agent scenarios. On SWE-bench Verified and SWE-bench Multilingual, MiMo-V2-Flash ranks as the top #1 open-source model globally, delivering performance comparable to Claude Sonnet 4.5 while costing only about 3.5% as much.

Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config).,
    created=1765731308'''
)

GLM_4_32B = ModelInfo(
    str_identifier="z-ai/glm-4-32b",
    price_in=1e-07,
    price_out=1e-07,
    creator="z-ai",
    description='''GLM 4 32B is a cost-effective foundation language model.

It can efficiently perform complex tasks and has significantly enhanced capabilities in tool use, online search, and code-related intelligent tasks.

It is made by the same lab behind the thudm models.,
    created=1753376617'''
)

GLM_4_5 = ModelInfo(
    str_identifier="z-ai/glm-4.5",
    price_in=3.5e-07,
    price_out=1.55e-06,
    creator="z-ai",
    description='''GLM-4.5 is our latest flagship foundation model, purpose-built for agent-based applications. It leverages a Mixture-of-Experts (MoE) architecture and supports a context length of up to 128k tokens. GLM-4.5 delivers significantly enhanced capabilities in reasoning, code generation, and agent alignment. It supports a hybrid inference mode with two options, a "thinking mode" designed for complex reasoning and tool use, and a "non-thinking mode" optimized for instant responses. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config),
    created=1753471347'''
)

GLM_4_5_AIR = ModelInfo(
    str_identifier="z-ai/glm-4.5-air",
    price_in=5e-08,
    price_out=2.2e-07,
    creator="z-ai",
    description='''GLM-4.5-Air is the lightweight variant of our latest flagship model family, also purpose-built for agent-centric applications. Like GLM-4.5, it adopts the Mixture-of-Experts (MoE) architecture but with a more compact parameter size. GLM-4.5-Air also supports hybrid inference modes, offering a "thinking mode" for advanced reasoning and tool use, and a "non-thinking mode" for real-time interaction. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config),
    created=1753471258'''
)

GLM_4_5_AIR_FREE = ModelInfo(
    str_identifier="z-ai/glm-4.5-air:free",
    price_in=0.0,
    price_out=0.0,
    creator="z-ai",
    description='''GLM-4.5-Air is the lightweight variant of our latest flagship model family, also purpose-built for agent-centric applications. Like GLM-4.5, it adopts the Mixture-of-Experts (MoE) architecture but with a more compact parameter size. GLM-4.5-Air also supports hybrid inference modes, offering a "thinking mode" for advanced reasoning and tool use, and a "non-thinking mode" for real-time interaction. Users can control the reasoning behaviour with the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config),
    created=1753471258'''
)

GLM_4_5V = ModelInfo(
    str_identifier="z-ai/glm-4.5v",
    price_in=6e-07,
    price_out=1.8e-06,
    creator="z-ai",
    description='''GLM-4.5V is a vision-language foundation model for multimodal agent applications. Built on a Mixture-of-Experts (MoE) architecture with 106B parameters and 12B activated parameters, it achieves state-of-the-art results in video understanding, image Q&A, OCR, and document parsing, with strong gains in front-end web coding, grounding, and spatial reasoning. It offers a hybrid inference mode: a "thinking mode" for deep reasoning and a "non-thinking mode" for fast responses. Reasoning behavior can be toggled via the `reasoning` `enabled` boolean. [Learn more in our docs](https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config),
    created=1754922288'''
)

GLM_4_6 = ModelInfo(
    str_identifier="z-ai/glm-4.6",
    price_in=3.5e-07,
    price_out=1.5e-06,
    creator="z-ai",
    description='''Compared with GLM-4.5, this generation brings several key improvements:

Longer context window: The context window has been expanded from 128K to 200K tokens, enabling the model to handle more complex agentic tasks.
Superior coding performance: The model achieves higher scores on code benchmarks and demonstrates better real-world performance in applications such as Claude Code、Cline、Roo Code and Kilo Code, including improvements in generating visually polished front-end pages.
Advanced reasoning: GLM-4.6 shows a clear improvement in reasoning performance and supports tool use during inference, leading to stronger overall capability.
More capable agents: GLM-4.6 exhibits stronger performance in tool using and search-based agents, and integrates more effectively within agent frameworks.
Refined writing: Better aligns with human preferences in style and readability, and performs more naturally in role-playing scenarios.,
    created=1759235576'''
)

GLM_4_6_EXACTO = ModelInfo(
    str_identifier="z-ai/glm-4.6:exacto",
    price_in=4.4e-07,
    price_out=1.76e-06,
    creator="z-ai",
    description='''Compared with GLM-4.5, this generation brings several key improvements:

Longer context window: The context window has been expanded from 128K to 200K tokens, enabling the model to handle more complex agentic tasks.
Superior coding performance: The model achieves higher scores on code benchmarks and demonstrates better real-world performance in applications such as Claude Code、Cline、Roo Code and Kilo Code, including improvements in generating visually polished front-end pages.
Advanced reasoning: GLM-4.6 shows a clear improvement in reasoning performance and supports tool use during inference, leading to stronger overall capability.
More capable agents: GLM-4.6 exhibits stronger performance in tool using and search-based agents, and integrates more effectively within agent frameworks.
Refined writing: Better aligns with human preferences in style and readability, and performs more naturally in role-playing scenarios.,
    created=1759235576'''
)

GLM_4_6V = ModelInfo(
    str_identifier="z-ai/glm-4.6v",
    price_in=3e-07,
    price_out=9e-07,
    creator="z-ai",
    description='''GLM-4.6V is a large multimodal model designed for high-fidelity visual understanding and long-context reasoning across images, documents, and mixed media. It supports up to 128K tokens, processes complex page layouts and charts directly as visual inputs, and integrates native multimodal function calling to connect perception with downstream tool execution. The model also enables interleaved image-text generation and UI reconstruction workflows, including screenshot-to-HTML synthesis and iterative visual editing.,
    created=1765207462'''
)

GLM_4_7 = ModelInfo(
    str_identifier="z-ai/glm-4.7",
    price_in=4e-07,
    price_out=1.5e-06,
    creator="z-ai",
    description='''GLM-4.7 is Z.AI’s latest flagship model, featuring upgrades in two key areas: enhanced programming capabilities and more stable multi-step reasoning/execution. It demonstrates significant improvements in executing complex agent tasks while delivering more natural conversational experiences and superior front-end aesthetics.,
    created=1766378014'''
)

GLM_4_7_FLASH = ModelInfo(
    str_identifier="z-ai/glm-4.7-flash",
    price_in=7e-08,
    price_out=4e-07,
    creator="z-ai",
    description='''As a 30B-class SOTA model, GLM-4.7-Flash offers a new option that balances performance and efficiency. It is further optimized for agentic coding use cases, strengthening coding capabilities, long-horizon task planning, and tool collaboration, and has achieved leading performance among open-source models of the same size on several current public benchmark leaderboards.,
    created=1768833913'''
)

ALL_MODELS = [
    JAMBA_LARGE_1_7,
    JAMBA_MINI_1_7,
    AION_1_0,
    AION_1_0_MINI,
    AION_RP_LLAMA_3_1_8B,
    CODELLAMA_7B_INSTRUCT_SOLIDITY,
    TONGYI_DEEPRESEARCH_30B_A3B,
    MOLMO_2_8B_FREE,
    OLMO_2_0325_32B_INSTRUCT,
    OLMO_3_32B_THINK,
    OLMO_3_7B_INSTRUCT,
    OLMO_3_7B_THINK,
    OLMO_3_1_32B_INSTRUCT,
    OLMO_3_1_32B_THINK,
    GOLIATH_120B,
    NOVA_2_LITE_V1,
    NOVA_LITE_V1,
    NOVA_MICRO_V1,
    NOVA_PREMIER_V1,
    NOVA_PRO_V1,
    MAGNUM_V4_72B,
    CLAUDE_3_HAIKU,
    CLAUDE_3_5_HAIKU,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_7_SONNET,
    CLAUDE_3_7_SONNET_THINKING,
    CLAUDE_HAIKU_4_5,
    CLAUDE_OPUS_4,
    CLAUDE_OPUS_4_1,
    CLAUDE_OPUS_4_5,
    CLAUDE_SONNET_4,
    CLAUDE_SONNET_4_5,
    CODER_LARGE,
    MAESTRO_REASONING,
    SPOTLIGHT,
    TRINITY_MINI,
    TRINITY_MINI_FREE,
    VIRTUOSO_LARGE,
    ERNIE_4_5_21B_A3B,
    ERNIE_4_5_21B_A3B_THINKING,
    ERNIE_4_5_300B_A47B,
    ERNIE_4_5_VL_28B_A3B,
    ERNIE_4_5_VL_424B_A47B,
    SEED_1_6,
    SEED_1_6_FLASH,
    UI_TARS_1_5_7B,
    DOLPHIN_MISTRAL_24B_VENICE_EDITION_FREE,
    COMMAND_A,
    COMMAND_R_08_2024,
    COMMAND_R_PLUS_08_2024,
    COMMAND_R7B_12_2024,
    COGITO_V2_PREVIEW_LLAMA_109B_MOE,
    COGITO_V2_PREVIEW_LLAMA_405B,
    COGITO_V2_PREVIEW_LLAMA_70B,
    COGITO_V2_1_671B,
    DEEPSEEK_CHAT,
    DEEPSEEK_CHAT_V3_0324,
    DEEPSEEK_CHAT_V3_1,
    DEEPSEEK_R1,
    DEEPSEEK_R1_0528,
    DEEPSEEK_R1_0528_FREE,
    DEEPSEEK_R1_DISTILL_LLAMA_70B,
    DEEPSEEK_R1_DISTILL_QWEN_32B,
    DEEPSEEK_V3_1_TERMINUS,
    DEEPSEEK_V3_1_TERMINUS_EXACTO,
    DEEPSEEK_V3_2,
    DEEPSEEK_V3_2_EXP,
    DEEPSEEK_V3_2_SPECIALE,
    LLEMMA_7B,
    RNJ_1_INSTRUCT,
    GEMINI_2_0_FLASH_001,
    GEMINI_2_0_FLASH_EXP_FREE,
    GEMINI_2_0_FLASH_LITE_001,
    GEMINI_2_5_FLASH,
    GEMINI_2_5_FLASH_IMAGE,
    GEMINI_2_5_FLASH_LITE,
    GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025,
    GEMINI_2_5_FLASH_PREVIEW_09_2025,
    GEMINI_2_5_PRO,
    GEMINI_2_5_PRO_PREVIEW,
    GEMINI_2_5_PRO_PREVIEW_05_06,
    GEMINI_3_FLASH_PREVIEW,
    GEMINI_3_PRO_IMAGE_PREVIEW,
    GEMINI_3_PRO_PREVIEW,
    GEMMA_2_27B_IT,
    GEMMA_2_9B_IT,
    GEMMA_3_12B_IT,
    GEMMA_3_12B_IT_FREE,
    GEMMA_3_27B_IT,
    GEMMA_3_27B_IT_FREE,
    GEMMA_3_4B_IT,
    GEMMA_3_4B_IT_FREE,
    GEMMA_3N_E2B_IT_FREE,
    GEMMA_3N_E4B_IT,
    GEMMA_3N_E4B_IT_FREE,
    MYTHOMAX_L2_13B,
    GRANITE_4_0_H_MICRO,
    MERCURY,
    MERCURY_CODER,
    INFLECTION_3_PI,
    INFLECTION_3_PRODUCTIVITY,
    KAT_CODER_PRO,
    LFM_2_2_6B,
    LFM_2_5_1_2B_INSTRUCT_FREE,
    LFM_2_5_1_2B_THINKING_FREE,
    LFM2_8B_A1B,
    WEAVER,
    LONGCAT_FLASH_CHAT,
    LLAMA_3_70B_INSTRUCT,
    LLAMA_3_8B_INSTRUCT,
    LLAMA_3_1_405B,
    LLAMA_3_1_405B_INSTRUCT,
    LLAMA_3_1_405B_INSTRUCT_FREE,
    LLAMA_3_1_70B_INSTRUCT,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_2_11B_VISION_INSTRUCT,
    LLAMA_3_2_1B_INSTRUCT,
    LLAMA_3_2_3B_INSTRUCT,
    LLAMA_3_2_3B_INSTRUCT_FREE,
    LLAMA_3_3_70B_INSTRUCT,
    LLAMA_3_3_70B_INSTRUCT_FREE,
    LLAMA_4_MAVERICK,
    LLAMA_4_SCOUT,
    LLAMA_GUARD_2_8B,
    LLAMA_GUARD_3_8B,
    LLAMA_GUARD_4_12B,
    PHI_4,
    WIZARDLM_2_8X22B,
    MINIMAX_01,
    MINIMAX_M1,
    MINIMAX_M2,
    MINIMAX_M2_HER,
    MINIMAX_M2_1,
    CODESTRAL_2508,
    DEVSTRAL_2512,
    DEVSTRAL_2512_FREE,
    DEVSTRAL_MEDIUM,
    DEVSTRAL_SMALL,
    MINISTRAL_14B_2512,
    MINISTRAL_3B,
    MINISTRAL_3B_2512,
    MINISTRAL_8B,
    MINISTRAL_8B_2512,
    MISTRAL_7B_INSTRUCT,
    MISTRAL_7B_INSTRUCT_V0_1,
    MISTRAL_7B_INSTRUCT_V0_2,
    MISTRAL_7B_INSTRUCT_V0_3,
    MISTRAL_LARGE,
    MISTRAL_LARGE_2407,
    MISTRAL_LARGE_2411,
    MISTRAL_LARGE_2512,
    MISTRAL_MEDIUM_3,
    MISTRAL_MEDIUM_3_1,
    MISTRAL_NEMO,
    MISTRAL_SABA,
    MISTRAL_SMALL_24B_INSTRUCT_2501,
    MISTRAL_SMALL_3_1_24B_INSTRUCT,
    MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE,
    MISTRAL_SMALL_3_2_24B_INSTRUCT,
    MISTRAL_SMALL_CREATIVE,
    MISTRAL_TINY,
    MIXTRAL_8X22B_INSTRUCT,
    MIXTRAL_8X7B_INSTRUCT,
    PIXTRAL_12B,
    PIXTRAL_LARGE_2411,
    VOXTRAL_SMALL_24B_2507,
    KIMI_DEV_72B,
    KIMI_K2,
    KIMI_K2_0905,
    KIMI_K2_0905_EXACTO,
    KIMI_K2_THINKING,
    KIMI_K2_FREE,
    MORPH_V3_FAST,
    MORPH_V3_LARGE,
    LLAMA_3_1_LUMIMAID_8B,
    NOROMAID_20B,
    DEEPSEEK_V3_1_NEX_N1,
    DEEPHERMES_3_MISTRAL_24B_PREVIEW,
    HERMES_2_PRO_LLAMA_3_8B,
    HERMES_3_LLAMA_3_1_405B,
    HERMES_3_LLAMA_3_1_405B_FREE,
    HERMES_3_LLAMA_3_1_70B,
    HERMES_4_405B,
    HERMES_4_70B,
    LLAMA_3_1_NEMOTRON_70B_INSTRUCT,
    LLAMA_3_1_NEMOTRON_ULTRA_253B_V1,
    LLAMA_3_3_NEMOTRON_SUPER_49B_V1_5,
    NEMOTRON_3_NANO_30B_A3B,
    NEMOTRON_3_NANO_30B_A3B_FREE,
    NEMOTRON_NANO_12B_V2_VL,
    NEMOTRON_NANO_12B_V2_VL_FREE,
    NEMOTRON_NANO_9B_V2,
    NEMOTRON_NANO_9B_V2_FREE,
    CHATGPT_4O_LATEST,
    GPT_3_5_TURBO,
    GPT_3_5_TURBO_0613,
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
    GPT_4O,
    GPT_4O_2024_05_13,
    GPT_4O_2024_08_06,
    GPT_4O_2024_11_20,
    GPT_4O_AUDIO_PREVIEW,
    GPT_4O_MINI,
    GPT_4O_MINI_2024_07_18,
    GPT_4O_MINI_SEARCH_PREVIEW,
    GPT_4O_SEARCH_PREVIEW,
    GPT_4O_EXTENDED,
    GPT_5,
    GPT_5_CHAT,
    GPT_5_CODEX,
    GPT_5_IMAGE,
    GPT_5_IMAGE_MINI,
    GPT_5_MINI,
    GPT_5_NANO,
    GPT_5_PRO,
    GPT_5_1,
    GPT_5_1_CHAT,
    GPT_5_1_CODEX,
    GPT_5_1_CODEX_MAX,
    GPT_5_1_CODEX_MINI,
    GPT_5_2,
    GPT_5_2_CHAT,
    GPT_5_2_CODEX,
    GPT_5_2_PRO,
    GPT_AUDIO,
    GPT_AUDIO_MINI,
    GPT_OSS_120B,
    GPT_OSS_120B_EXACTO,
    GPT_OSS_120B_FREE,
    GPT_OSS_20B,
    GPT_OSS_20B_FREE,
    GPT_OSS_SAFEGUARD_20B,
    O1,
    O1_PRO,
    O3,
    O3_DEEP_RESEARCH,
    O3_MINI,
    O3_MINI_HIGH,
    O3_PRO,
    O4_MINI,
    O4_MINI_DEEP_RESEARCH,
    O4_MINI_HIGH,
    INTERNVL3_78B,
    AUTO,
    BODYBUILDER,
    SONAR,
    SONAR_DEEP_RESEARCH,
    SONAR_PRO,
    SONAR_PRO_SEARCH,
    SONAR_REASONING_PRO,
    INTELLECT_3,
    QWEN_2_5_72B_INSTRUCT,
    QWEN_2_5_7B_INSTRUCT,
    QWEN_2_5_CODER_32B_INSTRUCT,
    QWEN_2_5_VL_7B_INSTRUCT,
    QWEN_2_5_VL_7B_INSTRUCT_FREE,
    QWEN_MAX,
    QWEN_PLUS,
    QWEN_PLUS_2025_07_28,
    QWEN_PLUS_2025_07_28_THINKING,
    QWEN_TURBO,
    QWEN_VL_MAX,
    QWEN_VL_PLUS,
    QWEN2_5_CODER_7B_INSTRUCT,
    QWEN2_5_VL_32B_INSTRUCT,
    QWEN2_5_VL_72B_INSTRUCT,
    QWEN3_14B,
    QWEN3_235B_A22B,
    QWEN3_235B_A22B_2507,
    QWEN3_235B_A22B_THINKING_2507,
    QWEN3_30B_A3B,
    QWEN3_30B_A3B_INSTRUCT_2507,
    QWEN3_30B_A3B_THINKING_2507,
    QWEN3_32B,
    QWEN3_4B_FREE,
    QWEN3_8B,
    QWEN3_CODER,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    QWEN3_CODER_FLASH,
    QWEN3_CODER_PLUS,
    QWEN3_CODER_EXACTO,
    QWEN3_CODER_FREE,
    QWEN3_MAX,
    QWEN3_NEXT_80B_A3B_INSTRUCT,
    QWEN3_NEXT_80B_A3B_INSTRUCT_FREE,
    QWEN3_NEXT_80B_A3B_THINKING,
    QWEN3_VL_235B_A22B_INSTRUCT,
    QWEN3_VL_235B_A22B_THINKING,
    QWEN3_VL_30B_A3B_INSTRUCT,
    QWEN3_VL_30B_A3B_THINKING,
    QWEN3_VL_32B_INSTRUCT,
    QWEN3_VL_8B_INSTRUCT,
    QWEN3_VL_8B_THINKING,
    QWQ_32B,
    SORCERERLM_8X22B,
    RELACE_APPLY_3,
    RELACE_SEARCH,
    L3_EURYALE_70B,
    L3_LUNARIS_8B,
    L3_1_70B_HANAMI_X1,
    L3_1_EURYALE_70B,
    L3_3_EURYALE_70B,
    STEP3,
    ROUTER,
    HUNYUAN_A13B_INSTRUCT,
    CYDONIA_24B_V4_1,
    ROCINANTE_12B,
    SKYFALL_36B_V2,
    UNSLOPNEMO_12B,
    DEEPSEEK_R1T_CHIMERA,
    DEEPSEEK_R1T_CHIMERA_FREE,
    DEEPSEEK_R1T2_CHIMERA,
    DEEPSEEK_R1T2_CHIMERA_FREE,
    TNG_R1T_CHIMERA,
    TNG_R1T_CHIMERA_FREE,
    REMM_SLERP_L2_13B,
    PALMYRA_X5,
    GROK_3,
    GROK_3_BETA,
    GROK_3_MINI,
    GROK_3_MINI_BETA,
    GROK_4,
    GROK_4_FAST,
    GROK_4_1_FAST,
    GROK_CODE_FAST_1,
    MIMO_V2_FLASH,
    MIMO_V2_FLASH_FREE,
    GLM_4_32B,
    GLM_4_5,
    GLM_4_5_AIR,
    GLM_4_5_AIR_FREE,
    GLM_4_5V,
    GLM_4_6,
    GLM_4_6_EXACTO,
    GLM_4_6V,
    GLM_4_7,
    GLM_4_7_FLASH,
]
FREE_MODELS = [
    MOLMO_2_8B_FREE,
    TRINITY_MINI_FREE,
    DOLPHIN_MISTRAL_24B_VENICE_EDITION_FREE,
    DEEPSEEK_R1_0528_FREE,
    GEMINI_2_0_FLASH_EXP_FREE,
    GEMMA_3_12B_IT_FREE,
    GEMMA_3_27B_IT_FREE,
    GEMMA_3_4B_IT_FREE,
    GEMMA_3N_E2B_IT_FREE,
    GEMMA_3N_E4B_IT_FREE,
    LFM_2_5_1_2B_INSTRUCT_FREE,
    LFM_2_5_1_2B_THINKING_FREE,
    LLAMA_3_1_405B_INSTRUCT_FREE,
    LLAMA_3_2_3B_INSTRUCT_FREE,
    LLAMA_3_3_70B_INSTRUCT_FREE,
    DEVSTRAL_2512_FREE,
    MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE,
    KIMI_K2_FREE,
    HERMES_3_LLAMA_3_1_405B_FREE,
    NEMOTRON_3_NANO_30B_A3B_FREE,
    NEMOTRON_NANO_12B_V2_VL_FREE,
    NEMOTRON_NANO_9B_V2_FREE,
    GPT_OSS_120B_FREE,
    GPT_OSS_20B_FREE,
    QWEN_2_5_VL_7B_INSTRUCT_FREE,
    QWEN3_4B_FREE,
    QWEN3_CODER_FREE,
    QWEN3_NEXT_80B_A3B_INSTRUCT_FREE,
    DEEPSEEK_R1T_CHIMERA_FREE,
    DEEPSEEK_R1T2_CHIMERA_FREE,
    TNG_R1T_CHIMERA_FREE,
    MIMO_V2_FLASH_FREE,
    GLM_4_5_AIR_FREE,
]
ALL_MODELS_DICT = {
    "JAMBA_LARGE_1_7": JAMBA_LARGE_1_7,
    "JAMBA_MINI_1_7": JAMBA_MINI_1_7,
    "AION_1_0": AION_1_0,
    "AION_1_0_MINI": AION_1_0_MINI,
    "AION_RP_LLAMA_3_1_8B": AION_RP_LLAMA_3_1_8B,
    "CODELLAMA_7B_INSTRUCT_SOLIDITY": CODELLAMA_7B_INSTRUCT_SOLIDITY,
    "TONGYI_DEEPRESEARCH_30B_A3B": TONGYI_DEEPRESEARCH_30B_A3B,
    "MOLMO_2_8B_FREE": MOLMO_2_8B_FREE,
    "OLMO_2_0325_32B_INSTRUCT": OLMO_2_0325_32B_INSTRUCT,
    "OLMO_3_32B_THINK": OLMO_3_32B_THINK,
    "OLMO_3_7B_INSTRUCT": OLMO_3_7B_INSTRUCT,
    "OLMO_3_7B_THINK": OLMO_3_7B_THINK,
    "OLMO_3_1_32B_INSTRUCT": OLMO_3_1_32B_INSTRUCT,
    "OLMO_3_1_32B_THINK": OLMO_3_1_32B_THINK,
    "GOLIATH_120B": GOLIATH_120B,
    "NOVA_2_LITE_V1": NOVA_2_LITE_V1,
    "NOVA_LITE_V1": NOVA_LITE_V1,
    "NOVA_MICRO_V1": NOVA_MICRO_V1,
    "NOVA_PREMIER_V1": NOVA_PREMIER_V1,
    "NOVA_PRO_V1": NOVA_PRO_V1,
    "MAGNUM_V4_72B": MAGNUM_V4_72B,
    "CLAUDE_3_HAIKU": CLAUDE_3_HAIKU,
    "CLAUDE_3_5_HAIKU": CLAUDE_3_5_HAIKU,
    "CLAUDE_3_5_SONNET": CLAUDE_3_5_SONNET,
    "CLAUDE_3_7_SONNET": CLAUDE_3_7_SONNET,
    "CLAUDE_3_7_SONNET_THINKING": CLAUDE_3_7_SONNET_THINKING,
    "CLAUDE_HAIKU_4_5": CLAUDE_HAIKU_4_5,
    "CLAUDE_OPUS_4": CLAUDE_OPUS_4,
    "CLAUDE_OPUS_4_1": CLAUDE_OPUS_4_1,
    "CLAUDE_OPUS_4_5": CLAUDE_OPUS_4_5,
    "CLAUDE_SONNET_4": CLAUDE_SONNET_4,
    "CLAUDE_SONNET_4_5": CLAUDE_SONNET_4_5,
    "CODER_LARGE": CODER_LARGE,
    "MAESTRO_REASONING": MAESTRO_REASONING,
    "SPOTLIGHT": SPOTLIGHT,
    "TRINITY_MINI": TRINITY_MINI,
    "TRINITY_MINI_FREE": TRINITY_MINI_FREE,
    "VIRTUOSO_LARGE": VIRTUOSO_LARGE,
    "ERNIE_4_5_21B_A3B": ERNIE_4_5_21B_A3B,
    "ERNIE_4_5_21B_A3B_THINKING": ERNIE_4_5_21B_A3B_THINKING,
    "ERNIE_4_5_300B_A47B": ERNIE_4_5_300B_A47B,
    "ERNIE_4_5_VL_28B_A3B": ERNIE_4_5_VL_28B_A3B,
    "ERNIE_4_5_VL_424B_A47B": ERNIE_4_5_VL_424B_A47B,
    "SEED_1_6": SEED_1_6,
    "SEED_1_6_FLASH": SEED_1_6_FLASH,
    "UI_TARS_1_5_7B": UI_TARS_1_5_7B,
    "DOLPHIN_MISTRAL_24B_VENICE_EDITION_FREE": DOLPHIN_MISTRAL_24B_VENICE_EDITION_FREE,
    "COMMAND_A": COMMAND_A,
    "COMMAND_R_08_2024": COMMAND_R_08_2024,
    "COMMAND_R_PLUS_08_2024": COMMAND_R_PLUS_08_2024,
    "COMMAND_R7B_12_2024": COMMAND_R7B_12_2024,
    "COGITO_V2_PREVIEW_LLAMA_109B_MOE": COGITO_V2_PREVIEW_LLAMA_109B_MOE,
    "COGITO_V2_PREVIEW_LLAMA_405B": COGITO_V2_PREVIEW_LLAMA_405B,
    "COGITO_V2_PREVIEW_LLAMA_70B": COGITO_V2_PREVIEW_LLAMA_70B,
    "COGITO_V2_1_671B": COGITO_V2_1_671B,
    "DEEPSEEK_CHAT": DEEPSEEK_CHAT,
    "DEEPSEEK_CHAT_V3_0324": DEEPSEEK_CHAT_V3_0324,
    "DEEPSEEK_CHAT_V3_1": DEEPSEEK_CHAT_V3_1,
    "DEEPSEEK_R1": DEEPSEEK_R1,
    "DEEPSEEK_R1_0528": DEEPSEEK_R1_0528,
    "DEEPSEEK_R1_0528_FREE": DEEPSEEK_R1_0528_FREE,
    "DEEPSEEK_R1_DISTILL_LLAMA_70B": DEEPSEEK_R1_DISTILL_LLAMA_70B,
    "DEEPSEEK_R1_DISTILL_QWEN_32B": DEEPSEEK_R1_DISTILL_QWEN_32B,
    "DEEPSEEK_V3_1_TERMINUS": DEEPSEEK_V3_1_TERMINUS,
    "DEEPSEEK_V3_1_TERMINUS_EXACTO": DEEPSEEK_V3_1_TERMINUS_EXACTO,
    "DEEPSEEK_V3_2": DEEPSEEK_V3_2,
    "DEEPSEEK_V3_2_EXP": DEEPSEEK_V3_2_EXP,
    "DEEPSEEK_V3_2_SPECIALE": DEEPSEEK_V3_2_SPECIALE,
    "LLEMMA_7B": LLEMMA_7B,
    "RNJ_1_INSTRUCT": RNJ_1_INSTRUCT,
    "GEMINI_2_0_FLASH_001": GEMINI_2_0_FLASH_001,
    "GEMINI_2_0_FLASH_EXP_FREE": GEMINI_2_0_FLASH_EXP_FREE,
    "GEMINI_2_0_FLASH_LITE_001": GEMINI_2_0_FLASH_LITE_001,
    "GEMINI_2_5_FLASH": GEMINI_2_5_FLASH,
    "GEMINI_2_5_FLASH_IMAGE": GEMINI_2_5_FLASH_IMAGE,
    "GEMINI_2_5_FLASH_LITE": GEMINI_2_5_FLASH_LITE,
    "GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025": GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025,
    "GEMINI_2_5_FLASH_PREVIEW_09_2025": GEMINI_2_5_FLASH_PREVIEW_09_2025,
    "GEMINI_2_5_PRO": GEMINI_2_5_PRO,
    "GEMINI_2_5_PRO_PREVIEW": GEMINI_2_5_PRO_PREVIEW,
    "GEMINI_2_5_PRO_PREVIEW_05_06": GEMINI_2_5_PRO_PREVIEW_05_06,
    "GEMINI_3_FLASH_PREVIEW": GEMINI_3_FLASH_PREVIEW,
    "GEMINI_3_PRO_IMAGE_PREVIEW": GEMINI_3_PRO_IMAGE_PREVIEW,
    "GEMINI_3_PRO_PREVIEW": GEMINI_3_PRO_PREVIEW,
    "GEMMA_2_27B_IT": GEMMA_2_27B_IT,
    "GEMMA_2_9B_IT": GEMMA_2_9B_IT,
    "GEMMA_3_12B_IT": GEMMA_3_12B_IT,
    "GEMMA_3_12B_IT_FREE": GEMMA_3_12B_IT_FREE,
    "GEMMA_3_27B_IT": GEMMA_3_27B_IT,
    "GEMMA_3_27B_IT_FREE": GEMMA_3_27B_IT_FREE,
    "GEMMA_3_4B_IT": GEMMA_3_4B_IT,
    "GEMMA_3_4B_IT_FREE": GEMMA_3_4B_IT_FREE,
    "GEMMA_3N_E2B_IT_FREE": GEMMA_3N_E2B_IT_FREE,
    "GEMMA_3N_E4B_IT": GEMMA_3N_E4B_IT,
    "GEMMA_3N_E4B_IT_FREE": GEMMA_3N_E4B_IT_FREE,
    "MYTHOMAX_L2_13B": MYTHOMAX_L2_13B,
    "GRANITE_4_0_H_MICRO": GRANITE_4_0_H_MICRO,
    "MERCURY": MERCURY,
    "MERCURY_CODER": MERCURY_CODER,
    "INFLECTION_3_PI": INFLECTION_3_PI,
    "INFLECTION_3_PRODUCTIVITY": INFLECTION_3_PRODUCTIVITY,
    "KAT_CODER_PRO": KAT_CODER_PRO,
    "LFM_2_2_6B": LFM_2_2_6B,
    "LFM_2_5_1_2B_INSTRUCT_FREE": LFM_2_5_1_2B_INSTRUCT_FREE,
    "LFM_2_5_1_2B_THINKING_FREE": LFM_2_5_1_2B_THINKING_FREE,
    "LFM2_8B_A1B": LFM2_8B_A1B,
    "WEAVER": WEAVER,
    "LONGCAT_FLASH_CHAT": LONGCAT_FLASH_CHAT,
    "LLAMA_3_70B_INSTRUCT": LLAMA_3_70B_INSTRUCT,
    "LLAMA_3_8B_INSTRUCT": LLAMA_3_8B_INSTRUCT,
    "LLAMA_3_1_405B": LLAMA_3_1_405B,
    "LLAMA_3_1_405B_INSTRUCT": LLAMA_3_1_405B_INSTRUCT,
    "LLAMA_3_1_405B_INSTRUCT_FREE": LLAMA_3_1_405B_INSTRUCT_FREE,
    "LLAMA_3_1_70B_INSTRUCT": LLAMA_3_1_70B_INSTRUCT,
    "LLAMA_3_1_8B_INSTRUCT": LLAMA_3_1_8B_INSTRUCT,
    "LLAMA_3_2_11B_VISION_INSTRUCT": LLAMA_3_2_11B_VISION_INSTRUCT,
    "LLAMA_3_2_1B_INSTRUCT": LLAMA_3_2_1B_INSTRUCT,
    "LLAMA_3_2_3B_INSTRUCT": LLAMA_3_2_3B_INSTRUCT,
    "LLAMA_3_2_3B_INSTRUCT_FREE": LLAMA_3_2_3B_INSTRUCT_FREE,
    "LLAMA_3_3_70B_INSTRUCT": LLAMA_3_3_70B_INSTRUCT,
    "LLAMA_3_3_70B_INSTRUCT_FREE": LLAMA_3_3_70B_INSTRUCT_FREE,
    "LLAMA_4_MAVERICK": LLAMA_4_MAVERICK,
    "LLAMA_4_SCOUT": LLAMA_4_SCOUT,
    "LLAMA_GUARD_2_8B": LLAMA_GUARD_2_8B,
    "LLAMA_GUARD_3_8B": LLAMA_GUARD_3_8B,
    "LLAMA_GUARD_4_12B": LLAMA_GUARD_4_12B,
    "PHI_4": PHI_4,
    "WIZARDLM_2_8X22B": WIZARDLM_2_8X22B,
    "MINIMAX_01": MINIMAX_01,
    "MINIMAX_M1": MINIMAX_M1,
    "MINIMAX_M2": MINIMAX_M2,
    "MINIMAX_M2_HER": MINIMAX_M2_HER,
    "MINIMAX_M2_1": MINIMAX_M2_1,
    "CODESTRAL_2508": CODESTRAL_2508,
    "DEVSTRAL_2512": DEVSTRAL_2512,
    "DEVSTRAL_2512_FREE": DEVSTRAL_2512_FREE,
    "DEVSTRAL_MEDIUM": DEVSTRAL_MEDIUM,
    "DEVSTRAL_SMALL": DEVSTRAL_SMALL,
    "MINISTRAL_14B_2512": MINISTRAL_14B_2512,
    "MINISTRAL_3B": MINISTRAL_3B,
    "MINISTRAL_3B_2512": MINISTRAL_3B_2512,
    "MINISTRAL_8B": MINISTRAL_8B,
    "MINISTRAL_8B_2512": MINISTRAL_8B_2512,
    "MISTRAL_7B_INSTRUCT": MISTRAL_7B_INSTRUCT,
    "MISTRAL_7B_INSTRUCT_V0_1": MISTRAL_7B_INSTRUCT_V0_1,
    "MISTRAL_7B_INSTRUCT_V0_2": MISTRAL_7B_INSTRUCT_V0_2,
    "MISTRAL_7B_INSTRUCT_V0_3": MISTRAL_7B_INSTRUCT_V0_3,
    "MISTRAL_LARGE": MISTRAL_LARGE,
    "MISTRAL_LARGE_2407": MISTRAL_LARGE_2407,
    "MISTRAL_LARGE_2411": MISTRAL_LARGE_2411,
    "MISTRAL_LARGE_2512": MISTRAL_LARGE_2512,
    "MISTRAL_MEDIUM_3": MISTRAL_MEDIUM_3,
    "MISTRAL_MEDIUM_3_1": MISTRAL_MEDIUM_3_1,
    "MISTRAL_NEMO": MISTRAL_NEMO,
    "MISTRAL_SABA": MISTRAL_SABA,
    "MISTRAL_SMALL_24B_INSTRUCT_2501": MISTRAL_SMALL_24B_INSTRUCT_2501,
    "MISTRAL_SMALL_3_1_24B_INSTRUCT": MISTRAL_SMALL_3_1_24B_INSTRUCT,
    "MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE": MISTRAL_SMALL_3_1_24B_INSTRUCT_FREE,
    "MISTRAL_SMALL_3_2_24B_INSTRUCT": MISTRAL_SMALL_3_2_24B_INSTRUCT,
    "MISTRAL_SMALL_CREATIVE": MISTRAL_SMALL_CREATIVE,
    "MISTRAL_TINY": MISTRAL_TINY,
    "MIXTRAL_8X22B_INSTRUCT": MIXTRAL_8X22B_INSTRUCT,
    "MIXTRAL_8X7B_INSTRUCT": MIXTRAL_8X7B_INSTRUCT,
    "PIXTRAL_12B": PIXTRAL_12B,
    "PIXTRAL_LARGE_2411": PIXTRAL_LARGE_2411,
    "VOXTRAL_SMALL_24B_2507": VOXTRAL_SMALL_24B_2507,
    "KIMI_DEV_72B": KIMI_DEV_72B,
    "KIMI_K2": KIMI_K2,
    "KIMI_K2_0905": KIMI_K2_0905,
    "KIMI_K2_0905_EXACTO": KIMI_K2_0905_EXACTO,
    "KIMI_K2_THINKING": KIMI_K2_THINKING,
    "KIMI_K2_FREE": KIMI_K2_FREE,
    "MORPH_V3_FAST": MORPH_V3_FAST,
    "MORPH_V3_LARGE": MORPH_V3_LARGE,
    "LLAMA_3_1_LUMIMAID_8B": LLAMA_3_1_LUMIMAID_8B,
    "NOROMAID_20B": NOROMAID_20B,
    "DEEPSEEK_V3_1_NEX_N1": DEEPSEEK_V3_1_NEX_N1,
    "DEEPHERMES_3_MISTRAL_24B_PREVIEW": DEEPHERMES_3_MISTRAL_24B_PREVIEW,
    "HERMES_2_PRO_LLAMA_3_8B": HERMES_2_PRO_LLAMA_3_8B,
    "HERMES_3_LLAMA_3_1_405B": HERMES_3_LLAMA_3_1_405B,
    "HERMES_3_LLAMA_3_1_405B_FREE": HERMES_3_LLAMA_3_1_405B_FREE,
    "HERMES_3_LLAMA_3_1_70B": HERMES_3_LLAMA_3_1_70B,
    "HERMES_4_405B": HERMES_4_405B,
    "HERMES_4_70B": HERMES_4_70B,
    "LLAMA_3_1_NEMOTRON_70B_INSTRUCT": LLAMA_3_1_NEMOTRON_70B_INSTRUCT,
    "LLAMA_3_1_NEMOTRON_ULTRA_253B_V1": LLAMA_3_1_NEMOTRON_ULTRA_253B_V1,
    "LLAMA_3_3_NEMOTRON_SUPER_49B_V1_5": LLAMA_3_3_NEMOTRON_SUPER_49B_V1_5,
    "NEMOTRON_3_NANO_30B_A3B": NEMOTRON_3_NANO_30B_A3B,
    "NEMOTRON_3_NANO_30B_A3B_FREE": NEMOTRON_3_NANO_30B_A3B_FREE,
    "NEMOTRON_NANO_12B_V2_VL": NEMOTRON_NANO_12B_V2_VL,
    "NEMOTRON_NANO_12B_V2_VL_FREE": NEMOTRON_NANO_12B_V2_VL_FREE,
    "NEMOTRON_NANO_9B_V2": NEMOTRON_NANO_9B_V2,
    "NEMOTRON_NANO_9B_V2_FREE": NEMOTRON_NANO_9B_V2_FREE,
    "CHATGPT_4O_LATEST": CHATGPT_4O_LATEST,
    "GPT_3_5_TURBO": GPT_3_5_TURBO,
    "GPT_3_5_TURBO_0613": GPT_3_5_TURBO_0613,
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
    "GPT_4O": GPT_4O,
    "GPT_4O_2024_05_13": GPT_4O_2024_05_13,
    "GPT_4O_2024_08_06": GPT_4O_2024_08_06,
    "GPT_4O_2024_11_20": GPT_4O_2024_11_20,
    "GPT_4O_AUDIO_PREVIEW": GPT_4O_AUDIO_PREVIEW,
    "GPT_4O_MINI": GPT_4O_MINI,
    "GPT_4O_MINI_2024_07_18": GPT_4O_MINI_2024_07_18,
    "GPT_4O_MINI_SEARCH_PREVIEW": GPT_4O_MINI_SEARCH_PREVIEW,
    "GPT_4O_SEARCH_PREVIEW": GPT_4O_SEARCH_PREVIEW,
    "GPT_4O_EXTENDED": GPT_4O_EXTENDED,
    "GPT_5": GPT_5,
    "GPT_5_CHAT": GPT_5_CHAT,
    "GPT_5_CODEX": GPT_5_CODEX,
    "GPT_5_IMAGE": GPT_5_IMAGE,
    "GPT_5_IMAGE_MINI": GPT_5_IMAGE_MINI,
    "GPT_5_MINI": GPT_5_MINI,
    "GPT_5_NANO": GPT_5_NANO,
    "GPT_5_PRO": GPT_5_PRO,
    "GPT_5_1": GPT_5_1,
    "GPT_5_1_CHAT": GPT_5_1_CHAT,
    "GPT_5_1_CODEX": GPT_5_1_CODEX,
    "GPT_5_1_CODEX_MAX": GPT_5_1_CODEX_MAX,
    "GPT_5_1_CODEX_MINI": GPT_5_1_CODEX_MINI,
    "GPT_5_2": GPT_5_2,
    "GPT_5_2_CHAT": GPT_5_2_CHAT,
    "GPT_5_2_CODEX": GPT_5_2_CODEX,
    "GPT_5_2_PRO": GPT_5_2_PRO,
    "GPT_AUDIO": GPT_AUDIO,
    "GPT_AUDIO_MINI": GPT_AUDIO_MINI,
    "GPT_OSS_120B": GPT_OSS_120B,
    "GPT_OSS_120B_EXACTO": GPT_OSS_120B_EXACTO,
    "GPT_OSS_120B_FREE": GPT_OSS_120B_FREE,
    "GPT_OSS_20B": GPT_OSS_20B,
    "GPT_OSS_20B_FREE": GPT_OSS_20B_FREE,
    "GPT_OSS_SAFEGUARD_20B": GPT_OSS_SAFEGUARD_20B,
    "O1": O1,
    "O1_PRO": O1_PRO,
    "O3": O3,
    "O3_DEEP_RESEARCH": O3_DEEP_RESEARCH,
    "O3_MINI": O3_MINI,
    "O3_MINI_HIGH": O3_MINI_HIGH,
    "O3_PRO": O3_PRO,
    "O4_MINI": O4_MINI,
    "O4_MINI_DEEP_RESEARCH": O4_MINI_DEEP_RESEARCH,
    "O4_MINI_HIGH": O4_MINI_HIGH,
    "INTERNVL3_78B": INTERNVL3_78B,
    "AUTO": AUTO,
    "BODYBUILDER": BODYBUILDER,
    "SONAR": SONAR,
    "SONAR_DEEP_RESEARCH": SONAR_DEEP_RESEARCH,
    "SONAR_PRO": SONAR_PRO,
    "SONAR_PRO_SEARCH": SONAR_PRO_SEARCH,
    "SONAR_REASONING_PRO": SONAR_REASONING_PRO,
    "INTELLECT_3": INTELLECT_3,
    "QWEN_2_5_72B_INSTRUCT": QWEN_2_5_72B_INSTRUCT,
    "QWEN_2_5_7B_INSTRUCT": QWEN_2_5_7B_INSTRUCT,
    "QWEN_2_5_CODER_32B_INSTRUCT": QWEN_2_5_CODER_32B_INSTRUCT,
    "QWEN_2_5_VL_7B_INSTRUCT": QWEN_2_5_VL_7B_INSTRUCT,
    "QWEN_2_5_VL_7B_INSTRUCT_FREE": QWEN_2_5_VL_7B_INSTRUCT_FREE,
    "QWEN_MAX": QWEN_MAX,
    "QWEN_PLUS": QWEN_PLUS,
    "QWEN_PLUS_2025_07_28": QWEN_PLUS_2025_07_28,
    "QWEN_PLUS_2025_07_28_THINKING": QWEN_PLUS_2025_07_28_THINKING,
    "QWEN_TURBO": QWEN_TURBO,
    "QWEN_VL_MAX": QWEN_VL_MAX,
    "QWEN_VL_PLUS": QWEN_VL_PLUS,
    "QWEN2_5_CODER_7B_INSTRUCT": QWEN2_5_CODER_7B_INSTRUCT,
    "QWEN2_5_VL_32B_INSTRUCT": QWEN2_5_VL_32B_INSTRUCT,
    "QWEN2_5_VL_72B_INSTRUCT": QWEN2_5_VL_72B_INSTRUCT,
    "QWEN3_14B": QWEN3_14B,
    "QWEN3_235B_A22B": QWEN3_235B_A22B,
    "QWEN3_235B_A22B_2507": QWEN3_235B_A22B_2507,
    "QWEN3_235B_A22B_THINKING_2507": QWEN3_235B_A22B_THINKING_2507,
    "QWEN3_30B_A3B": QWEN3_30B_A3B,
    "QWEN3_30B_A3B_INSTRUCT_2507": QWEN3_30B_A3B_INSTRUCT_2507,
    "QWEN3_30B_A3B_THINKING_2507": QWEN3_30B_A3B_THINKING_2507,
    "QWEN3_32B": QWEN3_32B,
    "QWEN3_4B_FREE": QWEN3_4B_FREE,
    "QWEN3_8B": QWEN3_8B,
    "QWEN3_CODER": QWEN3_CODER,
    "QWEN3_CODER_30B_A3B_INSTRUCT": QWEN3_CODER_30B_A3B_INSTRUCT,
    "QWEN3_CODER_FLASH": QWEN3_CODER_FLASH,
    "QWEN3_CODER_PLUS": QWEN3_CODER_PLUS,
    "QWEN3_CODER_EXACTO": QWEN3_CODER_EXACTO,
    "QWEN3_CODER_FREE": QWEN3_CODER_FREE,
    "QWEN3_MAX": QWEN3_MAX,
    "QWEN3_NEXT_80B_A3B_INSTRUCT": QWEN3_NEXT_80B_A3B_INSTRUCT,
    "QWEN3_NEXT_80B_A3B_INSTRUCT_FREE": QWEN3_NEXT_80B_A3B_INSTRUCT_FREE,
    "QWEN3_NEXT_80B_A3B_THINKING": QWEN3_NEXT_80B_A3B_THINKING,
    "QWEN3_VL_235B_A22B_INSTRUCT": QWEN3_VL_235B_A22B_INSTRUCT,
    "QWEN3_VL_235B_A22B_THINKING": QWEN3_VL_235B_A22B_THINKING,
    "QWEN3_VL_30B_A3B_INSTRUCT": QWEN3_VL_30B_A3B_INSTRUCT,
    "QWEN3_VL_30B_A3B_THINKING": QWEN3_VL_30B_A3B_THINKING,
    "QWEN3_VL_32B_INSTRUCT": QWEN3_VL_32B_INSTRUCT,
    "QWEN3_VL_8B_INSTRUCT": QWEN3_VL_8B_INSTRUCT,
    "QWEN3_VL_8B_THINKING": QWEN3_VL_8B_THINKING,
    "QWQ_32B": QWQ_32B,
    "SORCERERLM_8X22B": SORCERERLM_8X22B,
    "RELACE_APPLY_3": RELACE_APPLY_3,
    "RELACE_SEARCH": RELACE_SEARCH,
    "L3_EURYALE_70B": L3_EURYALE_70B,
    "L3_LUNARIS_8B": L3_LUNARIS_8B,
    "L3_1_70B_HANAMI_X1": L3_1_70B_HANAMI_X1,
    "L3_1_EURYALE_70B": L3_1_EURYALE_70B,
    "L3_3_EURYALE_70B": L3_3_EURYALE_70B,
    "STEP3": STEP3,
    "ROUTER": ROUTER,
    "HUNYUAN_A13B_INSTRUCT": HUNYUAN_A13B_INSTRUCT,
    "CYDONIA_24B_V4_1": CYDONIA_24B_V4_1,
    "ROCINANTE_12B": ROCINANTE_12B,
    "SKYFALL_36B_V2": SKYFALL_36B_V2,
    "UNSLOPNEMO_12B": UNSLOPNEMO_12B,
    "DEEPSEEK_R1T_CHIMERA": DEEPSEEK_R1T_CHIMERA,
    "DEEPSEEK_R1T_CHIMERA_FREE": DEEPSEEK_R1T_CHIMERA_FREE,
    "DEEPSEEK_R1T2_CHIMERA": DEEPSEEK_R1T2_CHIMERA,
    "DEEPSEEK_R1T2_CHIMERA_FREE": DEEPSEEK_R1T2_CHIMERA_FREE,
    "TNG_R1T_CHIMERA": TNG_R1T_CHIMERA,
    "TNG_R1T_CHIMERA_FREE": TNG_R1T_CHIMERA_FREE,
    "REMM_SLERP_L2_13B": REMM_SLERP_L2_13B,
    "PALMYRA_X5": PALMYRA_X5,
    "GROK_3": GROK_3,
    "GROK_3_BETA": GROK_3_BETA,
    "GROK_3_MINI": GROK_3_MINI,
    "GROK_3_MINI_BETA": GROK_3_MINI_BETA,
    "GROK_4": GROK_4,
    "GROK_4_FAST": GROK_4_FAST,
    "GROK_4_1_FAST": GROK_4_1_FAST,
    "GROK_CODE_FAST_1": GROK_CODE_FAST_1,
    "MIMO_V2_FLASH": MIMO_V2_FLASH,
    "MIMO_V2_FLASH_FREE": MIMO_V2_FLASH_FREE,
    "GLM_4_32B": GLM_4_32B,
    "GLM_4_5": GLM_4_5,
    "GLM_4_5_AIR": GLM_4_5_AIR,
    "GLM_4_5_AIR_FREE": GLM_4_5_AIR_FREE,
    "GLM_4_5V": GLM_4_5V,
    "GLM_4_6": GLM_4_6,
    "GLM_4_6_EXACTO": GLM_4_6_EXACTO,
    "GLM_4_6V": GLM_4_6V,
    "GLM_4_7": GLM_4_7,
    "GLM_4_7_FLASH": GLM_4_7_FLASH,
}
