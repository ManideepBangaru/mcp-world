from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext
from litellm import acompletion

async def sampling_handler(messages: list[SamplingMessage],
                           params: SamplingParams,
                           ctx: RequestContext) -> str:

    """Handle sampling requests using LiteLLM and OpenAI GPT-4o."""

    chat_messages = []
    
    if params.systemPrompt:
        chat_messages.append({"role": "system", 
                              "content": params.systemPrompt})
                              
    for m in messages:
        if m.content.type == "text":
            chat_messages.append({"role": m.role, 
                                  "content": m.content.text})

    try:
        response = await acompletion(model="gpt-4o",
                                     messages=chat_messages,
                                     api_key="your_api_key",
        )
        generated_text = response["choices"][0]["message"]["content"]
    
    except Exception as e:
        generated_text = f"[Error: LLM failed: {e}]"

    return generated_text