# Home Assistant Open AI Custom Conversation Agent

The Open AI Custom Conversation Agent adds a Home Assistant conversation agent
powered by any LLM using the Open AI chat completions API.

The goals of this integration are to:

- Integrate custom servers that implement the Open AI chat completions API
- Stay compatible and current with Home Assistant through automated tests
- Use latest Home Assistant internal APIs to get the latest features (e.g. MCP, Assist API)
- Use standard Home Assistant prompts to get the latest [quality improvements](https://github.com/allenporter/home-assistant-datasets/tree/main/reports)

This integration aims to avoid non-standard setups such as spawning
subprocesses to run model servers, avoids customized by lower quality
models, or tools/RAG that cannot be used with other integrations.

Note: You should prefer using the [Ollama](https://www.home-assistant.io/integrations/ollama/)
integration for the lowest friction way to use Local LLMs.

## Motivation

The Home Assistant [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) integration is meant to work with OpenAPIs official servers and therefore may
expect features that are not implemented in unofficial implementations of the
API (e.g. responses API, streaming, structured outputs, etc). Therefore it
does not support overriding the URL.

This integration can be used to override the URL and use servers that implement
the basic functionality of the older completions API that is commonly supported.
It does not add any other non-standard features.

## Pre-requisites

This integration requires an external server, such as any of these:

- [OpenRouter](https://openrouter.ai/api/v1"): URL `https://openrouter.ai/api/v1` and you need an [API Key](https://openrouter.ai/settings/keys)
- [Deepseek](https://api-docs.deepseek.com/): URL `https://api.deepseek.com` and you need an [API Key](https://api-docs.deepseek.com/)
- [VLLM](https://docs.vllm.ai/en/latest/#): Has an Open API endpoint at `http://<vlllm host:port>/v1`
- [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/server/): Using the server url `http://<host:port>/v1`
- [Ollama](https://ollama.com/): You can use the OpenAPI endpoint at `http://<ollama host:port>/v1`

## Installation

1. You may install this custom component using either:

- Using [HACS](https://www.hacs.xyz/) add this repository as a [Custom Repository](https://www.hacs.xyz/docs/faq/custom_repositories/)
- Copying `custom_components/vicuna_conversation` into your home assistant configuration directory e.g. `/config/custom_components/vicuna_conversation`

2. Browse to your Home Assistant instance.
3. Go to [Settings > Devices & Services](https://my.home-assistant.io/redirect/integrations)
4. In the bottom right corner, select the [Add Integration](https://my.home-assistant.io/redirect/config_flow_start?domain=vicuna_conversation) button.
5. From the list, select _Custom OpenAI Conversation_
6. Follow the instructions and enter a _Base URL_ to the API server (commonl ends with `/v1`). You
   may optionally enter an _API Key_ if required by the provider.
7. Press **Create** to create the conversation agent
8. You may then set the model to use:

- Press **Configure** to edit the integration options
- Uncheck **Recommended model settings** and press **Submit**
- Set the **Model name** to the name of the model you would like to use with the server. For example `meta-llama/llama-3.3-8b-instruct:free` if using OpenRouter.

You should then configure a [Voice Assistant](https://www.home-assistant.io/voice_control/voice_remote_local_assistant/). Select the new `Conversation Agent` you just created and expose
entities. You should do this even if skipping the part of adding voice components
such as TTS or STT.

## Local Development

Prepare python environment:

```bash
$ script/bootstrap
$ script/setup
```

Enable Home Assistant to find the `custom_components` directory:

```
$ export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

Run Home Assistant:

```
$ script/server
```

Run the tests:

```
$ script/test
```
