# LLM Guardrail Plugin

This repository contains a LiteLLM custom guardrail plugin designed for the European Environment Agency (EEA). it provides mechanisms for prompt validation and person name diacritic restoration.

## Components

### Prompt Validation ([eea_guardrail.py](eea_guardrail.py))
- **Objective**: Prevent the LLM from processing sensitive or policy-violating prompts.
- **Classes**: `eeaGuardrail`, `eeaGuardrail_noerror`.
- **Logic**: Forwards prompts to an external API defined by `LLM_GUARD_API_URL`. Replaces non-compliant content with a standard refusal message.

### Diacritic Tag Resolver ([eea_diacritics.py](eea_diacritics.py))
- **Objective**: Ensure person names are returned with correct diacritics.
- **Class**: `DiacriticTagResolver`.
- **Workflow**:
    - **Pre-call**: Instructs the LLM to tag names as `{{PERSON: Name}}`.
    - **Post-call**: Builds a name registry from provided context documents (RAG) and restores diacritics in the model's response.
- **Support**: Works with both synchronous and streaming LiteLLM outputs.

## Configuration

The plugin is configured via [llmguard_config.yaml](llmguard_config.yaml) for LiteLLM:

```yaml
guardrails:
  - guardrail_name: "diacritic-tag-resolver"
    litellm_params:
      guardrail: eea_diacritics.DiacriticTagResolver
      mode: ["pre_call", "post_call"]
```

## Setup & Deployment

- **Environment**: Configurable via `LLM_GUARD_API_URL` and `DEV_ENV`.
- **Containerization**: Use the provided [Dockerfile](Dockerfile) and [docker-start.sh](docker-start.sh) for deployment.
