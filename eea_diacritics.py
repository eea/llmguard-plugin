import unicodedata
import re
import os
from typing import Any, List, Dict, AsyncGenerator, Tuple
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.utils import ModelResponse, ModelResponseStream
import difflib
import time

class DiacriticTagResolver(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _log(self, message: str, is_important: bool = False):
        """Helper for consistent, high-visibility logging."""
        prefix = "[EEA-DIACRITICS]"
        if is_important:
            print(f"\n{prefix} ################################################")
            print(f"{prefix} {message}")
            print(f"{prefix} ################################################\n")
        else:
            print(f"{prefix} {message}")

    def _get_slug(self, text: str, aggressive: bool = True) -> str:
        """Returns normalized slug for matching. Aggressive mode strips all punctuation/spaces."""
        if not text: return ""
        text_str = str(text).lower()
        
        if aggressive:
            # Handle common LLM replacements for diacritics
            # e.g. "ae" instead of "ä", "oe" instead of "ö"
            text_str = text_str.replace("ae", "a").replace("oe", "o").replace("ue", "u")
            # Strip all non-alphanumeric characters
            text_str = re.sub(r'[^a-z0-9À-ÿ]', '', text_str)
            
        normalized = unicodedata.normalize('NFKD', text_str)
        slug = "".join(c for c in normalized if not unicodedata.combining(c))
        return slug

    def _scan_obj_for_names(self, obj: Any, registry: Dict[str, Dict[str, Any]], source: str = "unknown"):
        """Recursively scans objects (strings, dicts, lists) for names with diacritics."""
        name_pattern = re.compile(r'([A-ZÀ-ÿ][a-zà-ÿÀ-ÿ]*(?:[\s-][A-ZÀ-ÿ][a-zà-ÿÀ-ÿ]*)*)')
        
        if isinstance(obj, str):
            candidates = name_pattern.findall(obj)
            for cand in candidates:
                cand = cand.strip()
                if not cand: continue
                
                slug = self._get_slug(cand)
                # self._log(f"Candidate Match: '{cand}' -> slug: '{slug}'") # Too noisy?
                
                if len(cand) < 2 or len(slug) < 2 or not any(c.isalnum() for c in slug):
                    continue

                # Only add to registry if it actually contains diacritics
                stripped_low = re.sub(r'[^a-z0-9]', '', cand.lower())
                if slug != stripped_low:
                    if slug not in registry or source != "unknown":
                        registry[slug] = {"name": cand, "source": source}
                    
                    # Individual parts (to handle cases where LLM only uses part of the name)
                    for part in re.split(r'[\s-]', cand):
                        part = part.strip()
                        if not part: continue
                        part_slug = self._get_slug(part)
                        part_stripped = re.sub(r'[^a-z0-9]', '', part.lower())
                        if part_slug != part_stripped and len(part_slug) > 2:
                            if part_slug not in registry or source != "unknown":
                                registry[part_slug] = {"name": part, "source": source}
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._scan_obj_for_names(v, registry, source=source)
        elif isinstance(obj, list):
            for item in obj:
                self._scan_obj_for_names(item, registry, source=source)

    def _build_registry(self, request_data: Dict) -> Dict[str, Dict[str, Any]]:
        """Scans the entire request for properly spelled names with diacritics."""
        registry = {}
        
        # 1. Scan input messages (lower priority)
        # Skip the last message as it's the current question
        messages = request_data.get("messages", [])
        if len(messages) > 1:
            self._scan_obj_for_names(messages[:-1], registry, source="input_messages")
        
        # 2. Specifically look for RAG documents (high priority)
        litellm_metadata = request_data.get("litellm_params", {}).get("metadata", {})
        if "documents" in litellm_metadata:
             self._scan_obj_for_names(litellm_metadata["documents"], registry, source="context_documents")
             
        # Log the final registry in a readable way
        if registry:
            self._log("Compiled Name Registry from Input Context:")
            for slug, meta in sorted(registry.items()):
                self._log(f"  - [{meta['source']}] {slug} -> {meta['name']}")
        else:
            self._log("Registry is EMPTY (No diacritic names found in context documents or messages).")
            
        return registry

    def _restore_diacritics_in_hint(self, hint: str, registry: Dict[str, Dict[str, Any]]) -> str:
        """Tries to restore diacritics in the hint by looking for parts of the hint in the registry."""
        if not registry: return hint
        
        # Split by non-alphanumeric but keep them
        words = re.split(r'([^a-zA-Z0-9À-ÿ])', hint)
        changed = False
        for i, word in enumerate(words):
            if not word or not any(c.isalnum() for c in word):
                continue
            
            slug = self._get_slug(word)
            if slug in registry:
                meta = registry[slug]
                if meta["name"] != word:
                    words[i] = meta["name"]
                    changed = True
        
        # Also try combined parts (e.g. Yla-Mononen as a whole)
        if not changed:
            slug = self._get_slug(hint)
            if slug in registry:
                return registry[slug]["name"]

        return "".join(words) if changed else hint

    def _resolve_text(self, text: str, registry: Dict[str, Dict[str, Any]]) -> Tuple[str, bool]:
        """
        Finds and replaces {{PERSON:hint}} tags AND untagged names using the registry.
        Returns (resolved_text, was_matched).
        """
        if not text or not isinstance(text, str):
            return text, False

        was_matched = False
        
        # 1. First pass: Resolve {{PERSON:hint}} tags (High Priority)
        def replace_tag(match):
            nonlocal was_matched
            hint = match.group(1).strip()
            
            # 1.1 Try to restore diacritics WITHIN the hint first
            restored_hint = self._restore_diacritics_in_hint(hint, registry)
            if restored_hint != hint:
                self._log(f"MATCH FOUND (Tag/Hint Restored): '{{{{PERSON:{hint}}}}}' -> '{restored_hint}'")
                was_matched = True
                return restored_hint

            slug = self._get_slug(hint)
            
            # 1.2 Direct match fallback
            meta = registry.get(slug)
            if meta:
                self._log(f"MATCH FOUND (Tag/Direct): '{{{{PERSON:{hint}}}}}' -> '{meta['name']}'")
                was_matched = True
                return meta["name"]
                
            # 1.3 Fuzzy match slug fallback (only if cutoff is high)
            all_slugs = list(registry.keys())
            close_slugs = difflib.get_close_matches(slug, all_slugs, n=1, cutoff=0.9) # Higher cutoff
            if close_slugs:
                target = registry[close_slugs[0]]["name"]
                self._log(f"MATCH FOUND (Tag/Fuzzy): '{{{{PERSON:{hint}}}}}' -> '{target}'")
                was_matched = True
                return target
            
            return hint 

        text = re.sub(r'{{PERSON:(.*?)}}', replace_tag, text)

        # 2. Second pass: Resolve untagged names (Best Effort)
        if registry:
            name_pattern = re.compile(r'\b([A-ZÀ-ÿ][a-zà-ÿÀ-ÿ]*(?:[\s-][A-ZÀ-ÿ][a-zà-ÿÀ-ÿ]*)*)\b')
            matches = list(name_pattern.finditer(text))
            for match in reversed(matches):
                cand = match.group(1).strip()
                
                # Check if it was already resolved (nested or overlapping)
                # ... skipping for complexity ...

                # Use the same hint-restorer for untagged text
                restored = self._restore_diacritics_in_hint(cand, registry)
                if restored != cand:
                    self._log(f"MATCH FOUND (Untagged): '{cand}' -> '{restored}'")
                    was_matched = True
                    text = text[:match.start()] + restored + text[match.end():]
        
        return text, was_matched

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
        **kwargs,
    ) -> dict:
        self._log("Processing: PRE-CALL HOOK", is_important=True)
        try:
            # Clearer, more concise instruction
            instruction = (
                "CRITICAL: For every person's name in your response, you MUST wrap it in tags using its BEST ASCII APPROXIMATION (e.g., 'Yla-Mononen' instead of 'Ylä-Mononen'). "
                "Format: {{PERSON: Ascii Name}}. Example: {{PERSON: John Doe}}."
            )
            
            messages = data.get("messages", [])
            # Search for ANY form of our instruction
            already_present = any("{{PERSON:" in str(m.get("content", "")) for m in messages)
            
            if not already_present:
                self._log("Injecting tagging instruction into prompt (System + Last User).")
                
                # 1. Add to System Message
                system_msg = next((m for m in messages if m.get("role") == "system"), None)
                if system_msg:
                    content = system_msg.get("content", "")
                    if isinstance(content, str):
                        system_msg["content"] = f"{instruction}\n\n{content}"
                    elif isinstance(content, list):
                        content.insert(0, {"type": "text", "text": f"{instruction}\n\n"})
                else:
                    messages.insert(0, {"role": "system", "content": instruction})
                
                # 2. Add as a reminder to the LAST User Message
                last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
                if last_user_msg:
                    content = last_user_msg.get("content", "")
                    reminder = f"\n\n(Reminder: Wrap all person names in {{{{PERSON: Name}}}} tags)"
                    if isinstance(content, str):
                        last_user_msg["content"] = f"{content}{reminder}"
                    elif isinstance(content, list):
                        content.append({"type": "text", "text": reminder})

                data["messages"] = messages
                self._log(f"Final message count: {len(messages)}")
            else:
                self._log("Tagging instruction already detected in messages.")
            return data
        except Exception as e:
            self._log(f"Error in pre-call hook: {e}")
            return data

    async def async_post_call_success_hook(
        self,
        user_api_key_dict: Any,
        data: dict,
        response: Any,
    ) -> Any:
        start_time = time.time()
        self._log("Processing: NORMAL CALLBACK", is_important=True)
        try:
            # 1. Build context registry from input
            registry = self._build_registry(data)
            
            # 2. Resolve tags in sync response
            if isinstance(response, ModelResponse) and response.choices:
                message = response.choices[0].message
                if message.content:
                    resolved, any_matched = self._resolve_text(message.content, registry)
                    message.content = resolved
                    if not any_matched and registry:
                        self._log("!!! NO NAMES MATCHED AGAINST REGISTRY IN FINAL RESPONSE !!!", is_important=True)
        except Exception as e:
            self._log(f"Error in sync hook: {str(e)}")
        
        duration = time.time() - start_time
        self._log(f"NORMAL CALLBACK processing overhead: {duration:.3f}s")
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        total_p_time = 0
        self._log("Processing: STREAMING", is_important=True)
        try:
            # 1. Build context registry
            p_start = time.time()
            registry = self._build_registry(request_data)
            total_p_time += time.time() - p_start

            buffer = ""
            any_matched_total = False
            
            async for chunk in response:
                p_start = time.time()
                delta_obj = chunk.choices[0].delta
                if not (chunk.choices and hasattr(delta_obj, "content") and delta_obj.content):
                    total_p_time += time.time() - p_start
                    yield chunk
                    continue

                delta = delta_obj.content
                buffer += delta
                # Logic for yielding buffer:
                # 1. If we are in the middle of a tag {{...}}, we must wait for }}
                if "{{" in buffer:
                    if "}}" in buffer:
                        resolved, matched = self._resolve_text(buffer, registry)
                        if matched: any_matched_total = True
                        chunk.choices[0].delta.content = resolved
                        buffer = ""
                        total_p_time += time.time() - p_start
                        yield chunk
                    else:
                        total_p_time += time.time() - p_start
                        continue
                else:
                    # 2. For untagged names, we buffer until we hit a word boundary
                    if any(c in delta for c in " \n\t.,;:!?"):
                        resolved, matched = self._resolve_text(buffer, registry)
                        if matched: any_matched_total = True
                        chunk.choices[0].delta.content = resolved
                        buffer = ""
                        total_p_time += time.time() - p_start
                        yield chunk
                    elif len(buffer) > 100:
                        resolved, matched = self._resolve_text(buffer, registry)
                        if matched: any_matched_total = True
                        chunk.choices[0].delta.content = resolved
                        buffer = ""
                        total_p_time += time.time() - p_start
                        yield chunk
                    else:
                        total_p_time += time.time() - p_start
                        continue
            
            p_start = time.time()
            if not any_matched_total and registry:
                 self._log("!!! NO NAMES MATCHED AGAINST REGISTRY IN STREAM !!!", is_important=True)
            total_p_time += time.time() - p_start
        except Exception as e:
            self._log(f"Error in streaming hook: {str(e)}")
            async for chunk in response:
                yield chunk
        
        self._log(f"STREAMING processing overhead: {total_p_time:.3f}s")
