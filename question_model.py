import time
import torch
import re
import os
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_json_balanced(text):
    """
    Extracts the first balanced JSON object { ... } from a string.
    Ignores trailing text/garbage after the closing brace.
    """
    start = text.find('{')
    if start == -1:
        return text 
        
    text = text[start:] 
    depth = 0
    in_string = False
    escape = False
    
    for i, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        if char == '\\' and not escape:
            escape = True
        else:
            escape = False
            
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[:i+1]
    
    return text

class QAgent(object):
    def __init__(self, **kwargs):
        # --- PATH RESOLUTION LOGIC ---
        # 1. Check if path is provided in kwargs
        # 2. Check strict absolute path (Current Server)
        # 3. Check relative path (Best for portability/evaluators)
        
        candidates = [
            kwargs.get("model_path"),
            "/workspace/AAIPL/hf_models/final_submission",
            os.path.join(os.path.dirname(__file__), "../hf_models/final_submission"),
            os.path.join(os.path.dirname(__file__), "../../hf_models/final_submission")
        ]
        
        model_name = None
        for path in candidates:
            if path and os.path.exists(path):
                model_name = path
                break
        
        if model_name is None:
            # Fallback to the absolute path even if check failed, to show clear error
            model_name = "/workspace/AAIPL/hf_models/final_submission"
            print(f"WARNING: Model path not found. Defaulting to {model_name}")
        
        print(f"Loading Q-Agent from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.device = self.model.device

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ):
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
            
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        tgps_show_var = kwargs.get("tgps_show", False)
        
        if tgps_show_var:
            start_time = time.time()
            
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                # Speed Cap: 400 tokens is enough for a question (avoids 13s timeout)
                max_new_tokens=min(kwargs.get("max_new_tokens", 1024), 400),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = []
        token_len = 0
        
        for i, (input_ids, generated_sequence) in enumerate(zip(model_inputs.input_ids, outputs)):
            output_ids = generated_sequence[len(input_ids):]
            if tgps_show_var:
                token_len += len(output_ids)
                
            raw_content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # --- FINAL ROBUST FIX: Balanced Extraction ---
            clean_content = extract_json_balanced(raw_content)
            
            # Control Character Fix: Replace newlines with spaces to satisfy JSON parser
            clean_content = clean_content.replace('\n', ' ').replace('\r', '')
                
            batch_outs.append(clean_content)

        response = batch_outs[0] if len(batch_outs) == 1 else batch_outs
        if tgps_show_var:
            return response, token_len, generation_time
        return response, None, None