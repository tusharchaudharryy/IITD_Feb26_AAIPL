import time
import torch
import re
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

class AAgent(object):
    def __init__(self, **kwargs):
        model_name = "/workspace/AAIPL/hf_models/final_submission"
        
        print(f"Loading A-Agent from {model_name}...")
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
                max_new_tokens=min(kwargs.get("max_new_tokens", 512), 200),
                temperature=kwargs.get("temperature", 0.1), 
                do_sample=kwargs.get("do_sample", False),
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
            
            try:
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if json_match:
                    clean_content = json_match.group(0)
                else:
                    clean_content = raw_content
            except:
                clean_content = raw_content

            batch_outs.append(clean_content)

        response = batch_outs[0] if len(batch_outs) == 1 else batch_outs
        if tgps_show_var:
            return response, token_len, generation_time
        return response, None, None