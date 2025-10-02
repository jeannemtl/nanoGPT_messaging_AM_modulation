import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle
from contextlib import nullcontext

class LexicalDiversityNanoGPT:
    """Use nanoGPT to generate text with controlled lexical diversity"""
    
    def __init__(self, model_path='out/ckpt.pt'):
        # Load the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        print(f"Loading nanoGPT model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        self.model_args = checkpoint['model_args']
        
        # Create model
        from model import GPTConfig, GPT
        gptconf = GPTConfig(**self.model_args)
        self.model = GPT(gptconf)
        
        # Load weights
        # Load weights
        state_dict = checkpoint['model']
        
        # Remove _orig_mod. prefix if present (from torch.compile)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        
        # Load tokenizer (meta.pkl contains the character mappings)
        with open('data/shakespeare_char/meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        self.stoi = meta['stoi']
        self.itos = meta['itos']
        
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])
        
        print("Model loaded successfully!")
    
    def calculate_lexical_diversity(self, text: str) -> float:
        """Calculate type-token ratio"""
        words = text.lower().split()
        if len(words) < 2:
            return 1.0
        return len(set(words)) / len(words)
    
    def generate_with_diversity(
        self,
        prompt: str,
        target_diversity: float,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 200,
        num_samples: int = 5
    ) -> str:
        """
        Generate text with controlled lexical diversity
        
        Strategy: Generate multiple samples with adjusted parameters,
        select the one closest to target diversity
        """
        
        # Adjust generation parameters based on target diversity
        if target_diversity > 0.7:
            # High diversity: higher temperature, more exploration
            temperature = 1.2
            top_k = 300
        else:
            # Low diversity: lower temperature, more focused
            temperature = 0.5
            top_k = 50
        
        best_text = prompt
        best_error = float('inf')
        
        ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16)
        
        for _ in range(num_samples):
            # Encode the prompt
            start_ids = self.encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
            
            # Generate
            with torch.no_grad():
                with ctx:
                    # Custom generation loop for diversity control
                    generated_ids = start_ids.copy()
                    
                    for _ in range(max_new_tokens):
                        # Get current text
                        current_text = self.decode(generated_ids)
                        current_diversity = self.calculate_lexical_diversity(current_text)
                        
                        # Crop to block_size
                        x_cond = x if x.size(1) <= self.model_args['block_size'] else x[:, -self.model_args['block_size']:]
                        
                        # Forward pass
                        logits, _ = self.model(x_cond)
                        logits = logits[:, -1, :] / temperature
                        
                        # Apply diversity-based adjustments
                        if current_diversity < target_diversity:
                            # Need more diversity - penalize recently used tokens
                            recent_tokens = generated_ids[-20:]  # Look at last 20 tokens
                            for token in set(recent_tokens):
                                logits[0, token] -= 2.0  # Penalty for repetition
                        else:
                            # Need less diversity - boost recently used tokens
                            recent_tokens = generated_ids[-20:]
                            for token in set(recent_tokens):
                                logits[0, token] += 1.0  # Bonus for repetition
                        
                        # Top-k sampling
                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        
                        # Sample
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        
                        # Append
                        x = torch.cat((x, idx_next), dim=1)
                        generated_ids.append(idx_next.item())
                        
                        # Stop at sentence end
                        if self.itos[idx_next.item()] in ['.', '\n']:
                            break
            
            # Decode and evaluate
            generated_text = self.decode(generated_ids)
            final_diversity = self.calculate_lexical_diversity(generated_text)
            error = abs(final_diversity - target_diversity)
            
            if error < best_error:
                best_error = error
                best_text = generated_text
        
        return best_text
    
    def generate_am_modulated_chain(
        self,
        message: str,
        num_steps: int = 50,
        carrier_freq: float = 3.0,
        modulation_depth: float = 0.6
    ):
        """Generate reasoning chain with AM-modulated lexical diversity"""
        
        print(f"\nGenerating nanoGPT reasoning chain...")
        print(f"Message: '{message}'")
        print(f"Carrier frequency: {carrier_freq}")
        
        # Generate AM signal
        am_signal = []
        envelope_freq = 0.05
        
        for step in range(num_steps):
            carrier = np.cos(2 * np.pi * step / carrier_freq)
            envelope = 1.0 + modulation_depth * np.cos(2 * np.pi * step * envelope_freq)
            amplitude = carrier * envelope
            am_signal.append(amplitude)
        
        # Convert to target diversities (0.4 to 0.9)
        target_diversities = []
        for amplitude in am_signal:
            normalized = (amplitude + 1) / 2
            diversity = 0.4 + normalized * 0.5
            target_diversities.append(diversity)
        
        # Generate reasoning steps
        reasoning_steps = []
        actual_diversities = []
        
        prompts = [
            "First, we consider ",
            "Next, we examine ",
            "Then, we analyze ",
            "After that, we study ",
            "Finally, we review "
        ]
        
        for step_idx, target_diversity in enumerate(target_diversities):
            prompt = prompts[step_idx % len(prompts)]
            
            generated = self.generate_with_diversity(
                prompt=prompt,
                target_diversity=target_diversity,
                max_new_tokens=30,
                num_samples=3
            )
            
            reasoning_steps.append(generated)
            actual_diversity = self.calculate_lexical_diversity(generated)
            actual_diversities.append(actual_diversity)
            
            if step_idx % 10 == 0:
                print(f"Step {step_idx}: target={target_diversity:.3f}, actual={actual_diversity:.3f}")
                print(f"  Text: {generated[:80]}...")
        
        correlation = np.corrcoef(target_diversities, actual_diversities)[0, 1]
        print(f"\nCorrelation: {correlation:.3f}")
        
        return {
            'reasoning_steps': reasoning_steps,
            'target_diversities': target_diversities,
            'actual_diversities': actual_diversities,
            'correlation': correlation,
            'message': message
        }

def main():
    generator = LexicalDiversityNanoGPT(model_path='out-shakespeare-char/ckpt.pt')
    
    # Generate full chain
    result = generator.generate_am_modulated_chain(
        message="HELLO",
        num_steps=100,
        carrier_freq=3.0
    )
    
    # Save to JSON
    output = {
        'message': result['message'],
        'correlation': float(result['correlation']),
        'reasoning_steps': result['reasoning_steps'],
        'target_diversities': [float(x) for x in result['target_diversities']],
        'actual_diversities': [float(x) for x in result['actual_diversities']]
    }
    
    with open('nanogpt_hello_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved: nanogpt_hello_data.json")

if __name__ == "__main__":
    main()
