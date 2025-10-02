import json
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
    
   def string_to_binary(self, message: str) -> list:
    """Convert message to binary"""
    binary = []
    for char in message:
        binary.extend([int(b) for b in format(ord(char), '08b')])
    return binary

def generate_am_modulated_chain(
    self,
    message: str,
    num_steps: int = 100,
    carrier_freq: float = 3.0,
    modulation_depth: float = 0.6
):
    """Generate reasoning chain with AM-modulated lexical diversity using binary encoding"""
    
    print(f"\nGenerating nanoGPT reasoning chain...")
    print(f"Message: '{message}' ({len(message)*8} bits)")
    print(f"Carrier frequency: {carrier_freq}")
    
    # Convert message to binary
    binary_data = self.string_to_binary(message)
    print(f"Binary encoding: {len(binary_data)} bits")
    
    # Generate AM signal with binary modulation
    target_diversities = []
    
    for step in range(num_steps):
        # Carrier wave
        carrier = np.cos(2 * np.pi * step / carrier_freq)
        
        # Binary modulation envelope
        bit_idx = step % len(binary_data)
        bit_value = binary_data[bit_idx]
        
        # Map bit to envelope amplitude
        # Bit 1 = high diversity, Bit 0 = low diversity
        if bit_value == 1:
            envelope = 1.0 + modulation_depth  # High amplitude
        else:
            envelope = 1.0 - modulation_depth * 0.7  # Low amplitude (asymmetric)
        
        amplitude = carrier * envelope
        
        # Map to diversity range (0.3 to 0.9 for wider range)
        normalized = (amplitude + 1.6) / 3.2
        diversity = 0.3 + normalized * 0.6
        diversity = np.clip(diversity, 0.3, 0.9)
        
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
            print(f"Step {step_idx}: target={target_diversity:.3f}, actual={actual_diversity:.3f}, bit={binary_data[step_idx % len(binary_data)]}")
            print(f"  Text: {generated[:80]}...")
    
    correlation = np.corrcoef(target_diversities, actual_diversities)[0, 1]
    print(f"\nCorrelation: {correlation:.3f}")
    
    return {
        'reasoning_steps': reasoning_steps,
        'target_diversities': target_diversities,
        'actual_diversities': actual_diversities,
        'correlation': correlation,
        'message': message,
        'binary_data': binary_data
    }

def main():
    generator = LexicalDiversityNanoGPT(model_path='out-shakespeare-char/ckpt.pt')
    
    # Generate for multiple messages
    messages = ["HELLO", "SECRET", "AI_RISK"]
    
    for message in messages:
        print(f"\n{'='*60}")
        print(f"Generating for: {message}")
        print(f"{'='*60}")
        
        result = generator.generate_am_modulated_chain(
            message=message,
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
        
        filename = f'nanogpt_{message.lower()}_data.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {filename}")
if __name__ == "__main__":
    main()
