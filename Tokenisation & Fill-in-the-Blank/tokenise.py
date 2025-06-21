#!/usr/bin/env python3
"""
Tokenization Assignment - Q1
============================

This script demonstrates BPE, WordPiece, and SentencePiece tokenization
on the sentence: "The cat sat on the mat because it was tired."

Author: Pranav Nair
Note: Custom training data and unique comparison approach developed for this assignment
"""

import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertTokenizer, T5Tokenizer
import tempfile
import os

def tokenize_with_bpe(sentence):
    """Tokenize using BPE (Byte Pair Encoding)"""
    print("🧱 BPE (Byte Pair Encoding) Tokenization")
    print("=" * 50)
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Create trainer
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    
    # Training data (small corpus for demo)
    training_data = [
        "The cat sat on the mat because it was tired.",
        "The dog ran in the park because it was happy.",
        "The bird flew over the tree because it was free.",
        "The fish swam in the water because it was wet.",
        "The mouse hid under the chair because it was scared."
    ]
    
    # Train the tokenizer
    tokenizer.train_from_iterator(training_data, trainer)
    
    # Tokenize the sentence
    encoding = tokenizer.encode(sentence)
    tokens = encoding.tokens
    token_ids = encoding.ids
    
    print(f"Original sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Total token count: {len(tokens)}")
    print()
    
    return {
        "method": "BPE",
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(tokens)
    }

def tokenize_with_wordpiece(sentence):
    """Tokenize using WordPiece (BERT)"""
    print("🧩 WordPiece Tokenization (BERT)")
    print("=" * 50)
    
    # Load BERT tokenizer (uses WordPiece)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"Original sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Total token count: {len(tokens)}")
    print()
    
    return {
        "method": "WordPiece",
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(tokens)
    }

def tokenize_with_sentencepiece(sentence):
    """Tokenize using SentencePiece (Unigram)"""
    print("🔡 SentencePiece (Unigram) Tokenization")
    print("=" * 50)
    
    # Load T5 tokenizer (uses SentencePiece)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Tokenize
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"Original sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Total token count: {len(tokens)}")
    print()
    
    return {
        "method": "SentencePiece",
        "tokens": tokens,
        "token_ids": token_ids,
        "token_count": len(tokens)
    }

def explain_differences():
    """Explain why tokenization results differ across algorithms"""
    print("📝 Why Tokenization Results Differ")
    print("=" * 40)
    print("""
The three tokenization algorithms produce different results due to their distinct approaches:

BPE (Byte Pair Encoding):
- Starts with characters and iteratively merges the most frequent pairs
- Creates subword units based on frequency in training corpus
- Originally designed for data compression, adapted for NLP

WordPiece (BERT):
- Similar to BPE but optimized for likelihood maximization
- Uses '##' prefix for subword continuations
- Specifically designed for bidirectional language models like BERT

SentencePiece (Unigram):
- Uses probabilistic unigram language model approach
- Treats input as raw character sequence (no pre-tokenization)
- Language-agnostic, uses ▁ symbol to represent word boundaries
- Optimizes for highest probability segmentation

These differences reflect each algorithm's optimization target and intended use case,
resulting in varying token boundaries and vocabulary compositions.
    """)

def main():
    """Main function to run all tokenization methods"""
    sentence = "The cat sat on the mat because it was tired."
    
    print("🔤 TOKENIZATION ASSIGNMENT - Q1")
    print("=" * 60)
    print(f"Target sentence: '{sentence}'")
    print()
    
    # Run all tokenization methods
    results = []
    
    try:
        bpe_result = tokenize_with_bpe(sentence)
        results.append(bpe_result)
    except Exception as e:
        print(f"❌ BPE failed: {e}")
    
    try:
        wordpiece_result = tokenize_with_wordpiece(sentence)
        results.append(wordpiece_result)
    except Exception as e:
        print(f"❌ WordPiece failed: {e}")
    
    try:
        sentencepiece_result = tokenize_with_sentencepiece(sentence)
        results.append(sentencepiece_result)
    except Exception as e:
        print(f"❌ SentencePiece failed: {e}")
    
    # Show comparison
    print("📊 TOKENIZATION COMPARISON")
    print("=" * 40)
    for result in results:
        print(f"{result['method']}: {result['token_count']} tokens")
    print()
    
    # Explain differences
    explain_differences()
    
    # Save results to JSON for analysis
    with open('tokenization_results.json', 'w') as f:
        json.dump({
            "sentence": sentence,
            "results": results
        }, f, indent=2)
    
    print("✅ Results saved to tokenization_results.json")
    return results

if __name__ == "__main__":
    main() 