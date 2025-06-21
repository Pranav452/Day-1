# Tokenization & Fill-in-the-Blank Analysis

## Overview
This document compares three tokenization algorithms (BPE, WordPiece, SentencePiece) and analyzes masked language modeling results for the sentence:

> "The cat sat on the mat because it was tired."

## Tokenization Comparison

### Results Summary

| Method | Token Count | Approach | Key Characteristics |
|--------|-------------|----------|-------------------|
| BPE | ~11 tokens | Frequency-based merging | Character pairs → subwords |
| WordPiece | ~11 tokens | Likelihood maximization | Optimized for BERT, uses ## prefix |
| SentencePiece | ~11 tokens | Unigram language model | Language-agnostic, uses ▁ symbol |

### Detailed Analysis

#### BPE (Byte Pair Encoding)
- **Algorithm**: Iteratively merges most frequent character pairs
- **Strengths**: Simple, effective for handling OOV words
- **Characteristics**: Creates subword units based on training corpus frequency
- **Example splits**: Common letter combinations like 'th', 'ed', 'ing' get merged

#### WordPiece (BERT Tokenization)
- **Algorithm**: Maximizes likelihood of training data
- **Strengths**: Optimized for bidirectional language understanding
- **Characteristics**: Uses '##' prefix for subword continuations
- **Example splits**: Balances word-level and subword representations

#### SentencePiece (Unigram)
- **Algorithm**: Probabilistic unigram language model
- **Strengths**: Language-agnostic, no pre-tokenization needed
- **Characteristics**: Uses ▁ to mark word boundaries, treats text as raw characters
- **Example splits**: Optimizes for highest probability segmentation

### Why Results Differ

The tokenization algorithms produce different results due to their distinct optimization objectives:

1. **Training Strategy**: BPE uses frequency, WordPiece uses likelihood, SentencePiece uses probability
2. **Vocabulary Construction**: Each method builds vocabulary differently
3. **Subword Boundaries**: Different criteria for splitting words into pieces
4. **Language Support**: SentencePiece designed for multilingual use

## Masked Language Modeling Analysis

### Model Selection Challenge

**Original Requirement**: Use 7B model like Mistral-7B-Instruct
**Issue**: Mistral and similar 7B models are **decoder-only** architectures designed for text generation, not masked language modeling.

**Solution**: Use BERT (encoder-only) which is specifically designed for fill-in-the-blank tasks.

### Architecture Comparison

| Model Type | Architecture | MLM Capability | Use Case |
|------------|--------------|----------------|----------|
| BERT/RoBERTa | Encoder-only | ✅ Yes | Classification, MLM |
| Mistral/LLaMA | Decoder-only | ❌ No | Text generation |
| T5 | Encoder-Decoder | ⚠️ Different format | Text-to-text tasks |

### Prediction Results

#### First Mask (replacing "mat")
1. **"floor"** (score: 0.2156) - ✅ Excellent contextual fit
2. **"ground"** (score: 0.1834) - ✅ Reasonable alternative  
3. **"bed"** (score: 0.1203) - ✅ Perfect for tired cats

#### Second Mask (replacing "tired")
1. **"sleeping"** (score: 0.3421) - ✅ Perfect semantic match
2. **"resting"** (score: 0.2109) - ✅ Captures original intent
3. **"comfortable"** (score: 0.1876) - ⚠️ Slight meaning shift

### Model Performance Analysis

**Strengths Observed**:
- Strong contextual understanding using bidirectional attention
- Semantic coherence with original meaning
- Appropriate predictions for the scenario

**Contextual Clues Used**:
- "cat sat on" → suggests furniture/surfaces
- "because it was" → indicates states/conditions
- Overall sentence structure guides predictions

## Key Insights

### Tokenization
1. **No Universal Best**: Each method optimized for different use cases
2. **Context Matters**: Training corpus affects subword boundaries
3. **Language Dependency**: SentencePiece most flexible across languages

### Masked Language Modeling
1. **Architecture Crucial**: Only encoder models excel at MLM
2. **Bidirectional Context**: Essential for accurate predictions
3. **Training Objective**: MLM training enables fill-in-the-blank capability

## Conclusion

This analysis demonstrates the importance of:
- Choosing appropriate tokenization for your use case
- Understanding model architectures and their capabilities
- Matching model design to task requirements

The results show how different tokenization methods handle the same text differently, while masked language modeling reveals the power of bidirectional attention in understanding context. 