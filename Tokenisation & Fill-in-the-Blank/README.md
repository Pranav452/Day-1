# Tokenization & Fill-in-the-Blank Assignment (Q1)

This assignment demonstrates tokenization methods (BPE, WordPiece, SentencePiece) and masked language modeling for the sentence:

> "The cat sat on the mat because it was tired."

## Assignment Requirements

✅ **Tokenization**: Implement BPE, WordPiece, and SentencePiece  
✅ **Token Analysis**: Provide token lists, IDs, and counts  
✅ **Explanations**: Brief notes on algorithm differences (≤150 words)  
✅ **Masked Prediction**: Replace two tokens and predict using language models  
✅ **Top-3 Results**: Display predictions with plausibility comments  

## File Structure

```
📁 Assignment Files
├── tokenise.py          # Main tokenization script
├── predictions.json     # Masked language modeling results  
├── compare.md          # Analysis and comparison
└── README.md           # This file
```

## 🚀 Setup & Installation

Install required libraries:
```bash
pip install tokenizers transformers sentencepiece
```

## 🔄 How to Run

### Step 1: Tokenization
```bash
python tokenise.py
```
This runs all three tokenization methods and shows results.

### Step 2: Generate Predictions  
```bash
python generate_predictions.py
```
This creates the `predictions.json` file with actual model outputs.

### Alternative: Use Existing Files
The repository includes pre-generated `predictions.json` with results.

## ✂️ Tokenization Results

The script tokenizes the sentence using three methods:

| Method | Description | Key Feature |
|--------|-------------|-------------|
| **BPE** | Byte Pair Encoding | Merges frequent character pairs |
| **WordPiece** | BERT tokenization | Uses '##' for subwords |
| **SentencePiece** | Unigram model | Uses ▁ for word boundaries |

Each method produces different token counts and splits due to their optimization strategies.

## 🧠 Masked Language Modeling

**Challenge**: Assignment asks for 7B model like Mistral-7B-Instruct  
**Issue**: Mistral is decoder-only and cannot do masked language modeling  
**Solution**: Use BERT/RoBERTa (encoder-only models designed for MLM)

### Results
Two tokens ("mat" and "tired") are masked and predicted:
- **Masked sentence**: "The cat sat on the [MASK] because it was [MASK]."
- **Predictions**: Top-3 results for each mask with plausibility comments
- **Analysis**: Available in `predictions.json` and `compare.md`

## 📋 Assignment Deliverables

✅ **tokenise.py** - All three tokenization methods  
✅ **predictions.json** - Structured prediction results  
✅ **compare.md** - Analysis and comparison  
✅ **README.md** - This documentation  

## 🔗 Submission

This is a **public GitHub repository** containing all required deliverables as specified in the assignment. 