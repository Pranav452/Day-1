# Hallucination Detection & Guardrails

## 🧠 What is Hallucination?

**Hallucination** in AI refers to when language models generate information that is factually incorrect, made-up, or not grounded in reality. It's like when a human confidently states something that isn't true.

**Examples of AI Hallucination:**
- Asking "Who discovered gravity?" and getting "Elon Musk" instead of "Isaac Newton"
- Asking for a historical date and getting a completely wrong year
- Making up citations, names, or facts that don't exist

## 🛡️ How We Detect Hallucination

Our hallucination detection system works by:

1. **Knowledge Base (KB)**: We maintain a `kb.json` file with verified, factual question-answer pairs
2. **Validation**: Compare the AI model's answers against our known correct answers
3. **Categorization**: Flag responses as either:
   - ✅ **OK**: Answer matches knowledge base
   - 🛑 **RETRY: answer differs from KB**: Wrong answer for a known question
   - 🛑 **RETRY: out-of-domain**: Question not in our knowledge base

## 🔄 Retry Logic

When hallucination is detected, our system:

1. **First Attempt**: Ask the question to the model
2. **Validation**: Check against knowledge base
3. **Retry**: If validation fails, ask the same question again
4. **Final Result**: Report the outcome of both attempts

This gives the model a second chance and helps us understand consistency patterns.

## 🤖 Model Used

We use **distilgpt2** for this demonstration because:
- **Lightweight**: ~500MB, runs on CPU
- **Fast**: Quick responses for testing
- **Accessible**: Available through Hugging Face Transformers
- **Demonstrative**: Shows hallucination clearly due to its limitations

## 📁 File Structure

```
Hallucination Detection & Guardrails/
├── kb.json                 ← Knowledge base with 10 factual Q-A pairs
├── ask_model.py            ← Main script to ask questions to model
├── validator.py            ← Hallucination detection validator
├── run.log                 ← Execution log file
├── summary.md              ← This explanation file
└── results_log.json        ← Generated detailed test results
```

## 🚀 How to Run

1. **Install Requirements:**
   ```bash
   pip install transformers torch
   ```

2. **Run the Script:**
   ```bash
   cd q2
   python ask_and_validate.py
   ```

3. **What Happens:**
   - Loads the knowledge base
   - Downloads/loads the distilgpt2 model
   - Tests 15 questions (10 from KB + 5 tricky ones)
   - Shows validation results with retry logic
   - Saves detailed results to `results_log.json`

## 📊 Test Questions

### Knowledge Base Questions (10):
1. What is the capital of France?
2. Who wrote Hamlet?
3. What is the boiling point of water in Celsius?
4. Who is the CEO of Tesla?
5. What is the square root of 64?
6. When did World War 2 end?
7. What is the currency of Japan?
8. Which planet is known as the Red Planet?
9. What is the largest ocean?
10. What gas do plants inhale during photosynthesis?

### Tricky/Out-of-Domain Questions (5):
1. What is Elon Musk's dog's name?
2. What is the meaning of life?
3. Who invented the internet?
4. What is the color of the wind?
5. How many legs does a unicorn have?

## 🔍 Expected Outcomes

- **KB Questions**: May get correct answers or hallucinated responses
- **Out-of-Domain**: Will trigger "RETRY: out-of-domain" validation
- **Inconsistent Model**: distilgpt2 often gives different answers on retry

## 📈 Observations

1. **Model Limitations**: distilgpt2 frequently hallucinates due to its small size
2. **Consistency Issues**: Retry attempts often yield different (sometimes better) answers
3. **Domain Knowledge**: Out-of-domain questions reliably trigger our guardrails
4. **Validation Effectiveness**: Our simple string-matching catches obvious hallucinations

## 🚀 Potential Enhancements

- **Fuzzy Matching**: Use Levenshtein distance for more flexible answer comparison
- **Confidence Scoring**: Rate how confident the model is in its answers
- **Multi-Model Testing**: Compare hallucination rates across different models
- **Semantic Similarity**: Use embeddings to compare meaning rather than exact text
- **Fact-Checking APIs**: Integrate with external fact-checking services
