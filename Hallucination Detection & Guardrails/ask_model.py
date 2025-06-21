import json
import logging
from datetime import datetime
from transformers import pipeline
import warnings
from validator import Validator

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelAsker:
    """Class to handle asking questions to language models"""
    
    def __init__(self):
        """Initialize the model asker"""
        self.validator = Validator()
        self.model = self.load_model()
        self.results = []
        
    def load_model(self):
        """Load the text generation model"""
        try:
            logger.info("Loading model (this may take a moment)...")
            # Using distilgpt2 as it's lightweight and good for demonstration
            model = pipeline("text-generation", model="distilgpt2", pad_token_id=50256)
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def ask_question(self, question):
        """Ask a question to the model and get response"""
        if not self.model:
            return "Model not available"
        
        # Format the prompt for Q&A
        prompt = f"Q: {question}\nA:"
        
        try:
            # Generate response
            response = self.model(prompt, max_new_tokens=20, do_sample=True, temperature=0.7, pad_token_id=50256)
            
            # Extract the answer part (remove the original prompt)
            full_text = response[0]['generated_text']
            answer = full_text[len(prompt):].strip()
            
            # Clean up the answer (take only the first line/sentence)
            answer = answer.split('\n')[0].strip()
            if answer.endswith('.'):
                answer = answer[:-1]
                
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def process_question_with_retry(self, question):
        """Process a question with retry logic"""
        logger.info(f"\nQ: {question}")
        
        # First attempt
        first_answer = self.ask_question(question)
        logger.info(f"A (1st attempt): {first_answer}")
        
        # Validate first attempt
        validation_result = self.validator.validate_answer(question, first_answer)
        
        result_data = {
            "question": question,
            "first_answer": first_answer,
            "first_validation": validation_result,
            "timestamp": datetime.now().isoformat(),
            "correct_answer": self.validator.get_correct_answer(question),
            "in_kb": self.validator.is_in_kb(question)
        }
        
        if validation_result == "OK":
            logger.info("✅ OK")
            result_data.update({
                "retry_needed": False,
                "final_answer": first_answer,
                "final_validation": "OK"
            })
        else:
            logger.info(f"🛑 {validation_result}")
            
            # Retry once
            logger.info("Retrying...")
            second_answer = self.ask_question(question)
            logger.info(f"A (2nd attempt): {second_answer}")
            
            # Validate second attempt
            second_validation = self.validator.validate_answer(question, second_answer)
            if second_validation == "OK":
                logger.info("✅ OK (after retry)")
            else:
                logger.info(f"🛑 Still {second_validation}")
            
            result_data.update({
                "retry_needed": True,
                "second_answer": second_answer,
                "final_validation": second_validation,
                "final_answer": second_answer
            })
        
        self.results.append(result_data)
        return result_data
    
    def run_full_test(self):
        """Run the complete hallucination detection test"""
        logger.info("🧠 Hallucination Detection & Guardrails Test")
        logger.info("=" * 50)
        
        # Questions from knowledge base (10 questions)
        kb_questions = self.validator.get_all_questions()
        
        # Additional tricky questions (5 questions not in KB)
        edge_case_questions = [
            "What is Elon Musk's dog's name?",
            "What is the meaning of life?",
            "Who invented the internet?",
            "What is the color of the wind?",
            "How many legs does a unicorn have?"
        ]
        
        all_questions = kb_questions + edge_case_questions
        
        logger.info(f"Testing {len(all_questions)} questions ({len(kb_questions)} from KB + {len(edge_case_questions)} edge cases)")
        logger.info("=" * 50)
        
        for question in all_questions:
            self.process_question_with_retry(question)
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def generate_summary(self):
        """Generate and log summary statistics"""
        logger.info("\n" + "=" * 50)
        logger.info("📊 SUMMARY")
        logger.info("=" * 50)
        
        ok_count = sum(1 for r in self.results if r.get('final_validation') == 'OK')
        retry_count = sum(1 for r in self.results if r.get('retry_needed', False))
        kb_questions = sum(1 for r in self.results if r.get('in_kb', False))
        out_of_domain = sum(1 for r in self.results if not r.get('in_kb', False))
        
        logger.info(f"Total questions: {len(self.results)}")
        logger.info(f"OK answers: {ok_count}")
        logger.info(f"Retries needed: {retry_count}")
        logger.info(f"KB questions: {kb_questions}")
        logger.info(f"Edge case questions: {out_of_domain}")
        
        # Accuracy for KB questions
        kb_correct = sum(1 for r in self.results if r.get('in_kb', False) and r.get('final_validation') == 'OK')
        kb_accuracy = (kb_correct / kb_questions * 100) if kb_questions > 0 else 0
        logger.info(f"KB accuracy: {kb_accuracy:.1f}%")
    
    def save_results(self):
        """Save detailed results to JSON file"""
        try:
            with open('results_log.json', 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n📁 Detailed results saved to 'results_log.json'")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main function to run the hallucination detection test"""
    logger.info("Initializing Hallucination Detection System...")
    
    # Create model asker instance
    asker = ModelAsker()
    
    if not asker.validator.kb:
        logger.error("Failed to load knowledge base. Exiting.")
        return
    
    if not asker.model:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run the full test
    results = asker.run_full_test()
    
    logger.info("🎉 Test completed successfully!")

if __name__ == "__main__":
    main() 