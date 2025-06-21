import json

class Validator:
    """Validator class for detecting hallucinations in model responses"""
    
    def __init__(self, kb_path="kb.json"):
        """Initialize validator with knowledge base"""
        self.kb_path = kb_path
        self.kb = self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load the knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Knowledge base file '{self.kb_path}' not found!")
            return {}
    
    def validate_answer(self, question, model_answer):
        """
        Validate the model's answer against the knowledge base
        
        Returns:
        - "OK": Answer matches knowledge base
        - "RETRY: answer differs from KB": Wrong answer for known question
        - "RETRY: out-of-domain": Question not in knowledge base
        """
        correct_answer = self.kb.get(question)
        
        if correct_answer:
            # Check if the correct answer is in the model's response (case-insensitive)
            if correct_answer.lower() in model_answer.lower():
                return "OK"
            else:
                return "RETRY: answer differs from KB"
        else:
            return "RETRY: out-of-domain"
    
    def get_correct_answer(self, question):
        """Get the correct answer for a question from KB"""
        return self.kb.get(question, "Not in knowledge base")
    
    def is_in_kb(self, question):
        """Check if question exists in knowledge base"""
        return question in self.kb
    
    def get_all_questions(self):
        """Get all questions from knowledge base"""
        return list(self.kb.keys()) 