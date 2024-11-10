import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from ollama import Client
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStep:
    question: str
    context: str
    answer: Optional[str] = None

class OllamaProcessor:
    def __init__(
        self,
        model_name: str = "mistral",
        host: str = "http://localhost:11434",
        output_dir: Path = Path("output"),
        temperature: float = 0.7,
        max_tokens_per_chunk: int = 2048
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Ollama client
        logger.info(f"Connecting to Ollama at {host}")
        self.client = Client(host=host)
        
        # Test connection and model availability
        try:
            self.client.list()
            logger.info("Successfully connected to Ollama")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
        
        # Templates for prompts
        self.step_generation_template = """
        Your task is to break down the following input into smaller, manageable steps or questions.
        Each step should be self-contained and contribute to understanding the full input.
        
        Format your response strictly as a JSON list of dictionaries with 'question' and 'context' keys.
        Make sure your response can be parsed as valid JSON.
        
        Sample Input:
        Explain each on in details with examples .
        8. Defuzzification  
        9. Fuzzy Reasoning

        Sample Output (JSON format):
        question = "What is Defuzzification?",
        context = "Explain the concept of Defuzzification with examples."

        question = "What is Fuzzy Reasoning?",
        context = "Provide an explanation of Fuzzy Reasoning and its significance."

        Input: {input_text}
        
        Response (in JSON format):
        """
        
        self.answer_generation_template = """
        Question: {question}
        Context: {context}
        
        Provide a detailed answer to the question above based on the given context.
        Think through this step-by-step and explain your reasoning.
        
        Your response:
        """

    def generate_response(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """Generate a response using Ollama with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False,
                    options={
                        "temperature": self.temperature,
                    }
                )
                return response['response'].strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def generate_steps(self, input_text: str) -> List[ProcessingStep]:
        """Generate processing steps from input text."""
        prompt = self.step_generation_template.format(input_text=input_text)
        
        for _ in range(3):  # Retry with different prompts if JSON parsing fails
            try:
                response = self.generate_response(prompt)
                # Clean up the response to help with JSON parsing
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                
                steps_data = json.loads(response)
                
                # Convert to ProcessingStep objects
                steps = [
                    ProcessingStep(
                        question=step["question"],
                        context=step["context"]
                    )
                    for step in steps_data
                ]
                
                logger.info(f"Generated {len(steps)} processing steps")
                return steps
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse steps response as JSON: {e}")
                # Modify the prompt to be more explicit about JSON formatting
                prompt = prompt + "\nPlease ensure your response is valid JSON without any markdown formatting."
        
        raise Exception("Failed to generate valid JSON steps after multiple attempts")

    def process_step(self, step: ProcessingStep) -> str:
        """Process a single step and generate an answer."""
        prompt = self.answer_generation_template.format(
            question=step.question,
            context=step.context
        )
        
        return self.generate_response(prompt)

    def process_input(self, input_text: str) -> Dict:
        """Process the full input text through the pipeline."""
        logger.info("Starting input processing pipeline")
        
        # Generate steps
        steps = self.generate_steps(input_text)
        
        # Process each step
        results = []
        for i, step in enumerate(tqdm(steps, desc="Processing steps")):
            logger.info(f"Processing step {i+1}/{len(steps)}: {step.question}")
            
            answer = self.process_step(step)
            step.answer = answer
            
            results.append({
                "step": i + 1,
                "question": step.question,
                "context": step.context,
                "answer": answer
            })
            
            # Save intermediate results
            self.save_results(results, f"intermediate_results_{i+1}.json")
        
        # Save final results
        final_output = {
            "input_text": input_text,
            "steps": results,
            "summary": self.generate_summary(results)
        }
        
        self.save_results(final_output, "final_results.json")
        return final_output

    def generate_summary(self, results: List[Dict]) -> str:
        """Generate a summary of all processed steps."""
        summary_prompt = """
        Summarize the following findings into a coherent response.
        Highlight key insights and connections between different steps.
        
        Steps to summarize:
        """
        
        for result in results:
            summary_prompt += f"\nQuestion: {result['question']}\n"
            summary_prompt += f"Answer: {result['answer']}\n"
        
        return self.generate_response(summary_prompt)

    def save_results(self, results: Dict, filename: str):
        """Save results to a JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {output_path}")

def main():
    # Example usage
    input_text = """
    Explain each on in details with examples for 10 mark answer in exam.
    Fuzzy Rule Base
    Fuzzy If-Then Rules
    Fuzzy Inference System (FIS)
    Mamdani Inference Method
    Sugeno Inference Method
    Tsukamoto Inference Method
    Fuzzification
    Defuzzification
    Fuzzy Reasoning
    Rule Aggregation
    Rule Evaluation
    Linguistic Variables
    Fuzzy Antecedents
    Fuzzy Consequents
    Fuzzy Rule Weighting
    """
    
    processor = OllamaProcessor(
        model_name="mistral",  # or any other model you have pulled in Ollama
        host="http://localhost:11434",
        output_dir="output"
    )
    
    results = processor.process_input(input_text)
    logger.info("Processing complete!")


def final_json_to_readme():
    with open('output/final_results.json') as json_file:
        data = json.load(json_file)
        with open('output/README.md', 'w') as f:
            f.write(f"# Output\n\n## Input\n\n{data['input_text']}\n\n## Summary\n\n{data['summary']}\n\n## Steps\n\n")
            for step in data['steps']:
                f.write(f"### Step {step['step']}\n\n**Question:** {step['question']}\n\n**Context:** {step['context']}\n\n**Answer:** {step['answer']}\n\n")

if __name__ == "__main__":
    main()
    final_json_to_readme()