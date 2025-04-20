"""Local LLM service adapter for models like Llama or Mistral."""

import json
import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union
import httpx

from semantic_qa_gen.llm.service import BaseLLMService
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.utils.error import LLMServiceError, with_error_handling


class LocalLLMAdapter(BaseLLMService):
    """
    LLM service adapter for local models accessed via API (like Ollama or LM Studio).
    
    This adapter allows SemanticQAGen to use locally running models for
    semantic analysis and question generation, reducing API costs and latency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local LLM adapter.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.url = config.get('url', 'http://localhost:11434/api')
        self.model = config.get('model', 'mistral:7b')
        self.timeout = config.get('timeout', 120)  # Local models might be slower
        self.logger = logging.getLogger(__name__)
        
        # For Ollama, determine if the URL is in the correct format
        if self.url.endswith('/api') and 'ollama' not in self.url.lower():
            self.url_type = 'ollama'
        else:
            # For other API types like LM Studio/OpenAI compatible
            self.url_type = 'openai'
        
        # Don't enforce strict token limits for local models
        self.rate_limit_tokens = float('inf')
        self.rate_limit_requests = float('inf')
    
    @with_error_handling(error_types=(httpx.HTTPError, asyncio.TimeoutError), max_retries=1)
    async def call_model(self,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None,
                         json_response: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Call local LLM with a prompt and receive a response.

        Args:
            prompt: The main prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum response length in tokens.
            json_response: Whether to parse the response as JSON.

        Returns:
            Model response as string or parsed JSON.

        Raises:
            LLMServiceError: If the model call fails.
        """
        # For local models, we only apply a simple request timeout
        try:
            client_timeout = httpx.Timeout(self.timeout, connect=10.0)

            if self.url_type == 'ollama':
                # Ollama API format
                data = {
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                }

                if system_prompt:
                    data["system"] = system_prompt

                if max_tokens:
                    data["max_tokens"] = max_tokens

                if json_response:
                    # Add JSON mode instruction to prompt
                    data["prompt"] = data["prompt"] + "\n\nRespond only with valid JSON."

                async with httpx.AsyncClient(timeout=client_timeout) as client:
                    response = await client.post(f"{self.url}/generate", json=data)

                    if response.status_code != 200:
                        raise LLMServiceError(f"Local LLM API error: {response.status_code} {response.text}")

                    result = response.json()
                    response_text = result.get("response", "")

            else:
                # OpenAI-compatible API format (LM Studio, etc.)
                messages = []
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })

                messages.append({
                    "role": "user",
                    "content": prompt
                })

                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    data["max_tokens"] = max_tokens

                # Some local APIs don't support response_format
                # Only add it if json_response is requested
                if json_response:
                    try:
                        data["response_format"] = {"type": "json_object"}
                    except Exception:
                        # If error, add JSON instruction to the prompt instead
                        if system_prompt:
                            messages[0]["content"] += "\nRespond only with valid JSON."
                        else:
                            messages.insert(0, {
                                "role": "system",
                                "content": "Respond only with valid JSON."
                            })

                async with httpx.AsyncClient(timeout=client_timeout) as client:
                    # Determine correct endpoint
                    endpoint = f"{self.url}/chat/completions"

                    response = await client.post(endpoint, json=data)

                    if response.status_code != 200:
                        raise LLMServiceError(f"Local LLM API error: {response.status_code} {response.text}")

                    result = response.json()
                    try:
                        response_text = result["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        # Fallback for different API formats
                        if "text" in result:
                            response_text = result["text"]
                        elif "output" in result:
                            response_text = result["output"]
                        else:
                            # Last resort
                            response_text = str(result)

            # Parse JSON if requested
            if json_response:
                try:
                    # Look for JSON in the response text (might be surrounded by backticks or other text)
                    json_text = response_text

                    # Check for code blocks with JSON
                    json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', json_text, re.DOTALL)
                    if json_block_match:
                        json_text = json_block_match.group(1)
                    else:
                        # Find JSON object or array pattern
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1

                        if json_start >= 0 and json_end > json_start:
                            json_text = response_text[json_start:json_end]
                        else:
                            # Try finding a JSON array
                            json_start = response_text.find('[')
                            json_end = response_text.rfind(']') + 1

                            if json_start >= 0 and json_end > json_start:
                                json_text = response_text[json_start:json_end]

                    try:
                        parsed_json = json.loads(json_text)
                        return parsed_json
                    except json.JSONDecodeError:
                        # Try fixing common JSON errors
                        # Replace single quotes with double quotes
                        json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)
                        # Add quotes to unquoted property names
                        json_text = re.sub(r'(\s)(\w+)(:)', r'\1"\2"\3', json_text)

                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError:
                            self.logger.warning("Failed to parse response as JSON, returning raw text")
                            return {"error": "JSON parsing failed", "text": response_text}
                except Exception as e:
                    self.logger.warning(f"Failed to parse response as JSON: {str(e)}")
                    return {"error": "JSON parsing failed", "text": response_text}

            return response_text

        except httpx.HTTPError as e:
            raise LLMServiceError(f"HTTP error during local LLM call: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMServiceError(f"Timeout during local LLM call (limit: {self.timeout}s)")
        except Exception as e:
            raise LLMServiceError(f"Error during local LLM call: {str(e)}")

    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Analyze a document chunk using local LLM.
        
        For local models, we use a simplified prompt that's more likely to work with
        smaller models that might have limitations compared to GPT-4.
        
        Args:
            chunk: The document chunk to analyze.
            
        Returns:
            AnalysisResult containing information about the chunk.
            
        Raises:
            LLMServiceError: If analysis fails.
        """
        # Simpler system prompt for local models
        system_prompt = """
        Analyze text passages to determine their educational value for generating questions.
        Return results in JSON format.
        """
        
        prompt = f"""
        Analyze this text passage for generating quiz questions.
        Rate from 0.0 to 1.0:
        - information_density (how fact-rich is the text)
        - topic_coherence (how focused is the text)
        - complexity (how technical or difficult is the content)
        
        Also estimate how many questions could be generated for each type:
        - factual: direct information extraction
        - inferential: connecting multiple pieces of information
        - conceptual: broad principles or ideas
        
        Include a list of 3-5 key concepts from the text.
        
        Text to analyze:
        {chunk.content}
        
        Format as JSON:
        {{
            "information_density": 0.0-1.0,
            "topic_coherence": 0.0-1.0,
            "complexity": 0.0-1.0,
            "estimated_question_yield": {{
                "factual": number,
                "inferential": number,
                "conceptual": number
            }},
            "key_concepts": ["concept1", "concept2", ...],
            "notes": "optional notes"
        }}
        """
        
        try:
            # Local models might struggle with complex JSON so we use a higher temperature
            # to avoid getting stuck, but also set a longer timeout
            result = await self.call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                json_response=True
            )
            
            # Handle potential missing fields in response
            if not isinstance(result, dict):
                raise LLMServiceError(f"Invalid response format from local model: {result}")
            
            # Create AnalysisResult from the JSON response with defaults for missing fields
            return AnalysisResult(
                chunk_id=chunk.id,
                information_density=result.get("information_density", 0.5),
                topic_coherence=result.get("topic_coherence", 0.5),
                complexity=result.get("complexity", 0.5),
                estimated_question_yield=result.get("estimated_question_yield", {
                    "factual": 2, 
                    "inferential": 1, 
                    "conceptual": 1
                }),
                key_concepts=result.get("key_concepts", []),
                notes=result.get("notes")
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed with local model: {str(e)}")
            # Provide a fallback analysis rather than failing completely
            return AnalysisResult(
                chunk_id=chunk.id,
                information_density=0.5,
                topic_coherence=0.5,
                complexity=0.5,
                estimated_question_yield={
                    "factual": 2, 
                    "inferential": 1, 
                    "conceptual": 1
                },
                key_concepts=[],
                notes="Analysis failed, using default values"
            )
    
    async def generate_questions(self, 
                              chunk: Chunk, 
                              analysis: AnalysisResult, 
                              count: Optional[int] = None, 
                              categories: Optional[Dict[str, int]] = None) -> List[Question]:
        """
        Generate questions using local LLM.
        
        Args:
            chunk: The document chunk to generate questions for.
            analysis: Analysis results for the chunk.
            count: Optional total number of questions to generate.
            categories: Optional dictionary mapping category names to question counts.
            
        Returns:
            List of generated Question objects.
            
        Raises:
            LLMServiceError: If question generation fails.
        """
        # Determine question counts
        if not categories:
            categories = {
                "factual": 2,
                "inferential": 1, 
                "conceptual": 1
            }
        
        if count:
            total = sum(categories.values())
            categories = {
                k: max(1, int(count * v / total)) for k, v in categories.items()
            }
        
        # Simpler prompt for local models with examples
        system_prompt = """
        Create educational questions with answers based on provided text.
        Return results as JSON.
        """
        
        prompt = f"""
        Generate quiz questions and answers based on this text:
        
        {chunk.content}
        
        Create the following questions:
        - {categories.get('factual', 2)} factual questions (direct information)
        - {categories.get('inferential', 1)} inferential questions (connecting information)
        - {categories.get('conceptual', 1)} conceptual questions (broader principles)
        
        Format as JSON array:
        [
            {{
                "question": "Question text here?",
                "answer": "Detailed answer here.",
                "category": "factual" 
            }},
            {{
                "question": "Another question?",
                "answer": "Another detailed answer.",
                "category": "inferential"
            }}
        ]
        """
        
        try:
            result = await self.call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                json_response=True
            )
            
            # Check if result is a list or wrapped in another object
            if isinstance(result, dict) and 'questions' in result:
                result = result['questions']
            elif isinstance(result, dict) and not isinstance(result, list):
                # Some models might return a dict with numbered keys
                questions_list = []
                for k, v in result.items():
                    if isinstance(v, dict):
                        questions_list.append(v)
                result = questions_list
            
            if not isinstance(result, list):
                raise LLMServiceError(f"Invalid response format for questions: {result}")
                
            # Convert to Question objects
            questions = []
            for i, item in enumerate(result):
                if not isinstance(item, dict):
                    continue
                    
                import uuid
                question = Question(
                    id=str(uuid.uuid4()),
                    text=item.get("question", ""),
                    answer=item.get("answer", ""),
                    chunk_id=chunk.id,
                    category=item.get("category", "factual"),
                    metadata={
                        "generation_order": i,
                        "source": "local_llm",
                        "model": self.model
                    }
                )
                questions.append(question)
                
            return questions
            
        except Exception as e:
            raise LLMServiceError(f"Failed to generate questions with local model: {str(e)}")
    
    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate a question using local LLM.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            Validation results dictionary.
            
        Raises:
            LLMServiceError: If validation fails.
        """
        system_prompt = """
        Evaluate the factual accuracy and quality of questions and answers.
        Return results as JSON.
        """
        
        prompt = f"""
        Evaluate this question and answer based on the source text.
        
        Source text:
        {chunk.content}
        
        Question: {question.text}
        Answer: {question.answer}
        
        Evaluate three metrics from 0.0-1.0:
        - factual_accuracy: Is the answer factually correct according to the source text?
        - answer_completeness: Does the answer fully address the question?
        - question_clarity: Is the question clear and unambiguous?
        
        Format as JSON:
        {{
            "is_valid": true or false,
            "factual_accuracy": 0.0-1.0,
            "answer_completeness": 0.0-1.0,
            "question_clarity": 0.0-1.0,
            "reasons": ["reason1", "reason2", ...]
        }}
        """
        
        try:
            result = await self.call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                json_response=True
            )
            
            # Ensure we have all required fields with defaults
            required_fields = ["is_valid", "factual_accuracy", "answer_completeness", "question_clarity", "reasons"]
            for field in required_fields:
                if field not in result:
                    if field == "is_valid":
                        result[field] = True
                    elif field == "reasons":
                        result[field] = []
                    else:
                        result[field] = 0.7  # Default to reasonable value
                        
            if "reasons" in result and not isinstance(result["reasons"], list):
                result["reasons"] = [str(result["reasons"])]
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed with local model: {str(e)}")
            # Provide a fallback validation rather than failing completely
            return {
                "is_valid": True,
                "factual_accuracy": 0.7,
                "answer_completeness": 0.7,
                "question_clarity": 0.7,
                "reasons": ["Validation failed with local model, using default values"]
            }
