"""Remote LLM service adapter using OpenAI API."""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import httpx
import tiktoken
import time

from semantic_qa_gen.llm.service import BaseLLMService
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.utils.error import LLMServiceError, with_error_handling


class OpenAIAdapter(BaseLLMService):
    """
    LLM service adapter for OpenAI API.
    
    This adapter allows SemanticQAGen to use OpenAI models like GPT-4
    for semantic analysis and question generation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI adapter.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise LLMServiceError("OpenAI API key is required")

        self.model = config.get('model', 'gpt-4')
        self.api_base = config.get('api_base', 'https://api.openai.com/v1')
        self.organization = config.get('organization')
        self.logger = logging.getLogger(__name__)

        # Initialize token counter
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.logger.warning(f"No specific tokenizer found for {self.model}, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize token usage tracking
        self._token_usage = []
        self._request_timestamps = []

    @with_error_handling(error_types=(httpx.HTTPError, asyncio.TimeoutError), max_retries=2)
    async def call_model(self, 
                       prompt: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       json_response: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Call OpenAI API with a prompt and receive a response.
        
        Args:
            prompt: The main prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum response length in tokens.
            json_response: Whether to parse the response as JSON.
            
        Returns:
            Model response as string or parsed JSON.
            
        Raises:
            LLMServiceError: If the API call fails.
        """
        # Estimate token usage for rate limiting
        estimated_input_tokens = self._count_tokens(prompt)
        if system_prompt:
            estimated_input_tokens += self._count_tokens(system_prompt)
        
        # Add overhead for message formatting
        estimated_input_tokens += 50
        
        # Enforce rate limits
        await self._enforce_rate_limits(estimated_input_tokens)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
            
        # Prepare response format if JSON is requested
        response_format = {"type": "json_object"} if json_response else None
        
        # Prepare messages
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
        
        # Prepare request data
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
            
        if response_format:
            data["response_format"] = response_format
        
        # Make API request
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                # Handle API errors
                if response.status_code != 200:
                    error_message = f"OpenAI API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_message = f"OpenAI API error: {error_data['error'].get('message', 'Unknown error')}"
                    except:
                        pass
                    
                    raise LLMServiceError(error_message)
                
                # Parse response
                response_data = response.json()
                result = response_data["choices"][0]["message"]["content"]

                # Update token usage tracking
                if 'usage' in response_data:
                    total_tokens = response_data.get("usage", {}).get("total_tokens", 0)
                    # Ensure we have at least one item in the token usage list
                    if not self._token_usage:
                        self._token_usage.append((time.time(), total_tokens))
                    else:
                        self._token_usage[-1] = (self._token_usage[-1][0], total_tokens)
                
                # Return result in requested format
                if json_response:
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse response as JSON, returning raw text")
                        return {"error": "JSON parsing failed", "text": result}
                
                return result
                
            except httpx.HTTPError as e:
                raise LLMServiceError(f"HTTP error during OpenAI API call: {str(e)}")
            except asyncio.TimeoutError:
                raise LLMServiceError(f"Timeout during OpenAI API call (limit: {self.timeout}s)")
            except Exception as e:
                raise LLMServiceError(f"Error during OpenAI API call: {str(e)}")
    
    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Analyze a document chunk to determine its information density and question potential.
        
        Args:
            chunk: The document chunk to analyze.
            
        Returns:
            AnalysisResult containing information about the chunk.
            
        Raises:
            LLMServiceError: If analysis fails.
        """
        # This is a placeholder - we'll implement this with proper prompts later
        # For now, let's just create a stub that can be expanded with prompt templates
        system_prompt = """
        You are an AI assistant specialized in analyzing text passages to determine:
        1. Information density: How rich in facts and information the text is
        2. Topic coherence: How focused the text is on specific topics
        3. Complexity: How complex or technical the content is
        4. Question yield potential: How many good questions could be generated from this text
        5. Key concepts: The main ideas or terms in the text

        Provide your analysis in JSON format.
        """
        
        prompt = f"""
        Please analyze the following text passage and provide information about its 
        educational value for generating quiz questions. Focus on aspects like information
        density (0.0-1.0), topic coherence (0.0-1.0), complexity (0.0-1.0), and how many
        questions of different types could be generated from it.

        Text passage:
        ---
        {chunk.content}
        ---
        
        Include estimates for:
        - factual questions (direct information in the text)
        - inferential questions (requiring connecting information)
        - conceptual questions (dealing with broader principles)
        
        Format your response as JSON with the following structure:
        {{
            "information_density": float,
            "topic_coherence": float,
            "complexity": float,
            "estimated_question_yield": {{
                "factual": int,
                "inferential": int,
                "conceptual": int
            }},
            "key_concepts": [string],
            "notes": string
        }}
        """
        
        try:
            result = await self.call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent analysis
                json_response=True
            )
            
            # Create AnalysisResult from the JSON response
            return AnalysisResult(
                chunk_id=chunk.id,
                information_density=result.get("information_density", 0.5),
                topic_coherence=result.get("topic_coherence", 0.5),
                complexity=result.get("complexity", 0.5),
                estimated_question_yield=result.get("estimated_question_yield", {
                    "factual": 3, 
                    "inferential": 2, 
                    "conceptual": 1
                }),
                key_concepts=result.get("key_concepts", []),
                notes=result.get("notes")
            )
            
        except Exception as e:
            raise LLMServiceError(f"Failed to analyze chunk: {str(e)}")
    
    async def generate_questions(self, 
                               chunk: Chunk, 
                               analysis: AnalysisResult, 
                               count: Optional[int] = None, 
                               categories: Optional[Dict[str, int]] = None) -> List[Question]:
        """
        Generate questions for a document chunk based on analysis.
        
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
        # This is a placeholder implementation
        # We'll expand this with proper prompt templates later
        system_prompt = """
        You are an AI assistant specialized in creating educational questions and answers
        based on provided text. Generate questions that accurately reflect the content
        and are suitable for testing knowledge and understanding.
        
        Follow these guidelines:
        - Create questions of different cognitive levels (factual, inferential, conceptual)
        - Ensure questions are clear, concise, and unambiguous
        - Provide comprehensive answers that fully address the questions
        - Use the exact terminology from the source text when appropriate
        - Format your response as JSON
        """
        
        # Determine how many questions of each category to generate
        if not categories:
            categories = {
                "factual": 3,
                "inferential": 2, 
                "conceptual": 1
            }
        
        # Use the analysis to adjust if needed
        if analysis and not count:
            categories = analysis.estimated_question_yield
        
        if count:
            # Adjust the category distribution based on the total count
            total = sum(categories.values())
            categories = {
                k: max(1, int(count * v / total)) for k, v in categories.items()
            }
            # Ensure the sum matches the requested count
            remaining = count - sum(categories.values())
            for k in sorted(categories.keys(), key=lambda k: categories[k]):
                if remaining <= 0:
                    break
                categories[k] += 1
                remaining -= 1
        
        prompt = f"""
        Generate questions and answers based on the following text. Create {sum(categories.values())} questions total:
        - {categories.get('factual', 0)} factual questions (based directly on information in the text)
        - {categories.get('inferential', 0)} inferential questions (requiring connecting information from the text)
        - {categories.get('conceptual', 0)} conceptual questions (addressing broader principles or ideas)
        
        Text:
        ---
        {chunk.content}
        ---
        
        Format your response as a JSON array of question objects with the following structure:
        [
            {{
                "question": "The question text",
                "answer": "The comprehensive answer",
                "category": "factual|inferential|conceptual"
            }},
            ...
        ]
        
        Make the answers comprehensive and accurate based on the text. Each answer should fully explain the concept
        being asked about, not just provide a short answer.
        """
        
        try:
            result = await self.call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,  # Higher temperature for more diverse questions
                json_response=True
            )
            
            # Check if result is a list (expected format)
            if not isinstance(result, list):
                if isinstance(result, dict) and 'error' in result:
                    raise LLMServiceError(f"JSON formatting error: {result.get('text', '')}")
                elif isinstance(result, dict) and 'questions' in result:
                    # Some models might wrap the array in a "questions" field
                    result = result['questions']
                else:
                    raise LLMServiceError("Unexpected response format from API")
            
            # Convert to Question objects
            questions = []
            for i, item in enumerate(result):
                import uuid
                question = Question(
                    id=str(uuid.uuid4()),
                    text=item.get("question", ""),
                    answer=item.get("answer", ""),
                    chunk_id=chunk.id,
                    category=item.get("category", "factual"),
                    metadata={
                        "generation_order": i,
                        "key_concepts": [c for c in analysis.key_concepts if c.lower() in 
                                         item.get("question", "").lower() or 
                                         c.lower() in item.get("answer", "").lower()]
                    }
                )
                questions.append(question)
                
            return questions
            
        except Exception as e:
            raise LLMServiceError(f"Failed to generate questions: {str(e)}")
    
    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate a generated question against the source chunk.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            Validation results dictionary.
            
        Raises:
            LLMServiceError: If validation fails.
        """
        # Placeholder implementation - to be expanded with proper prompts
        system_prompt = """
        You are an AI assistant specialized in evaluating the quality of educational
        questions and answers. Verify that questions are clear and that answers are
        accurate, complete, and supported by the source text.
        """
        
        prompt = f"""
        Evaluate the following question and answer based on the provided source text.
        
        Source text:
        ---
        {chunk.content}
        ---
        
        Question: {question.text}
        
        Answer: {question.answer}
        
        Please verify:
        1. Factual accuracy: Is the answer factually correct according to the source text?
        2. Answer completeness: Does the answer fully address the question?
        3. Question clarity: Is the question clear and unambiguous?
        
        Format your response as JSON with the following structure:
        {{
            "is_valid": true/false,
            "factual_accuracy": float (0.0-1.0),
            "answer_completeness": float (0.0-1.0),
            "question_clarity": float (0.0-1.0),
            "reasons": [string],
            "suggested_improvements": string (optional)
        }}
        """
        
        try:
            result = await self.call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                json_response=True
            )
            
            # Ensure we have all required fields
            required_fields = ["is_valid", "factual_accuracy", "answer_completeness", "question_clarity", "reasons"]
            for field in required_fields:
                if field not in result:
                    result[field] = None if field == "reasons" else 0.0
                if field == "reasons" and not result[field]:
                    result[field] = []
                    
            return result
            
        except Exception as e:
            raise LLMServiceError(f"Failed to validate question: {str(e)}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens in.
            
        Returns:
            Number of tokens.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough approximation
            return len(text) // 4
