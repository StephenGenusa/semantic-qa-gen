"""Natural language processing helpers for chunking."""

import re
from typing import List, Set, Optional, Tuple

# Check if NLTK is available
try:
    import nltk
    NLTK_AVAILABLE = True
    
    # Try to import common NLTK resources
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences.
    
    Args:
        text: Text to tokenize.
        
    Returns:
        List of sentences.
    """
    if NLTK_AVAILABLE:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            # Fall back to regex if NLTK fails for any reason
            pass

    # Fallback to simple regex-based sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Text to tokenize.

    Returns:
        List of words.
    """
    if NLTK_AVAILABLE:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            # Fall back to regex if NLTK fails
            pass

    # Fallback to simple word pattern matching with punctuation handling
    return re.findall(r'\b\w+\b', text.lower())



def get_stopwords() -> Set[str]:
    """
    Get a set of stopwords.
    
    Returns:
        Set of stopwords.
    """
    if NLTK_AVAILABLE:
        try:
            return set(nltk.corpus.stopwords.words('english'))
        except:
            return set()
    else:
        # Basic English stopwords
        return {
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now'
        }


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from.
        max_keywords: Maximum number of keywords to extract.
        
    Returns:
        List of keywords.
    """
    if NLTK_AVAILABLE:
        try:
            # Simple keyword extraction using frequency distribution
            words = [word.lower() for word in nltk.word_tokenize(text) 
                  if word.isalpha() and word.lower() not in get_stopwords()]
            
            # Get frequency distribution
            fdist = nltk.FreqDist(words)
            
            # Get the most common words
            return [word for word, _ in fdist.most_common(max_keywords)]
        except:
            # Fallback to simple word frequency
            pass
    
    # Fallback implementation
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in get_stopwords()]
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text.
        text2: Second text.
        
    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if NLTK_AVAILABLE:
        try:
            # Tokenize texts
            words1 = set(w.lower() for w in nltk.word_tokenize(text1) 
                       if w.isalpha() and w.lower() not in get_stopwords())
            words2 = set(w.lower() for w in nltk.word_tokenize(text2) 
                       if w.isalpha() and w.lower() not in get_stopwords())
            
            # Calculate Jaccard similarity
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        except:
            # Fallback to simple implementation
            pass
    
    # Fallback implementation using sets of words
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0.0
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def estimate_reading_time(text: str) -> int:
    """
    Estimate reading time in seconds.
    
    Args:
        text: Text to estimate reading time for.
        
    Returns:
        Estimated reading time in seconds.
    """
    # Average reading speed: 200-250 words per minute
    # We'll use 225 wpm = 3.75 words per second
    if NLTK_AVAILABLE:
        words = len(nltk.word_tokenize(text))
    else:
        words = len(re.findall(r'\b\w+\b', text))
        
    return int(words / 3.75)


def find_section_boundaries(text: str) -> List[Tuple[int, int]]:
    """
    Find natural section boundaries in text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        List of (start, end) character indices for sections.
    """
    if not NLTK_AVAILABLE:
        # Fallback to simple paragraph splitting
        paragraphs = text.split('\n\n')
        
        start_idx = 0
        boundaries = []
        
        for para in paragraphs:
            if not para.strip():
                start_idx += len(para) + 2  # +2 for the newlines
                continue
                
            end_idx = start_idx + len(para)
            boundaries.append((start_idx, end_idx))
            start_idx = end_idx + 2  # +2 for the newlines
            
        return boundaries
    
    try:
        # Use NLTK to find paragraph and sentence boundaries
        sentences = nltk.sent_tokenize(text)
        
        start_idx = 0
        current_section_start = 0
        current_length = 0
        boundaries = []
        
        for sent in sentences:
            sent_idx = text.find(sent, start_idx)
            if sent_idx == -1:  # Safety check
                continue
                
            sent_end = sent_idx + len(sent)
            
            # Check if this is a potential section boundary
            next_char_idx = min(sent_end + 1, len(text) - 1)
            if next_char_idx < len(text) and text[next_char_idx] == '\n' and text[next_char_idx-1] in '.!?':
                # End of paragraph, create a section boundary
                boundaries.append((current_section_start, sent_end))
                current_section_start = sent_end + 1
                current_length = 0
            elif current_length > 2000:  # Prevent sections from getting too large
                # Create a boundary at this sentence
                boundaries.append((current_section_start, sent_end))
                current_section_start = sent_end + 1
                current_length = 0
            else:
                current_length += len(sent)
                
            start_idx = sent_end
        
        # Add the last section if there's text remaining
        if current_section_start < len(text):
            boundaries.append((current_section_start, len(text)))
            
        return boundaries
    except:
        # Fallback to simple paragraph splitting
        paragraphs = text.split('\n\n')
        
        start_idx = 0
        boundaries = []
        
        for para in paragraphs:
            if not para.strip():
                start_idx += len(para) + 2  # +2 for the newlines
                continue
                
            end_idx = start_idx + len(para)
            boundaries.append((start_idx, end_idx))
            start_idx = end_idx + 2  # +2 for the newlines
            
        return boundaries
