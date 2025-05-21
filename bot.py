import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import requests
import json
from dotenv import load_dotenv
import re
from collections import deque
import difflib
from typing import List, Set, Dict, Tuple
import random
import uuid
import time
import asyncio
import telegram
from openai import OpenAI
import aiohttp

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get sensitive data from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
NOVITA_API_KEY = os.getenv('NOVITA_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
MIDNIGHT_ROSE_API_KEY = os.getenv('MIDNIGHT_ROSE_API_KEY')

# Validate required environment variables
if not all([TELEGRAM_BOT_TOKEN, NOVITA_API_KEY, DEEPSEEK_API_KEY, MIDNIGHT_ROSE_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Initialize OpenAI clients for different APIs
novita_client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key=NOVITA_API_KEY
)

deepseek_client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key=DEEPSEEK_API_KEY
)

midnight_rose_client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key=MIDNIGHT_ROSE_API_KEY
)

# British English word mappings
BRITISH_WORDS = {
    'color': 'colour',
    'center': 'centre',
    'theater': 'theatre',
    'realize': 'realise',
    'analyze': 'analyse',
    'organize': 'organise',
    'recognize': 'recognise',
    'customize': 'customise',
    'specialize': 'specialise',
    'standardize': 'standardise',
    'apologize': 'apologise',
    'memorize': 'memorise',
    'emphasize': 'emphasise',
    'minimize': 'minimise',
    'maximize': 'maximise',
    'optimize': 'optimise',
    'criticize': 'criticise',
    'summarize': 'summarise',
    'visualize': 'visualise',
    'stabilize': 'stabilise',
    'utilize': 'utilise',
    'authorize': 'authorise',
    'categorize': 'categorise',
    'characterize': 'characterise',
    'civilize': 'civilise',
    'commercialize': 'commercialise',
    'computerize': 'computerise',
    'criminalize': 'criminalise',
    'decentralize': 'decentralise',
    'democratize': 'democratise',
    'dramatize': 'dramatise',
    'economize': 'economise',
    'energize': 'energise',
    'equalize': 'equalise',
    'familiarize': 'familiarise',
    'formalize': 'formalise',
    'generalize': 'generalise',
    'globalize': 'globalise',
    'harmonize': 'harmonise',
    'hospitalize': 'hospitalise',
    'idealize': 'idealise',
    'industrialize': 'industrialise',
    'institutionalize': 'institutionalise',
    'internalize': 'internalise',
    'legalize': 'legalise',
    'legitimize': 'legitimise',
    'liberalize': 'liberalise',
    'localize': 'localise',
    'marginalize': 'marginalise',
    'materialize': 'materialise',
    'mechanize': 'mechanise',
    'memorialize': 'memorialise',
    'mobilize': 'mobilise',
    'modernize': 'modernise',
    'monopolize': 'monopolise',
    'moralize': 'moralise',
    'nationalize': 'nationalise',
    'naturalize': 'naturalise',
    'neutralize': 'neutralise',
    'normalize': 'normalise',
    'paralyze': 'paralyse',
    'penalize': 'penalise',
    'personalize': 'personalise',
    'pluralize': 'pluralise',
    'polarize': 'polarise',
    'popularize': 'popularise',
    'privatize': 'privatise',
    'publicize': 'publicise',
    'rationalize': 'rationalise',
    'recognize': 'recognise',
    'regularize': 'regularise',
    'reorganize': 'reorganise',
    'revitalize': 'revitalise',
    'revolutionize': 'revolutionise',
    'ritualize': 'ritualise',
    'romanticize': 'romanticise',
    'sanitize': 'sanitise',
    'satirize': 'satirise',
    'scrutinize': 'scrutinise',
    'sensationalize': 'sensationalise',
    'sensitize': 'sensitise',
    'serialize': 'serialise',
    'socialize': 'socialise',
    'stabilize': 'stabilise',
    'standardize': 'standardise',
    'sterilize': 'sterilise',
    'stigmatize': 'stigmatise',
    'subsidize': 'subsidise',
    'summarize': 'summarise',
    'symbolize': 'symbolise',
    'synchronize': 'synchronise',
    'synthesize': 'synthesise',
    'systematize': 'systematise',
    'temporize': 'temporise',
    'terrorize': 'terrorise',
    'theorize': 'theorise',
    'traumatize': 'traumatise',
    'trivialize': 'trivialise',
    'tyrannize': 'tyrannise',
    'unionize': 'unionise',
    'urbanize': 'urbanise',
    'vandalize': 'vandalise',
    'vaporize': 'vaporise',
    'visualize': 'visualise',
    'vitalize': 'vitalise',
    'vocalize': 'vocalise',
    'vulgarize': 'vulgarise',
    'westernize': 'westernise',
    'womanize': 'womanise',
}

# Common misspellings and their corrections
COMMON_MISSPELLINGS = {
    'recieve': 'receive',
    'seperate': 'separate',
    'occured': 'occurred',
    'accomodate': 'accommodate',
    'existance': 'existence',
    'persistant': 'persistent',
    'refered': 'referred',
    'occassion': 'occasion',
    'commited': 'committed',
    'embarass': 'embarrass',
    'existance': 'existence',
    'persistant': 'persistent',
    'refered': 'referred',
    'occassion': 'occasion',
    'commited': 'committed',
    'embarass': 'embarrass',
    'existance': 'existence',
    'persistant': 'persistent',
    'refered': 'referred',
    'occassion': 'occasion',
    'commited': 'committed',
    'embarass': 'embarrass',
}

# List of allowed usernames (replace with your desired usernames)
ALLOWED_USERNAMES = [
     "MJay_07",  # Replace with your actual username
  #  "Michael4420" ,
     "soullord65" ,
     "gabi08099"# Add more usernames as needed
]

# File to store message history
HISTORY_FILE = "sent_messages.txt"

# File to store reactivation messages
REACTIVATION_MESSAGES_FILE = "reactivation_messages.txt"
USED_REACTIVATION_MESSAGES_FILE = "used_reactivation_messages.txt"

# Global message history to store all sent messages
sent_messages = []

# Global message history dictionary to store per-user messages
message_history = {}

# Global deque to store recent bot replies (last 5)
recent_replies = deque(maxlen=5)

# Store input-response pairs to prevent duplicate responses to same input
input_response_pairs: Dict[str, str] = {}

# Global set to track used reactivation messages
used_reactivation_messages = {
    "Sweet": set(),
    "Needy": set(),
    "Dirty": set(),
    "Bratty": set()
}

# Similarity threshold for considering messages too similar (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.85

# Common patterns to avoid
COMMON_PATTERNS = [
    r"i (love|enjoy|like) .*",
    r"what about you\?",
    r"tell me about .*",
    r"how about .*\?",
    r"i'm (glad|happy) .*",
    r"that's (interesting|great|amazing) .*",
    r"i (think|believe) .*",
    r"would you like to .*\?",
    r"let's .*",
    r"i (wonder|wish) .*"
]

# Rate limiting configuration
RATE_LIMIT_DELAY = 2.0  # Base delay between requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for rate limit errors
MAX_BACKOFF = 30  # Maximum backoff time in seconds

# Message memory configuration
MESSAGE_MEMORY_SIZE = 1000
message_memory = deque(maxlen=MESSAGE_MEMORY_SIZE)

# Common intro phrases to remove (case-insensitive)
INTRO_PHRASES = [
    r'^oh\s+darlings?\b',
    r'^my\s+loves?\b',
    r'^sweethearts?\b',
    r'^babes?\b',
    r'^honeys?\b',
    r'^dears?\b',
    r'^lovelies?\b',
    r'^angels?\b',
    r'^beauties?\b',
    r'^sweeties?\b',
    r'^darlings?\b',
    r'^sweeties?\b',
    r'^honeys?\b',
    r'^babes?\b',
    r'^dears?\b',
]

# Compile regex patterns for intro phrases
INTRO_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INTRO_PHRASES]

# Categories for reactivation messages
REACTIVATION_CATEGORIES = {
    "Sweet": [],
    "Needy": [],
    "Dirty": [],
    "Bratty": []
}

# Phrase memory management
PHRASE_COOLDOWN_PERIOD = 10  # Number of messages before a phrase can be used again
PHRASE_FREQUENCY_LIMIT = 3   # Maximum times a phrase can be used within cooldown period
PHRASE_BLACKLIST = set()     # Set of phrases that are temporarily blacklisted

# Dictionary to track phrase usage: {phrase: {'count': int, 'last_used': int}}
phrase_usage = {}

# Dictionary to track message numbers for cooldown
message_counter = 0

# Add after the other global variables
BLACKLIST_FILE = "blacklisted_phrases.txt"

# Add after the other global variables
PHRASE_USAGE_FILE = "phrase_usage.json"

# Add after other global variables
RESPONSE_MEMORY_FILE = "response_memory.json"
RESPONSE_MEMORY_SIZE = 100  # Number of recent responses to keep
response_memory = []  # List to store recent responses
response_patterns = {}  # Dictionary to track response patterns

def save_phrase_usage():
    """Save phrase usage data to file."""
    try:
        # Convert the phrase_usage dictionary to a serializable format
        serializable_data = {
            phrase: {
                'count': stats['count'],
                'last_used': stats['last_used']
            }
            for phrase, stats in phrase_usage.items()
        }
        
        with open(PHRASE_USAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2)
        logger.info(f"Saved phrase usage data for {len(phrase_usage)} phrases")
        
    except Exception as e:
        logger.error(f"Error saving phrase usage: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")

def load_phrase_usage():
    """Load phrase usage data from file."""
    global phrase_usage, message_counter
    try:
        if os.path.exists(PHRASE_USAGE_FILE):
            with open(PHRASE_USAGE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                phrase_usage = {
                    phrase: {
                        'count': stats['count'],
                        'last_used': stats['last_used']
                    }
                    for phrase, stats in data.items()
                }
            logger.info(f"Loaded phrase usage data for {len(phrase_usage)} phrases")
            
            # Update message counter to be higher than any last_used value
            if phrase_usage:
                message_counter = max(stats['last_used'] for stats in phrase_usage.values()) + 1
        else:
            logger.info("No phrase usage file found. Starting with empty tracking.")
            phrase_usage = {}
            message_counter = 0
            
    except Exception as e:
        logger.error(f"Error loading phrase usage: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        phrase_usage = {}
        message_counter = 0

def update_phrase_usage(phrase: str):
    """Update the usage statistics for a phrase."""
    global message_counter, phrase_usage, PHRASE_BLACKLIST
    
    # Clean and normalize the phrase
    phrase = clean_message(phrase)
    
    # Skip if phrase is too short
    if len(phrase.split()) < 3:
        return
        
    # Initialize phrase tracking if not exists
    if phrase not in phrase_usage:
        phrase_usage[phrase] = {'count': 0, 'last_used': 0}
    
    # Update usage statistics
    phrase_usage[phrase]['count'] += 1
    phrase_usage[phrase]['last_used'] = message_counter
    
    # Check if phrase should be blacklisted
    if phrase_usage[phrase]['count'] >= PHRASE_FREQUENCY_LIMIT:
        PHRASE_BLACKLIST.add(phrase)
        logger.info(f"Phrase blacklisted due to overuse: {phrase[:50]}...")
    
    # Clean up old entries
    cleanup_phrase_usage()
    
    # Save updated usage data
    save_phrase_usage()

def cleanup_phrase_usage():
    """Remove old phrase usage entries and expired blacklist entries."""
    global phrase_usage, PHRASE_BLACKLIST
    
    # Remove phrases from blacklist if their cooldown period has passed
    phrases_to_remove = set()
    for phrase in PHRASE_BLACKLIST:
        if phrase in phrase_usage:
            if message_counter - phrase_usage[phrase]['last_used'] >= PHRASE_COOLDOWN_PERIOD:
                phrases_to_remove.add(phrase)
                phrase_usage[phrase]['count'] = 0
    
    PHRASE_BLACKLIST -= phrases_to_remove
    
    # Remove old entries from phrase_usage
    phrases_to_clean = []
    for phrase, stats in phrase_usage.items():
        if message_counter - stats['last_used'] >= PHRASE_COOLDOWN_PERIOD:
            phrases_to_clean.append(phrase)
    
    for phrase in phrases_to_clean:
        del phrase_usage[phrase]

def is_phrase_allowed(phrase: str) -> bool:
    """Check if a phrase is allowed to be used based on cooldown and blacklist."""
    phrase = clean_message(phrase)
    
    # Skip check for very short phrases
    if len(phrase.split()) < 3:
        return True
    
    # Check if phrase is blacklisted
    if phrase in PHRASE_BLACKLIST:
        return False
    
    # Check if phrase is in cooldown
    if phrase in phrase_usage:
        if message_counter - phrase_usage[phrase]['last_used'] < PHRASE_COOLDOWN_PERIOD:
            return False
    
    return True

def extract_phrases_from_response(response: str) -> List[str]:
    """Extract meaningful phrases from a response."""
    # Split response into sentences
    sentences = re.split(r'[.!?]+', response)
    phrases = []
    
    for sentence in sentences:
        # Clean and normalize the sentence
        sentence = clean_message(sentence)
        
        # Skip very short sentences
        if len(sentence.split()) < 3:
            continue
            
        # Add the sentence as a phrase
        phrases.append(sentence)
        
        # Also add 3-5 word combinations
        words = sentence.split()
        for i in range(len(words) - 2):
            for j in range(3, min(6, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+j])
                if len(phrase.split()) >= 3:
                    phrases.append(phrase)
    
    return phrases

def sanitize_input(text):
    """Sanitize user input to prevent injection attacks."""
    # Remove any potentially harmful characters
    sanitized = re.sub(r'[<>]', '', text)
    # Limit message length
    return sanitized[:1000]

def load_message_history():
    """Load message history and reactivation messages from files."""
    global sent_messages, recent_replies, message_history, used_reactivation_messages
    try:
        # Load regular message history
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                sent_messages = [line.strip() for line in f if line.strip()]
                # Initialize recent_replies with the last 5 messages
                recent_replies = deque(sent_messages[-5:], maxlen=5)
                # Initialize message_history as empty dictionary
                message_history = {}
                logger.info(f"Loaded {len(sent_messages)} messages from history file")
                logger.info(f"Initialized recent_replies with {len(recent_replies)} messages")
        else:
            sent_messages = []
            recent_replies = deque(maxlen=5)
            message_history = {}
            logger.info("No history file found, starting with empty history")

        # Initialize used_reactivation_messages as a dictionary of sets
        used_reactivation_messages = {
            "Sweet": set(),
            "Needy": set(),
            "Dirty": set(),
            "Bratty": set()
        }

    except Exception as e:
        logger.error(f"Error loading message history: {str(e)}")
        sent_messages = []
        recent_replies = deque(maxlen=5)
        message_history = {}
        used_reactivation_messages = {
            "Sweet": set(),
            "Needy": set(),
            "Dirty": set(),
            "Bratty": set()
        }

def save_message_history():
    """Save message history and reactivation messages to files."""
    try:
        # Save regular message history
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            for message in sent_messages:
                f.write(f"{message}\n")
            logger.info(f"Saved {len(sent_messages)} messages to history file")

        # Save reactivation message history
        with open(USED_REACTIVATION_MESSAGES_FILE, 'w', encoding='utf-8') as f:
            for category, messages in used_reactivation_messages.items():
                for message in messages:
                    f.write(f"{category}|{message}\n")
            logger.info(f"Saved used reactivation messages to history file")

    except Exception as e:
        logger.error(f"Error saving message history: {str(e)}")

def get_similarity_ratio(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def extract_phrases(text: str) -> Set[str]:
    """Extract meaningful phrases from text."""
    # Remove common words and split into phrases
    words = text.lower().split()
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
    phrases = set()
    
    # Extract 3-5 word phrases (increased from 2-4)
    for i in range(len(words) - 2):
        phrase = ' '.join(words[i:i+3])
        if not any(word in common_words for word in phrase.split()):
            phrases.add(phrase)
    for i in range(len(words) - 3):
        phrase = ' '.join(words[i:i+4])
        if not any(word in common_words for word in phrase.split()):
            phrases.add(phrase)
    for i in range(len(words) - 4):
        phrase = ' '.join(words[i:i+5])
        if not any(word in common_words for word in phrase.split()):
            phrases.add(phrase)
            
    return phrases

def check_pattern_similarity(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the common patterns."""
    text = text.lower()
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

def is_message_unique(message: str, user_input: str) -> bool:
    """Enhanced check if a message is unique compared to previously sent messages and recent replies."""
    message = message.lower().strip()
    user_input = user_input.lower().strip()
    message_phrases = extract_phrases(message)
    
    # Check if this is a reactivation message
    if "[Please reactivate the user!]" in user_input:
        # Check if this exact reactivation message has been used before
        if message in used_reactivation_messages["Sweet"] or message in used_reactivation_messages["Needy"] or message in used_reactivation_messages["Dirty"] or message in used_reactivation_messages["Bratty"]:
            logger.info("Reactivation message has been used before")
            return False
            
        # Add to used reactivation messages and save immediately
        used_reactivation_messages["Sweet"].add(message)
        used_reactivation_messages["Needy"].add(message)
        used_reactivation_messages["Dirty"].add(message)
        used_reactivation_messages["Bratty"].add(message)
        save_message_history()  # Save after adding new reactivation message
        logger.info(f"Added new reactivation message to history. Total reactivation messages: {len(used_reactivation_messages['Sweet'])}")
    
    # Check similarity with other reactivation messages
    for category in used_reactivation_messages:
        for used_message in used_reactivation_messages[category]:
            if get_similarity_ratio(message, used_message) > SIMILARITY_THRESHOLD:
                logger.info("Reactivation message too similar to previous ones")
                return False
    
    # Check if we've already responded to this exact input
    if user_input in input_response_pairs:
        if message == input_response_pairs[user_input].lower().strip():
            logger.info("Exact response found for this input")
            return False
    
    # Check for pattern similarity
    if check_pattern_similarity(message, COMMON_PATTERNS):
        logger.info("Message matches common pattern")
        return False
    
    # Check against all sent messages
    for sent_message in sent_messages:
        # Check exact match
        if message == sent_message.lower().strip():
            logger.info("Exact match found in sent messages")
            return False
            
        # Check similarity ratio
        if get_similarity_ratio(message, sent_message) > SIMILARITY_THRESHOLD:
            logger.info("High similarity found in sent messages")
            return False
            
        # Check phrase overlap with increased threshold
        sent_phrases = extract_phrases(sent_message)
        overlap = message_phrases.intersection(sent_phrases)
        if len(overlap) > 3:  # Increased from 2 to 3
            logger.info("Too many similar phrases found in sent messages")
            return False
    
    # Check against recent replies
    for recent_reply in recent_replies:
        # Check exact match
        if message == recent_reply.lower().strip():
            logger.info("Exact match found in recent replies")
            return False
            
        # Check similarity ratio
        if get_similarity_ratio(message, recent_reply) > SIMILARITY_THRESHOLD:
            logger.info("High similarity found in recent replies")
            return False
            
        # Check phrase overlap with increased threshold
        recent_phrases = extract_phrases(recent_reply)
        overlap = message_phrases.intersection(recent_phrases)
        if len(overlap) > 3:  # Increased from 2 to 3
            logger.info("Too many similar phrases found in recent replies")
            return False
            
    return True

def is_user_allowed(username):
    """Check if the user is allowed to use the bot."""
    return username in ALLOWED_USERNAMES

def build_prompt(model_name: str) -> str:
    """Build the system prompt for the specified model."""
    try:
        if model_name == "Deepseek":
            with open("deepseek_prompt.txt", "r") as f:
                return f.read()
        elif model_name == "Llama":
            with open("llama_prompt.txt", "r") as f:
                return f.read()
        elif model_name == "Midnight Rose":
            with open("midnight_rose_prompt.txt", "r") as f:
                return f.read()
        else:
            logger.error(f"Unknown model: {model_name}")
            return ""
    except Exception as e:
        logger.error(f"Error loading prompt for {model_name}: {str(e)}")
        return ""

async def generate_unique_response(user_message: str, model_name: str) -> str:
    """Generate a unique response using the specified model."""
    try:
        # Build the prompt for this specific model
        system_prompt = build_prompt(model_name)
        if not system_prompt:
            logger.error(f"Failed to build prompt for {model_name}")
            return ""

        # Create message array for this model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Get response from the model
        response = await get_model_response(messages, model_name)
        if not response:
            logger.error(f"Failed to get response from {model_name}")
            return ""

        # Check if response is too similar to recent ones
        if is_response_too_similar(response):
            logger.warning(f"Response from {model_name} too similar to recent ones, trying again")
            return await generate_unique_response(user_message, model_name)

        # Add to memory
        add_to_memory(response)
        return response

    except Exception as e:
        logger.error(f"Error generating response from {model_name}: {str(e)}")
        return ""

def is_response_relevant(response: str, user_message: str) -> bool:
    """Check if the response is relevant to the user's message."""
    # Convert both to lowercase for comparison
    response_lower = response.lower()
    user_message_lower = user_message.lower()
    
    # Check for key terms in user message
    key_terms = [
        "favorite", "favourite", "love", "like", "enjoy", "adore",
        "beautiful", "gorgeous", "sexy", "hot", "stunning", "amazing",
        "wonderful", "fantastic", "brilliant", "perfect", "lovely"
    ]
    
    # If user message contains any key terms, check if response acknowledges them
    if any(term in user_message_lower for term in key_terms):
        # Check if response contains any of the same key terms or related terms
        response_terms = [
            "favorite", "favourite", "love", "like", "enjoy", "adore",
            "beautiful", "gorgeous", "sexy", "hot", "stunning", "amazing",
            "wonderful", "fantastic", "brilliant", "perfect", "lovely",
            "appreciate", "cherish", "treasure", "value", "mean", "special"
        ]
        return any(term in response_lower for term in response_terms)
    
    # If no key terms found, check for general relevance
    # Split messages into words and check for common words
    user_words = set(user_message_lower.split())
    response_words = set(response_lower.split())
    common_words = user_words.intersection(response_words)
    
    # If there are enough common words, consider it relevant
    return len(common_words) >= 2

def format_messages_for_ai(raw_message: str) -> str:
    """
    Format the raw message into the required structure for the AI.
    The raw message contains both the customer message and operator's last message.
    """
    # Split the message into lines
    lines = raw_message.split('\n')
    
    # Initialize variables
    customer_message = ""
    operator_message = ""
    current_section = None
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip metadata lines
        if any(pattern in line.lower() for pattern in ['perm_identity', 'report', 'by another agent']):
            continue
            
        # Skip timestamp lines
        if re.match(r'\d{2}/\d{2}/\d{4}', line):
            continue
            
        # If this is the first non-metadata line, it's the customer message
        if not customer_message and not operator_message:
            customer_message = line
        # Otherwise, it's the operator message
        else:
            operator_message = line
    
    # Format the final message
    formatted_message = f"""Previous operator: {operator_message}

Customer: {customer_message}

Rules: Generate a flirty, engaging reply that fits the topic. Always ask a question. Never suggest meeting in person. Avoid repeating earlier responses."""

    return formatted_message

def load_reactivation_messages():
    """Load available and used reactivation messages from files."""
    global REACTIVATION_CATEGORIES, used_reactivation_messages
    try:
        # Initialize categories
        REACTIVATION_CATEGORIES = {
            "Sweet": [],
            "Needy": [],
            "Dirty": [],
            "Bratty": []
        }
        
        # Initialize used messages dictionary
        used_reactivation_messages = {
            "Sweet": set(),
            "Needy": set(),
            "Dirty": set(),
            "Bratty": set()
        }
        
        # Debug: Print the full content of the file
        logger.info("Reading reactivation messages file...")
        with open(REACTIVATION_MESSAGES_FILE, 'r', encoding='utf-8') as f:
            file_content = f.read()
            logger.info(f"File content length: {len(file_content)} characters")
        
        current_category = None
        with open(REACTIVATION_MESSAGES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Debug: Log each line being processed
                logger.debug(f"Processing line: {line[:50]}...")
                
                # Check if this is a category header
                if line in REACTIVATION_CATEGORIES:
                    current_category = line
                    logger.info(f"Found category: {current_category}")
                elif current_category and line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', 
                                                         '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.')):
                    # Remove the number and dot from the start
                    message = line[line.find('.')+1:].strip()
                    if message:  # Only add non-empty messages
                        REACTIVATION_CATEGORIES[current_category].append(message)
                        logger.info(f"Added message to category {current_category}: {message[:50]}...")
        
        # Debug: Print the contents of each category
        for category in REACTIVATION_CATEGORIES:
            logger.info(f"Category {category} has {len(REACTIVATION_CATEGORIES[category])} messages:")
            for msg in REACTIVATION_CATEGORIES[category]:
                logger.info(f"  - {msg[:50]}...")
        
        # Load used messages from file if not already loaded
        if os.path.exists(USED_REACTIVATION_MESSAGES_FILE):
            with open(USED_REACTIVATION_MESSAGES_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            # Try new format first (category|message)
                            if '|' in line:
                                category, message = line.split('|', 1)
                                if category in used_reactivation_messages:
                                    used_reactivation_messages[category].add(message)
                            else:
                                # Handle old format (just message)
                                # Add to all categories to ensure no repetition
                                for category in REACTIVATION_CATEGORIES:
                                    used_reactivation_messages[category].add(line)
                        except Exception as e:
                            logger.warning(f"Error processing used message line: {line}. Error: {str(e)}")
                            continue
        
        # Log the number of messages loaded per category
        for category in REACTIVATION_CATEGORIES:
            logger.info(f"Loaded {len(REACTIVATION_CATEGORIES[category])} messages for category {category}")
            logger.info(f"Loaded {len(used_reactivation_messages[category])} used messages for category {category}")
            
            # If no messages were loaded for a category, log a warning
            if len(REACTIVATION_CATEGORIES[category]) == 0:
                logger.warning(f"No messages loaded for category {category}!")
        
        return True
    except Exception as e:
        logger.error(f"Error loading reactivation messages: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        return False

def load_blacklist():
    """Load manually blacklisted phrases from file."""
    global PHRASE_BLACKLIST
    try:
        logger.info(f"Attempting to load blacklist from {BLACKLIST_FILE}")
        
        # Initialize PHRASE_BLACKLIST if it doesn't exist
        if PHRASE_BLACKLIST is None:
            PHRASE_BLACKLIST = set()
            logger.info("Initialized empty PHRASE_BLACKLIST")
        
        if os.path.exists(BLACKLIST_FILE):
            with open(BLACKLIST_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    phrase = line.strip()
                    if phrase:
                        PHRASE_BLACKLIST.add(phrase)
                        logger.info(f"Loaded blacklisted phrase: {phrase}")
            logger.info(f"Successfully loaded {len(PHRASE_BLACKLIST)} blacklisted phrases")
        else:
            logger.info(f"Blacklist file {BLACKLIST_FILE} does not exist yet. It will be created when first phrase is added.")
            # Create an empty file to ensure it exists
            with open(BLACKLIST_FILE, 'w', encoding='utf-8') as f:
                pass
            logger.info(f"Created empty blacklist file: {BLACKLIST_FILE}")
        
    except Exception as e:
        logger.error(f"Error loading blacklist: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        # Initialize empty blacklist on error
        PHRASE_BLACKLIST = set()

def save_blacklist():
    """Save manually blacklisted phrases to file."""
    try:
        # Create the file if it doesn't exist
        with open(BLACKLIST_FILE, 'w', encoding='utf-8') as f:
            for phrase in sorted(PHRASE_BLACKLIST):
                f.write(f"{phrase}\n")
        logger.info(f"Saved {len(PHRASE_BLACKLIST)} blacklisted phrases to {BLACKLIST_FILE}")
        
        # Verify the file was created and has content
        if os.path.exists(BLACKLIST_FILE):
            with open(BLACKLIST_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Verified blacklist file contents: {content}")
        else:
            logger.error(f"Failed to create blacklist file: {BLACKLIST_FILE}")
        
    except Exception as e:
        logger.error(f"Error saving blacklist: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")

async def blacklist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /blacklist command."""
    username = update.effective_user.username
    if not is_user_allowed(username):
        logger.warning(f"Unauthorized access attempt by user: {username}")
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return

    # Check if a phrase was provided
    if not context.args:
        # Show current blacklist
        if PHRASE_BLACKLIST:
            blacklist_text = "Current blacklisted phrases:\n\n"
            for phrase in sorted(PHRASE_BLACKLIST):
                blacklist_text += f"• {phrase}\n"
            await update.message.reply_text(blacklist_text)
        else:
            await update.message.reply_text("No phrases are currently blacklisted.")
        return

    # Get the phrase to blacklist
    phrase = ' '.join(context.args)
    phrase = clean_message(phrase)

    # Check if phrase is too short
    if len(phrase.split()) < 3:
        await update.message.reply_text("Please provide a phrase with at least 3 words.")
        return

    # Add to blacklist
    PHRASE_BLACKLIST.add(phrase)
    logger.info(f"Adding phrase to blacklist: {phrase}")
    save_blacklist()
    
    # Verify the phrase was added
    if phrase in PHRASE_BLACKLIST:
        await update.message.reply_text(f"Added to blacklist: {phrase}")
        logger.info(f"Successfully added phrase to blacklist: {phrase}")
    else:
        await update.message.reply_text("Failed to add phrase to blacklist. Please try again.")
        logger.error(f"Failed to add phrase to blacklist: {phrase}")

async def unblacklist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /unblacklist command."""
    username = update.effective_user.username
    if not is_user_allowed(username):
        logger.warning(f"Unauthorized access attempt by user: {username}")
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return

    # Check if a phrase was provided
    if not context.args:
        await update.message.reply_text("Please provide a phrase to remove from the blacklist.")
        return

    # Get the phrase to unblacklist
    phrase = ' '.join(context.args)
    phrase = clean_message(phrase)

    # Remove from blacklist
    if phrase in PHRASE_BLACKLIST:
        PHRASE_BLACKLIST.remove(phrase)
        save_blacklist()
        await update.message.reply_text(f"Removed from blacklist: {phrase}")
    else:
        await update.message.reply_text("That phrase is not in the blacklist.")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries from inline keyboard."""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("reactivate_"):
        category = query.data.replace("reactivate_", "")
        reactivation_message = get_reactivation_message(category)
        await query.edit_message_text(text=reactivation_message)

async def get_model_response(messages: List[Dict[str, str]], model_name: str) -> str:
    """Get a response from the specified model."""
    try:
        response = await get_ai_response(messages[-1]["content"], None)  # Pass None as user_id since it's not needed here
        if response and isinstance(response, str) and is_valid_response(response):
            # Apply grammar correction
            response = correct_grammar(response)
            return response
        return None
    except Exception as e:
        logger.error(f"Error from {model_name}: {str(e)}")
        return None

def load_response_memory():
    """Load response memory from file."""
    global response_memory, response_patterns
    try:
        if os.path.exists(RESPONSE_MEMORY_FILE):
            with open(RESPONSE_MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                response_memory = data.get('responses', [])
                response_patterns = data.get('patterns', {})
            logger.info(f"Loaded {len(response_memory)} responses and {len(response_patterns)} patterns from memory")
        else:
            logger.info("No response memory file found. Starting with empty memory.")
            response_memory = []
            response_patterns = {}
            
    except Exception as e:
        logger.error(f"Error loading response memory: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        response_memory = []
        response_patterns = {}

def save_response_memory():
    """Save response memory to file."""
    try:
        data = {
            'responses': response_memory,
            'patterns': response_patterns
        }
        with open(RESPONSE_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(response_memory)} responses and {len(response_patterns)} patterns to memory")
        
    except Exception as e:
        logger.error(f"Error saving response memory: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")

def add_to_memory(response: str):
    """Add a response to memory and update patterns."""
    global response_memory, response_patterns
    
    # Add response to memory
    response_memory.append(response)
    if len(response_memory) > RESPONSE_MEMORY_SIZE:
        response_memory.pop(0)
    
    # Extract and update patterns
    words = response.lower().split()
    for i in range(len(words) - 2):
        pattern = ' '.join(words[i:i+3])
        response_patterns[pattern] = response_patterns.get(pattern, 0) + 1
    
    # Save memory after update
    save_response_memory()

def is_response_too_similar(response: str) -> bool:
    """Check if a response is too similar to recent ones or contains overused patterns."""
    # Check against recent responses
    for recent_response in response_memory[-10:]:  # Check last 10 responses
        if get_similarity_ratio(response, recent_response) > SIMILARITY_THRESHOLD:
            logger.info("Response too similar to recent one")
            return True
    
    # Check for overused patterns
    words = response.lower().split()
    for i in range(len(words) - 2):
        pattern = ' '.join(words[i:i+3])
        if response_patterns.get(pattern, 0) > 5:  # Pattern used more than 5 times
            logger.info("Response contains overused pattern")
            return True
    
    return False

def get_reactivation_message(category: str) -> str:
    """Get a random reactivation message from the specified category."""
    try:
        if category not in REACTIVATION_CATEGORIES:
            logger.error(f"Invalid category: {category}")
            return "Sorry, I couldn't find a message for that category."
            
        available_messages = [
            msg for msg in REACTIVATION_CATEGORIES[category]
            if msg not in used_reactivation_messages[category]
        ]
        
        if not available_messages:
            # If all messages have been used, reset the used messages for this category
            used_reactivation_messages[category].clear()
            available_messages = REACTIVATION_CATEGORIES[category]
            logger.info(f"Reset used messages for category {category}")
        
        # Select a random message
        message = random.choice(available_messages)
        
        # Add to used messages
        used_reactivation_messages[category].add(message)
        
        # Save the updated used messages
        save_message_history()
        
        return message
        
    except Exception as e:
        logger.error(f"Error getting reactivation message: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        return "Sorry, I encountered an error while getting your message."

def main():
    """Start the bot."""
    try:
        logger.info("Starting bot...")
        
        # Load message history and reactivation messages
        load_message_history()
        if not load_reactivation_messages():
            logger.error("Failed to load reactivation messages!")
            return
            
        # Load blacklist, phrase usage data, and response memory
        load_blacklist()
        load_phrase_usage()
        load_response_memory()
        
        # Create the Application and pass it your bot's token
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("models", show_models))
        application.add_handler(CommandHandler("blacklist", blacklist_command))
        application.add_handler(CommandHandler("unblacklist", unblacklist_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(handle_callback))

        # Add error handler for Conflict error
        async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle errors."""
            logger.error(f"Update {update} caused error {context.error}")
            if isinstance(context.error, telegram.error.Conflict):
                logger.error("Bot conflict detected. Attempting to restart...")
                # Wait a moment before restarting
                await asyncio.sleep(5)
                # Restart the bot
                await application.stop()
                await application.start()
                await application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

        application.add_error_handler(error_handler)

        # Start the Bot with error handling
        logger.info("Bot is ready and listening for messages...")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        
    except telegram.error.Conflict as e:
        logger.error(f"Bot conflict error: {str(e)}")
        logger.info("Attempting to restart bot...")
        # Wait a moment before restarting
        time.sleep(5)
        main()  # Restart the bot
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

# Add after the handle_callback function
async def show_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show details about the models."""
    try:
        # Model details
        models = {
            "Novita AI (Llama 3.3)": {
                "model_name": "meta-llama/llama-3.3-70b-instruct",
                "api_base": "https://api.novita.ai/v3/openai",
                "temperature": 0.7,
                "max_tokens": 1000,
                "api_type": "Novita AI",
                "response_format": "text"
            },
            "Deepseek": {
                "model_name": "deepseek-chat",
                "api_base": "https://api.deepseek.com/v1",
                "temperature": 0.7,
                "max_tokens": 1000,
                "api_type": "Deepseek"
            }
        }
        
        # Format the response
        response = "=== Model Details ===\n\n"
        for model_name, details in models.items():
            response += f"{model_name}:\n"
            for key, value in details.items():
                response += f"{key}: {value}\n"
            response += "\n"
            
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error showing model details: {str(e)}")
        await update.message.reply_text("Sorry, I encountered an error while fetching model details.")

# Add after the show_models function
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the user message and generate responses from all three models."""
    try:
        # Check if user is allowed
        username = update.effective_user.username
        if not is_user_allowed(username):
            logger.warning(f"Unauthorized access attempt by user: {username}")
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
            return

        # Get user ID
        user_id = update.effective_user.id

        # Sanitize user input
        user_message = sanitize_input(update.message.text)
        logger.info(f"Received message from user {username}: {user_message}")
        
        # Check if this is a reactivation request
        if "[Please reactivate the user!]" in user_message:
            # Create inline keyboard with categories
            keyboard = []
            for category in REACTIVATION_CATEGORIES.keys():
                keyboard.append([InlineKeyboardButton(category, callback_data=f"reactivate_{category}")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Choose a category for your reactivation message:",
                reply_markup=reply_markup
            )
            return
        
        # Format the message for AI
        formatted_message = format_messages_for_ai(user_message)
        logger.info(f"Formatted message for AI: {formatted_message}")
        
        # Send typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Get responses from all three models simultaneously
        async def get_model_response_with_client(model_name: str, client: OpenAI) -> Tuple[str, str]:
            try:
                response = await get_ai_response(formatted_message, user_id)
                if response and isinstance(response, str) and is_valid_response(response):
                    # Apply grammar correction
                    response = correct_grammar(response)
                    
                    # If this is a reactivation message, add it to the tracking set
                    if "[Please reactivate the user!]" in user_message:
                        used_reactivation_messages[category].add(response.lower().strip())
                        save_message_history()
                        logger.info(f"Added reactivation message to history. Total reactivation messages: {len(used_reactivation_messages[category])}")
                    
                    return model_name, response
                return model_name, None
            except Exception as e:
                logger.error(f"Error from {model_name}: {str(e)}")
                return model_name, None

        # Try all three models simultaneously
        tasks = [
            get_model_response_with_client("deepseek-v3", deepseek_client),
            get_model_response_with_client("meta-llama/llama-3.1-405b-instruct", novita_client),
            get_model_response_with_client("sophosympatheia/midnight-rose-70b", midnight_rose_client)
        ]
        
        # Wait for all models to respond
        results = await asyncio.gather(*tasks)
        
        # Send responses from all models
        for model_name, response in results:
            if response:
                # Format the response with model name
                formatted_response = f"🤖 {model_name}:\n{response}"
                await update.message.reply_text(formatted_response)
            else:
                await update.message.reply_text(f"❌ {model_name} failed to generate a response.")
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        await update.message.reply_text("I'm sorry, I encountered an error. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    username = update.effective_user.username
    if not is_user_allowed(username):
        logger.warning(f"Unauthorized access attempt by user: {username}")
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return

    welcome_message = (
        "👋 Welcome to the AI Chat Bot!\n\n"
        "I'm here to chat with you using multiple AI models. Here are some commands you can use:\n\n"
        "/help - Show this help message\n"
        "/models - Show details about the AI models\n"
        "/blacklist - View or add phrases to the blacklist\n"
        "/unblacklist - Remove phrases from the blacklist\n\n"
        "Just send me a message and I'll respond using different AI models!"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    username = update.effective_user.username
    if not is_user_allowed(username):
        logger.warning(f"Unauthorized access attempt by user: {username}")
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return

    help_message = (
        "🤖 Bot Commands:\n\n"
        "/start - Start the bot and see the welcome message\n"
        "/help - Show this help message\n"
        "/models - Show details about the AI models being used\n"
        "/blacklist - View current blacklisted phrases or add new ones\n"
        "/unblacklist - Remove phrases from the blacklist\n\n"
        "📝 Usage Tips:\n"
        "- Send any message to get responses from multiple AI models\n"
        "- Each model will respond with its own unique style\n"
        "- The bot will avoid repeating similar responses\n"
        "- Blacklisted phrases will be avoided in responses\n\n"
        "❓ Need more help? Just ask!"
    )
    await update.message.reply_text(help_message)

async def get_ai_response(message: str, user_id: int = None) -> str:
    """Get a response from the AI model."""
    try:
        # Determine which client and model to use
        if "deepseek" in message.lower():
            client = deepseek_client
            model = "deepseek-chat"
            system_prompt = build_prompt("Deepseek")
        elif "llama" in message.lower():
            client = novita_client
            model = "meta-llama/llama-3.3-70b-instruct"
            system_prompt = build_prompt("Llama")
        else:
            client = midnight_rose_client
            model = "sophosympatheia/midnight-rose-70b"
            system_prompt = build_prompt("Midnight Rose")

        # Log the system prompt for debugging
        logger.info(f"Using system prompt for {model}: {system_prompt[:100]}...")

        # Create the chat completion request with the model-specific system prompt
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "text"},
            stop=["</assistant>", "<|im_end|>", "<|endoftext|>", "✨", "️", "💫", "🌟", "⭐", "💝", "💖", "💕", "💓", "💗", "💘", "💞", "💟", "#", "@"]
        )

        # Extract and clean the response
        if response and response.choices:
            response_text = response.choices[0].message.content.strip()
            
            # Remove any common intro phrases
            for pattern in INTRO_PATTERNS:
                response_text = pattern.sub('', response_text).strip()
            
            # Remove all emoji and emoji-like characters
            response_text = re.sub(r'[\U0001F300-\U0001F9FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]', '', response_text)
            
            # Remove scare quotes
            response_text = re.sub(r'"([^"]*)"', r'\1', response_text)
            
            # Remove hashtags and mentions
            response_text = re.sub(r'[#@]\w+', '', response_text)
            
            # Convert to British English
            for american, british in BRITISH_WORDS.items():
                response_text = re.sub(r'\b' + american + r'\b', british, response_text, flags=re.IGNORECASE)
            
            # Fix common misspellings
            for wrong, correct in COMMON_MISSPELLINGS.items():
                response_text = re.sub(r'\b' + wrong + r'\b', correct, response_text, flags=re.IGNORECASE)
            
            # Ensure response is in English only and contains no special characters
            if not all(ord(c) < 128 for c in response_text) or re.search(r'[^\w\s.,!?\'"-]', response_text):
                logger.warning(f"Non-English characters or special characters detected in response: {response_text}")
                # Try to get a new response
                return await get_ai_response(message, user_id)
            
            # Remove any remaining special characters
            response_text = re.sub(r'[^\w\s.,!?\'"-]', '', response_text)
            
            # Remove multiple spaces
            response_text = re.sub(r'\s+', ' ', response_text)
            
            # Remove leading/trailing spaces
            response_text = response_text.strip()
            
            return response_text

        return None

    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        return None

def is_valid_response(response: str) -> bool:
    """Check if a response is valid and meets our criteria."""
    if not response or not isinstance(response, str):
        return False
        
    # Check minimum length
    if len(response.split()) < 3:
        return False
        
    # Check for question mark (should ask a question)
    if '?' not in response:
        return False
        
    # Check for blacklisted phrases
    for phrase in PHRASE_BLACKLIST:
        if phrase.lower() in response.lower():
            return False
            
    return True

def clean_message(text: str) -> str:
    """Clean and normalize a message."""
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation except for question marks
    text = re.sub(r'[^\w\s?]', '', text)
    
    return text

def correct_grammar(text: str) -> str:
    """Correct grammar using the LanguageTool public API (synchronous fallback)."""
    try:
        url = "https://api.languagetool.org/v2/check"
        data = {
            "text": text,
            "language": "en-GB"
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            matches = result.get("matches", [])
            # Apply corrections from the end to the start
            corrected = text
            offset_shift = 0
            for match in sorted(matches, key=lambda m: m["offset"], reverse=True):
                offset = match["offset"]
                length = match["length"]
                replacement = match["replacements"][0]["value"] if match["replacements"] else None
                if replacement:
                    corrected = corrected[:offset] + replacement + corrected[offset+length:]
            return corrected
        else:
            return text
    except Exception as e:
        logger.error(f"Grammar correction failed: {e}")
        return text

if __name__ == '__main__':
    main() 