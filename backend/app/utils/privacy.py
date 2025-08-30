import re
import hashlib
from typing import Dict, List, Any
import json

class PIIRedactor:
    def __init__(self):
        # PII detection patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'url': r'https?://[^\s]+',
            'windows_path': r'[C-Z]:\\\\[^\\s]+',
            'unix_path': r'/[^\\s]+',
        }
        
        # Replacement patterns (deterministic)
        self.replacements = {
            'email': '[EMAIL_REDACTED]',
            'phone': '[PHONE_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'ip_address': '[IP_REDACTED]',
            'credit_card': '[CARD_REDACTED]',
            'url': '[URL_REDACTED]',
            'windows_path': '[PATH_REDACTED]',
            'unix_path': '[PATH_REDACTED]',
        }
    
    def redact_text(self, text: str) -> Dict[str, Any]:
        """
        Redact PII from text and return both redacted text and redaction log
        """
        if not text:
            return {"redacted_text": text, "redactions": []}
        
        redacted_text = text
        redactions = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, redacted_text)
            for match in matches:
                original_value = match.group()
                replacement = self.replacements[pii_type]
                
                # Log the redaction (but not the original value)
                redactions.append({
                    "type": pii_type,
                    "position": match.start(),
                    "length": len(original_value),
                    "replacement": replacement
                })
                
                # Replace the PII
                redacted_text = redacted_text.replace(original_value, replacement)
        
        return {
            "redacted_text": redacted_text,
            "redactions": redactions
        }
    
    def redact_dict(self, data: Dict[str, Any], fields_to_redact: List[str] = None) -> Dict[str, Any]:
        """
        Redact PII from dictionary values
        """
        if fields_to_redact is None:
            fields_to_redact = ['title', 'description', 'comment', 'content']
        
        redacted_data = data.copy()
        
        for field in fields_to_redact:
            if field in redacted_data and isinstance(redacted_data[field], str):
                result = self.redact_text(redacted_data[field])
                redacted_data[field] = result["redacted_text"]
        
        return redacted_data

class DeterminismUtils:
    @staticmethod
    def generate_deterministic_seed(input_data: str) -> int:
        """Generate deterministic seed from input data"""
        return int(hashlib.md5(input_data.encode()).hexdigest()[:8], 16)
    
    @staticmethod
    def sort_list_deterministically(items: List[Any], key_func=None) -> List[Any]:
        """Sort list in a deterministic way"""
        if key_func:
            return sorted(items, key=key_func)
        
        # For mixed types, convert to string for comparison
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=str)
    
    @staticmethod
    def normalize_dict_for_hashing(data: Dict[str, Any]) -> str:
        """
        Normalize dictionary to ensure consistent hashing regardless of key order
        """
        # Recursively sort all nested dictionaries
        def normalize_value(value):
            if isinstance(value, dict):
                return {k: normalize_value(v) for k, v in sorted(value.items())}
            elif isinstance(value, list):
                return [normalize_value(item) for item in value]
            else:
                return value
        
        normalized = normalize_value(data)
        return json.dumps(normalized, sort_keys=True)
    
    @staticmethod
    def calculate_content_hash(content: str) -> str:
        """Calculate deterministic hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def ensure_deterministic_ordering(
        items: List[Dict[str, Any]], 
        primary_key: str = 'id',
        secondary_key: str = None
    ) -> List[Dict[str, Any]]:
        """
        Ensure deterministic ordering of results
        """
        def sort_key(item):
            primary = item.get(primary_key, 0)
            if secondary_key:
                secondary = item.get(secondary_key, '')
                return (primary, secondary)
            return primary
        
        return sorted(items, key=sort_key)

class ReproducibilityValidator:
    def __init__(self):
        self.pii_redactor = PIIRedactor()
        self.determinism_utils = DeterminismUtils()
    
    def validate_deterministic_output(
        self, 
        input_data: Dict[str, Any], 
        output_data: Dict[str, Any],
        previous_output: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate that the same input produces the same output
        """
        validation_result = {
            "is_deterministic": True,
            "input_hash": self.determinism_utils.calculate_content_hash(
                self.determinism_utils.normalize_dict_for_hashing(input_data)
            ),
            "output_hash": self.determinism_utils.calculate_content_hash(
                self.determinism_utils.normalize_dict_for_hashing(output_data)
            ),
            "errors": []
        }
        
        if previous_output:
            previous_hash = self.determinism_utils.calculate_content_hash(
                self.determinism_utils.normalize_dict_for_hashing(previous_output)
            )
            
            if validation_result["output_hash"] != previous_hash:
                validation_result["is_deterministic"] = False
                validation_result["errors"].append("Output hash differs from previous run with same input")
        
        return validation_result
    
    def create_replay_signature(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """
        Create a signature for replay validation
        """
        combined_data = {
            "input": input_data,
            "output": output_data,
            "timestamp": output_data.get("timestamp", datetime.now().isoformat())
        }
        
        normalized = self.determinism_utils.normalize_dict_for_hashing(combined_data)
        return self.determinism_utils.calculate_content_hash(normalized)
