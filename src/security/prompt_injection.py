# Security utilities for detecting prompt injection and validating LLM outputs

import re

from loguru import logger

# Source: OWASP CheatSheetSeries - LLM Prompt Injection Prevention
# Classes were copied from OWASP repository.
# https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html#input-validation-and-sanitization


class PromptInjectionFilter:
    def __init__(self):
        self.dangerous_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
            r"system\s+override",
            r"reveal\s+prompt",
            r"disregard\s+(all\s+)?instructions?",
            r"forget\s+(your\s+)?system\s+prompt",
            r"new\s+instructions?",
            r"ignore\s+above",
            r"act\s+as\s+(a\s+)?different",
        ]

    def detect_injection(self, text: str) -> bool:
        for pattern in self.dangerous_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                logger.info("BLOCKED")
                return True

        logger.info("PASSED")
        return False

    def sanitize_input(self, text: str) -> str:
        # Normalize common obfuscations
        text = re.sub(r"\s+", " ", text)  # Collapse whitespace
        text = re.sub(r"(.)\1{3,}", r"\1", text)  # Remove char repetition

        for pattern in self.dangerous_patterns:
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
        return text[:10000]  # Limit length


class OutputValidator:
    def __init__(self):
        self.suspicious_patterns = [
            r"SYSTEM\s*[:]\s*You\s+are",  # System prompt leakage
            r"API[_\s]KEY[:=]\s*\w+",  # API key exposure
            r"instructions?[:]\s*\d+\.",  # Numbered instructions
        ]

    def validate_output(self, output: str) -> bool:
        is_valid = not any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in self.suspicious_patterns
        )
        logger.info("BLOCKED" if not is_valid else "PASSED")
        return is_valid

    def filter_response(self, response: str) -> str:
        is_valid = self.validate_output(response)
        is_too_long = len(response) > 5000

        if not is_valid or is_too_long:
            return "I cannot provide that information for security reasons."

        return response
