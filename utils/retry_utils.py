# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Reusable retry decorator for Bedrock API calls and other operations.
"""

import backoff
from .logger import log_warning, log_error


def bedrock_retry(max_tries: int = 20, interval: int = 10):
    """
    Decorator that adds retry logic for Bedrock API calls.
    
    Retries on any exception with constant backoff, except missing packages and code errors.
    Logs retry attempts and final failures.
    
    Args:
        max_tries: Maximum number of retry attempts (default: 20)
        interval: Seconds between retry attempts (default: 10)
    
    Returns:
        Decorator function
    
    Example:
        @bedrock_retry()
        def my_bedrock_call(self):
            # Your Bedrock API call here
            pass
    """
    def decorator(func):
        return backoff.on_exception(
            backoff.constant,
            Exception,  # Catch all exceptions
            max_tries=max_tries,
            # ... except those a retry cannot fix (missing package, bug in the code)
            giveup=lambda e: isinstance(e, (ImportError, NameError, SyntaxError)),
            interval=interval,
            on_backoff=lambda details: log_warning(
                f"{func.__qualname__} error (attempt {details['tries']}/{max_tries}): {details['exception']}. "
                f"Retrying in {details['wait']:.0f} seconds..."
            ),
            on_giveup=lambda details: log_error(
                f"{func.__qualname__} failed after {details['tries']} attempts. Last error: {details['exception']}"
            )
        )(func)
    return decorator
