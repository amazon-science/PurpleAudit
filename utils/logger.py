#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Simple logging utility with colored output
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict


class Colors:
    """ANSI color codes for terminal output"""
    # Status colors
    SUCCESS = '\033[92m'     # Green
    ERROR = '\033[91m'       # Red
    WARNING = '\033[93m'     # Yellow
    INFO = '\033[96m'        # Cyan
    
    # Reset
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def _get_timestamp() -> str:
    """Get formatted timestamp for logging"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_success(message: str, **kwargs):
    """Print success message"""
    timestamp = _get_timestamp()
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.SUCCESS}✅ {message}{Colors.RESET}")


def log_error(message: str, **kwargs):
    """Print error message"""
    timestamp = _get_timestamp()
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.ERROR}❌ {message}{Colors.RESET}")


def log_warning(message: str, **kwargs):
    """Print warning message"""
    timestamp = _get_timestamp()
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.WARNING}⚠️  {message}{Colors.RESET}")


def log_info(message: str, **kwargs):
    """Print info message"""
    timestamp = _get_timestamp()
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.INFO}  {message}{Colors.RESET}")


def log_separator(title: str = None, **kwargs):
    """Print separator"""
    timestamp = _get_timestamp()
    if title:
        print(f"{Colors.DIM}[{timestamp}]{Colors.RESET}\n{Colors.BOLD}{'─' * 20} {title} {'─' * 20}{Colors.RESET}")
    else:
        print(f"{Colors.DIM}[{timestamp}] {'─' * 60}{Colors.RESET}")

# JSON file writing functions - for data persistence
def log_json_conversation(name: str, role: str, content: str, model: str = None, json_file: str = None):
    """Log conversation to JSON file (for data persistence)"""
    timestamp = _get_timestamp()
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.INFO}🤖 {name}: {content[:100]}...{Colors.RESET}")
    
    # Save to JSON file for local persistence
    if json_file:
        log_entry = {
            "role": role,
            "name": name,
            "content": content
        }
        
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"messages": []}
            
            data["messages"].append(log_entry)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            timestamp = _get_timestamp()
            print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.ERROR}Error writing JSON: {e}{Colors.RESET}")


def log_json_results(result: dict, json_file: str = None):
    """Log results to JSON file (for data persistence)"""
    if json_file:
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"messages": []}
            
            if "results" not in data:
                data["results"] = []
            data["results"].append(result)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            timestamp = _get_timestamp()
            print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.ERROR}Error writing results JSON: {e}{Colors.RESET}")
