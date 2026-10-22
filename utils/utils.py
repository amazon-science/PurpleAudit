# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

def extract_project_name(results: dict):
    for c in results["chat_history"]:
        if "file_name=" in c["content"]:
            return c["content"].split("environment.json")[0].split("/")[-2]
    
    return None
