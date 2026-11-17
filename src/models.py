"""Pydantic models for function definitions, prompts and outputs."""

from typing import Any

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """Type description of one function parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """One callable function described in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptItem(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """One result entry written to function_calls.json."""

    prompt: str
    name: str
    parameters: dict[str, Any]
