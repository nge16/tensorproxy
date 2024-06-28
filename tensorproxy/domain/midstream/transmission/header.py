"""Информация о расчете."""

from datetime import datetime
from pydantic import BaseModel, Field


class Header(BaseModel):
    company: str | None = None
    author: str | None = None
    name: str | None = None
    date: datetime = Field(default_factory=datetime.now)
    version: str | None = None
    description: str | None = None
    methods: str | None = None
    json_schema: str | None = Field(None, serialization_alias="schema")
