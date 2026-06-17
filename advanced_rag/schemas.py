"""Structured output schemas for routing and grading."""

from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for groundedness check on generated answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score to assess whether the answer addresses the question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: str = Field(
        ...,
        description="Given a user question choose to route it to websearch or vectorstore.",
    )
