import unittest

from advanced_rag.utils import (
    does_generation_answer_question,
    has_lexical_relevance,
    is_generation_grounded_in_context,
)


class UtilityTests(unittest.TestCase):
    def test_lexical_relevance_is_document_agnostic(self):
        question = "What warranty period is mentioned for the device?"
        document = "The device includes a two year warranty for manufacturing defects."

        self.assertTrue(has_lexical_relevance(question, document))

    def test_numeric_value_grounding_allows_formatting_differences(self):
        generation = "The invoice total is 1,250.00."
        context = "Invoice summary: total amount due is INR 1250.00 before Friday."

        self.assertTrue(is_generation_grounded_in_context(generation, context))

    def test_money_value_grounding_allows_omitted_zero_decimals(self):
        generation = "The base salary is 50,000."
        context = "Earnings table: Basic Pay 50,000.00"

        self.assertTrue(is_generation_grounded_in_context(generation, context))

    def test_generation_with_value_answers_question(self):
        question = "What is the order ID?"
        generation = "The order ID is 984512."

        self.assertTrue(does_generation_answer_question(question, generation))

    def test_unknown_answer_is_allowed_when_context_lacks_answer(self):
        generation = "I don't know based on the provided context."
        context = "The document contains only project dates."

        self.assertTrue(is_generation_grounded_in_context(generation, context))


if __name__ == "__main__":
    unittest.main()
