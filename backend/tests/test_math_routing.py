from mcp_math_client import classify_query, extract_numeric_values


def test_classify_math_query() -> None:
    assert classify_query("What is 345 × 879?") == "math"
    assert classify_query("Calculate the average marks from this document.") == "mixed"
    assert classify_query("Explain this topic in the uploaded PDF.") == "rag"


def test_extract_numeric_values() -> None:
    context = "Employee salaries: 1200, 1500, and 1800 dollars."
    assert extract_numeric_values(context) == [1200, 1500, 1800]
