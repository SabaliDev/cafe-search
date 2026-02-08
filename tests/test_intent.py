from app.intent import classify_intent


def test_explicit_intent():
    result = classify_intent("data science jobs")
    assert result.intent in {"explicit", "inferred"}


def test_company_intent():
    result = classify_intent("companies with strong mentorship")
    assert result.intent == "company"


def test_inferred_intent():
    result = classify_intent("something in climate tech")
    assert result.intent == "inferred"
