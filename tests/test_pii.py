from email_assistant.core.pii import PiiMasker, scrub_structure


def test_mask_and_unmask_roundtrip():
    text = "Mail john@acme.com or +1 (415) 555-0134, card 4111 1111 1111 1111, ssn 123-45-6789, john@acme.com"
    m = PiiMasker()
    masked = m.mask(text)
    assert "john@acme.com" not in masked and "4111" not in masked and "555-0134" not in masked
    assert masked.count("<EMAIL_1>") == 2  # stable placeholder per value
    assert m.unmask(masked) == text


def test_indian_identifiers_masked():
    masked = PiiMasker().mask("PAN ABCDE1234F aadhaar 1234 5678 9012")
    assert "ABCDE1234F" not in masked and "1234 5678 9012" not in masked


def test_non_luhn_numbers_and_iso_dates_are_kept():
    text = "order 1234567890123456 at 2025-11-03T10:00:00"
    assert PiiMasker().mask(text) == text


def test_redact_is_one_way():
    assert PiiMasker.redact("x@y.com") == "[REDACTED_EMAIL]"
    assert scrub_structure({"a": ["x@y.com"], "n": 5}) == {"a": ["[REDACTED_EMAIL]"], "n": 5}
