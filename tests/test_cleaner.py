from email_assistant.preprocessing.cleaner import (
    clean_body, html_to_text, strip_signature, truncate,
)
from email_assistant.preprocessing.mime_parser import parse_raw_email
from fakes import make_raw


def test_html_removes_hidden_quotes_and_keeps_structure():
    html = (
        '<div style="display:none">preheader</div><p>Hi</p><p>See <b>Tuesday</b><br>3pm</p>'
        "<ul><li>one</li></ul><blockquote>old</blockquote>"
    )
    text = html_to_text(html)
    assert "preheader" not in text and "old" not in text and "- one" in text and "Tuesday" in text


def test_quote_and_signature_stripped():
    raw = (
        "Sounds good.\n\nBest regards,\nAlice\nVP Sales\n+1 555 123 4567\n\n"
        "On Mon, Mar 3, 2025 at 9:00 AM Bob <b@x.com> wrote:\n> Can we meet?\n"
    )
    out = clean_body(raw, is_html=False)
    assert out.startswith("Sounds good") and "Alice" not in out and "Can we meet" not in out


def test_rfc3676_and_mobile_signatures():
    assert strip_signature("Review please.\n\n-- \nJohn\nCTO").strip() == "Review please."
    assert "iPhone" not in clean_body("Yes.\n\nSent from my iPhone", is_html=False)


def test_footer_removed_and_never_empty():
    body = "Great deal on shoes today only! " * 8 + "\n\nYou are receiving this email because you signed up."
    assert "receiving" not in clean_body(body, is_html=False)
    fwd = "From: Boss <b@x.com>\nSent: Monday\nTo: me\nSubject: hi\n\nPlease handle this"
    assert clean_body(fwd, is_html=False)


def test_truncate_keeps_head_and_tail():
    out = truncate("a" * 500 + "z" * 500, 100)
    assert out.startswith("a") and out.endswith("z") and "omitted" in out


def test_parse_raw_email_extracts_context_and_flags():
    raw = make_raw(
        sender="Promo <noreply@shop.com>",
        subject="Sale!",
        body="Buy now",
        headers={"List-Unsubscribe": "<mailto:u@shop.com>", "Precedence": "bulk"},
    )
    parsed = parse_raw_email(raw.raw, gmail_id="m1", thread_id="t1", own_email="dana@example.com")
    assert parsed.sender_email == "noreply@shop.com" and parsed.sender_domain == "shop.com"
    assert parsed.has_list_unsubscribe and parsed.is_bulk and parsed.is_automated
    assert not parsed.is_from_self and parsed.date.year == 2025


def test_parse_detects_self_sent_and_internal_sender():
    raw = make_raw(sender="Me <alice@acme.com>", subject="note to self", body="x")
    parsed = parse_raw_email(raw.raw, gmail_id="m1", thread_id="t1", own_email="alice@acme.com")
    assert parsed.is_from_self and parsed.is_internal
