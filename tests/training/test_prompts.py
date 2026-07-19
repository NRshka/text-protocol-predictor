from pathlib import Path

from text_render_protocol_predictor.training import ProtocolPromptTemplate


def test_conversation_converts_path_for_transformers_processor() -> None:
    conversation = ProtocolPromptTemplate().conversation(
        image=Path("images/sample.png"),
        width=100,
        height=50,
        target="{}",
    )

    image_content = conversation[1]["content"][0]
    assert image_content == {"type": "image", "image": "images/sample.png"}


def test_prompt_requests_the_record_protocol_version() -> None:
    text = ProtocolPromptTemplate().user_text(100, 50, "2.1")
    assert "Use protocol version 2.1." in text
    assert "text and shape objects" in text
