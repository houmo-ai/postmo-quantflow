import argparse

from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

SYSTEM_PROMPT = """\
You are a helpful AI assistant.

When external information is required:
- MUST call tools
- NEVER fabricate answers
"""

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }
]


def main(model_id: str) -> None:
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "上海今天的天气怎么样？"},
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="模型路径或名称",
    )
    args = parser.parse_args()
    main(args.model_id)
