from pathlib import Path
from enum import Enum

class ContentType(Enum):
    ABSTRACTION = 1,
    KEY_POINTS = 2,
    ACTION_ITEMS = 3,
    SENTIMENT = 4

def read_content(type: ContentType):
    filename = ''
    match type:
        case ContentType.ABSTRACTION:
            filename = 'abstraction.txt'
        case ContentType.SENTIMENT:
            filename = 'sentiment.txt'
        case ContentType.KEY_POINTS:
            filename = 'keypoints.txt'
        case ContentType.ACTION_ITEMS:
            filename = 'actionitems.txt'
        case _:
            raise 'Unsupported content type'
    
    content = Path(f"messages/{filename}").read_text(encoding="utf-8")
    return content

def get_text_model_messages(type: ContentType, user_message: str):
    return [
        {"role": "system", "content": read_content(type)},
        {"role": "user", "content": user_message}
    ]
