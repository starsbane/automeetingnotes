# AutoMeetingNotes

## Description
This project is a POC that automate the process of generating abstractions, key points, action items and sentiment of meeting from audio recordings by using local OpenAI Whisper 3 and OpenAI OSS model.

## Requirements
No OpenAI key required, but your device should meet system requirements of [OpenAI Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) and [OpenAI GPT OSS 20b](https://huggingface.co/openai/gpt-oss-20b). This project was developed using Python 3.14.2.

Create a virtual environment and install the requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to use
```bash
python -m venv venv
source venv/bin/activate
python main.py
```
## License
[MIT](https://choosealicense.com/licenses/mit/)
