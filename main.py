import torch
from utils import ContentType, get_text_model_messages
from transformers import AutoModelForSpeechSeq2Seq, AutoModel, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

asr_model_id = "openai/whisper-large-v3"

asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    asr_model_id, 
    dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
asr_model.to(device)

text_model_id = "openai/gpt-oss-20b"

text_model = AutoModel.from_pretrained(
    text_model_id
)

asr_processor = AutoProcessor.from_pretrained(asr_model_id)

pipe1 = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

pipe2 = pipeline(
    "text-generation",
    model=text_model_id,
    dtype="auto",
    device=device,
    max_new_tokens=256
)


dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

asr_result = pipe1(sample)
transcription = asr_result["text"]
print('Transcription:')
print(transcription)
print()

abstraction = pipe2(get_text_model_messages(ContentType.ABSTRACTION, transcription))
print('Abstraction:')
print(abstraction[0]["generated_text"][-1]["content"])
print()

keypoints = pipe2(get_text_model_messages(ContentType.KEY_POINTS, transcription))
print('Keypoints')
print(keypoints[0]["generated_text"][-1]["content"])
print()

action_items = pipe2(get_text_model_messages(ContentType.ACTION_ITEMS, transcription))
print('Action Items:')
print(action_items[0]["generated_text"][-1]["content"])
print()

sentiment = pipe2(get_text_model_messages(ContentType.SENTIMENT, transcription))
print('Sentiment:')
print(sentiment[0]["generated_text"][-1]["content"])
print()