# Load model directly
#from transformers import AutoTokenizer, AutoModelForSequenceClassification

#tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
#model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

# Load model directly
# model.py
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
# The correct classes for audio classification (no tokenizer)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-4017")
model = AutoModelForAudioClassification.from_pretrained("facebook/mms-lid-4017")
device = "cuda" if torch.cuda.is_available() else "cpu"