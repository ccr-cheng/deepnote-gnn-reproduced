from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .._base import register_model


@register_model('clinicalbert')
def get_clinicalbert(model_dir, use_classifier=True):
    # clinicalBERT is available at https://huggingface.co/AndyJ/clinicalBERT
    # keyword "model_type" = "bert" needs to be added to `config.json`
    if use_classifier:
        return AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        return AutoModel.from_pretrained(model_dir)
