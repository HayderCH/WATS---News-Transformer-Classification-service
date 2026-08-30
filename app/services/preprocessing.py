import re
from sentence_transformers import SentenceTransformer

_whitespace_re = re.compile(r"\s+")


class Preprocessor:
    def __init__(self):
        self._model = None

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def basic_clean(self, text: str) -> str:
        text = text.strip()
        text = _whitespace_re.sub(" ", text)
        return text

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        model = self.get_model()
        return model.encode(texts).tolist()


# For backward compatibility
def basic_clean(text: str) -> str:
    return Preprocessor().basic_clean(text)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    return Preprocessor().get_embeddings(texts)
