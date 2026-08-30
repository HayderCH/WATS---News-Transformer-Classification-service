from fastapi import APIRouter
from app.models.schemas import (
    SummarizationRequest,
    SummarizationResponse,
    SummarizationBatchRequest,
    SummarizationBatchResponse,
    SummarizationBatchItemResponse,
)
from app.services.summarizer import summarize_text

router = APIRouter()


@router.post("/summarize", response_model=SummarizationResponse)
def summarize(req: SummarizationRequest):
    result = summarize_text(req.text, max_len=req.max_len, min_len=req.min_len)
    return SummarizationResponse(**result)


@router.post("/summarize_batch", response_model=SummarizationBatchResponse)
def summarize_batch(req: SummarizationBatchRequest):
    outputs = []
    for it in req.items:
        r = summarize_text(it.text, max_len=it.max_len, min_len=it.min_len)
        outputs.append(SummarizationBatchItemResponse(**r))
    return SummarizationBatchResponse(results=outputs)
