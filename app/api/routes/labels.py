from fastapi import APIRouter
from app.services.classifier import _classifier_holder


router = APIRouter()


@router.get("/labels")
def labels():
    _classifier_holder.load()
    labels_map = getattr(_classifier_holder, "label_names", {})
    # Return labels ordered by id when possible
    try:
        ordered = [labels_map[i] for i in sorted(labels_map)]
    except Exception:
        ordered = list(labels_map.values())
    return {"count": len(ordered), "labels": ordered}
