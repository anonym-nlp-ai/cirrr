from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import ValidationError

from aimw.app.core.ai_config import get_ai_settings
from aimw.app.schemas.models.qas import QASet
from aimw.app.topology.graph.cir3_graph import Cir3Graph

router = APIRouter()


@router.post(
    path="/qag",
    response_model=QASet,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    response_description="The translated text.",
)
async def translate_text(doc: QASet) -> QASet:
    input_text = doc.context
    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input document cannot be empty.",
        )
    try:
        logger.info("Started QAG ...")
        cir3Graph = Cir3Graph()

        input = {
            "document": input_text,
            "num_steps": 0,
            "M": get_ai_settings().M,
            "N": get_ai_settings().N,
            "L": get_ai_settings().L,
            "K": get_ai_settings().K,
        }

        res = cir3Graph.topology.invoke(input=input)
        doc.qa_set = res["final_qas"]
        logger.debug(f"Final set of question-answer pairs: \n {doc.qa_set}")
        
        logger.info("QAG is successful")
    except ValidationError as e:
        error_msg = e.errors()[0]["msg"]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=error_msg
        ) from e
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occured during generation",
        ) from e

    return doc
