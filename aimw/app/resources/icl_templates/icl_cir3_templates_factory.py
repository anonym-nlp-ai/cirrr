from aimw.app.resources.icl_templates import icl_cir3_templates
from aimw.app.schemas.enum.ai_enums import Role

prompt_templates = {
    Role.CLASSIFIER: icl_cir3_templates.classifier_prompt,
    Role.MODERATOR: icl_cir3_templates.moderator_prompt,
    Role.WRITER: icl_cir3_templates.writers_refinement_prompt,
    Role.CURMUDGEON: icl_cir3_templates.curmudgeon_prompt,
}
