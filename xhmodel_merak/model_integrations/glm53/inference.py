from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs


class _Glm53Adapter:
    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        from xhmodel_merak.xh_llm.models.glm5_3_flash.demo import generate_glm53

        values = validate_inputs(recipe, inputs)
        for name in ("image", "video"):
            if values.get(name):
                values[name] = str(require_local_file(recipe, name, values[name]))
        result = generate_glm53(meta_file, device=device, **values)
        return TextInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="text",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            **result,
        )


adapter = _Glm53Adapter()
