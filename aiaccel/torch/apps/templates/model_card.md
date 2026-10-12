---
{{ card_data }}
---

# {{ model_name }}

{{ description | default("TODO: Describe the model.", true) }}

## Usage

You can run inference with the following command:

```bash
# TODO: Replace with the actual inference command.
python path/to/inference.py --model path/to/model
```

## Training

TODO: Describe the training procedure.

## Limitations

TODO: Describe known limitations.

## Citation

If you use this model in your research, please cite the following work:

```bibtex
@misc{TODO,
  title = {TODO: Model title},
  author = {TODO: Authors},
  year = {TODO: Year},
  url = {TODO: Publication or model URL}
}
```

## License

{% if license_name %}
This model is released under the **{{ license_name }}** license.
{% else %}
TODO: Specify the model license.
{% endif %}

{% if base_models %}
Note that this model is built on the following base models and is also subject to their respective license terms:

{% for model in base_models -%}
- {% if model.url %}[{{ model.name }}]({{ model.url }}){% else %}{{ model.name }}{% endif %} — {{ model.license or "TODO: Specify the base model license." }}
{% endfor -%}
{% endif %}
