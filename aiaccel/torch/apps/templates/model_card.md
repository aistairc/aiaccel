---
{{ card_data }}
---

# {{ model_name }}

{{ description | default("TODO: Describe the model.", true) }}

## Usage

TODO: Add usage instructions.

## Training

TODO: Describe the training procedure.

## Limitations

TODO: Describe known limitations.

## Citation

TODO: Replace this placeholder with the model's BibTeX citation.

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
This model is based on the following models:

{% for model in base_models %}
- {% if model.url %}[{{ model.name }}]({{ model.url }}){% else %}{{ model.name }}{% endif %} — {{ model.license or "TODO: Specify the base model license." }}
{% endfor %}
{% endif %}
