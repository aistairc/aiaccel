---
{{ card_data }}
---

# {{ model_name }}

{{ description | default("TODO: Describe the model.", true) }}

## Usage

TODO: Add usage instructions.

## Training

{% if base_model %}
Base model: [{{ base_model }}](https://huggingface.co/{{ base_model }}).

{% endif %}
TODO: Describe the training procedure.

## Limitations

TODO: Describe known limitations.

## Citation

TODO: Add a paper or citation if available.

## License

{% if license_name %}
This model is released under the **{{ license_name }}** license.
{% else %}
TODO: Specify the model license.
{% endif %}
