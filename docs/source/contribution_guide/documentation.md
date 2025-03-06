# Documentation (WIP)

## Docstrings

- Write a basic description of the implemented functions, the types and meanings of parameters and return values, and examples of their usage.
- Write in accordance with the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- See also [Coding Conventions](coding_styles).

## Documentation

- Create source files for documentation in a directory under docs.
- The recommended document file format is markdown format.
- Create documentation for any major feature additions.

## Confirming rendering

If you have added, changed, or modified documents, make sure that it renders correctly in the local environment.
Move to the aiaccel directory and execute the following command to generate an API reference.

~~~bash
cd aiaccel
sphinx-apidoc --maxdepth 2 -f -o ./docs/source/api_reference/ ./aiaccel/
~~~

Move to aiaccel/docs and build html files to see how the document is rendered.

~~~bash
cd docs
make html
~~~
