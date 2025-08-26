# Pull Requests

When you want to the modified code to be reflected in the repository, please execute a pull request.

## Procedures

- Please fork aiaccel repository on GitHub.
- After forking, run `git clone` command for aiaccel repository.

~~~bash
git clone https://github.com/[YOUR USERNAME]/aiaccel.git
~~~

## Developments
- Update a local repository to the latest version.

~~~bash
git checkout main
git pull upstream main
~~~

- Make a branch.

~~~bash
git checkout -b feature/add-new-feature
~~~

- Commit on local by using `git add` and `git commit` command as you progress.

  - The commit message describes the motivation for the change, the nature of the bug, or details the enhancement.
  - The message should be written in such a way that their contents can be  understood without refering code.


## Submitting

Before submitting a pull request, confirm the following:
- Did you discuss it with other developer on issues in advance?
- Can it be distributed under the MIT licence?
- Is there appropriate [unit tests](#test)?
- Can the [unit tests](#test) be run on local?
- Does the public function have a docstring?
- Can the [documentation](#documentation-wip) be rendered correctly?
- Is the [coding style](#coding-style) appropriate?
- Is the commit message appropriate?
- For larger commit, please provide the example (docs/source/examples) and the description of module level.
- If you are adding complied codes, have you modified setup.py?

After confirming above, do following:
- Push changes to the fork on GitHub.

~~~bash
git push origin feature/add-new-feature
~~~

- Enter your GitHub username and password.
- Move to the GitHub web page and write the title and message, noting the following.
  - Title
    - Briefly describe the changes.
    - Codes should be enclosed in backquotes.
    - Do not end with a period.
  - Descriptions
    - Write the motivation.
    - Write the changes.
    - If the related issues can be closed, please close it with `Close #N`.
    - If work-in-progress, write the remaining tasks.
- Submit the pull request.

# Review processes

- Other developers can contribute comments to improve implementations, documents, and coding styles in the pull request.
- When updating codes in the pull request, please commit the changes in the local repository and push the changes to the fork only if they have been successfully tested in the local environment.
- If the pull request has been reviewed and approved by at least one member of the aiaccel development team, it will be merged into the main branch.
