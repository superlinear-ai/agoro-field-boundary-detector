# Agoro Field Boundary Detector

Detect field boundaries using satellite imagery.


## Installation

To install this package in your environment, run:

```bash
pip install git+https://github.com/radix-ai/agoro-field-boundary-detector.git
```


## Usage
```python
from agoro_field_boundary_detector import FieldBoundaryDetectorInterface

# Load in the model, will start GEE session
model = FieldBoundaryDetectorInterface(model_path=...)

# Make the prediction
pred = model(lat=39.6679328199836, lng=-95.4287818841267)

# Result
# [
#     (39.683761135289785, -95.4369042849299),
#     (39.6837431689841, -95.43695096539429), 
#     ...
#     (39.683761135289785, -95.4368809446977), 
#     (39.683761135289785, -95.4369042849299),
# ]
```


## Development

### Setup Environment

### Exporting images

### Data Annotation

### Data Augmentation

### Model Training

### Model inference







### Development environment setup

<details>
<summary>Once per machine</summary>

1. [Generate an SSH key](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair) for GitLab, [add the SSH key to GitLab](https://docs.gitlab.com/ee/ssh/README.html#adding-an-ssh-key-to-your-gitlab-account), and [add the SSH key to your authentication agent](https://docs.gitlab.com/ee/ssh/README.html#working-with-non-default-ssh-key-pair-paths).
2. Install [Docker](https://www.docker.com/get-started).
3. Install [VS Code](https://code.visualstudio.com/).
4. Install [VS Code's Remote-Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
5. Install [Fira Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions).

</details>

<details open>
<summary>Once per repository</summary>

You can set up your development environment as a self-contained [development container](https://code.visualstudio.com/docs/remote/containers) with a single step. In VS Code, press <kbd>⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd>, select _Remote-Containers: Clone Repository in Container Volume.. and enter:

```
git@gitlab.com:radix-ai.git
```

Alternatively, if you prefer to install your environment locally, run `./tasks/init.sh` from VS Code's Terminal.
</details>

### Common tasks

<details>
<summary>Activating the Python environment</summary>

1. Open any Python file in the project to load VS Code's Python extension.
2. Open an integrated Terminal with <kbd>⌃</kbd> + <kbd>~</kbd> and you should see that the conda environment `agoro-field-boundary-detector-env` is active.
3. Now you're ready to run any of tasks listed by `invoke --list`.

</details>

<details>
<summary>Running and debugging tests</summary>

1. Activate the Python environment.
2. If you don't see _⚡ Run tests_ in the blue bar, run <kbd>⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> > _Python: Discover Tests_. Optionally debug the output in _View_ > _Output_ > _Python Test Log_ in case this step fails.
3. Go to any test function in `src/tests/pytest`.
4. Optional: put a breakpoint 🔴 next to the line number where you want to stop.
5. Click on _Run Test_ or _Debug Test_ above the test you want to debug.

</details>

<details>
<summary>Updating the Cookiecutter scaffolding</summary>

1. Activate the Python environment.
2. Run `cruft check` to check for updates.
3. Run `cruft update` to update to the latest scaffolding.
4. Address failed merges in any `.rej` files.

</details>

<details>
<summary>Contributing code</summary>

You are responsible for the full lifecycle to get your code integerated into `master`:

1. Create a new branch from `master`.<sup>1</sup>
2. Push your branch and create an MR with prefix `WIP:`.<sup>2,3</sup>
3. Rebase on `master` with `git pull --rebase origin master` before requesting a review.
4. Request a review on Slack. [Mention someone](https://slack.com/intl/en-be/help/articles/205240127-Use-mentions-in-Slack#mention-someone) if no one takes action.
5. Address the comments and ask the reviewer to validate that they are resolved. Repeat until there are no unresolved comments.
6. Rebase on `master` with `git pull --rebase origin master`.
7. Bump the version with `invoke bump [patch|minor|major]`.<sup>4</sup>
8. Merge the MR and ensure that your branch is deleted.

Notes:

1. Prefix your branch name with a [Jira issue key](https://support.atlassian.com/jira-software-cloud/docs/what-is-an-issue/#Workingwithissues-Projectandissuekeys), or your initials if there is no related Jira issue.
2. The `WIP:` prefix indicates that the MR is still a Work In Progress.
3. A good commit message completes the sentence [If applied, this commit will ..](https://chris.beams.io/posts/git-commit/).
4. Use [Semantic Versioning](https://semver.org/) to decide whether you bump the patch, minor, or major version.

</details>
