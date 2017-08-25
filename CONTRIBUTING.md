# Contributing Guidelines

Hello! We're glad you're interested in contributing to CellProfiler.

This document provides guidelines for contributing to our project. You'll find information on what you can contribute, choosing what to contribute, setting up your local workstation for development, and submitting your contribution for review and acceptance.

We'd like to ensure this document is always accurate and understandable. Please [file an issue](https://github.com/CellProfiler/CellProfiler/issues/new) if any information is missing, unclear, or incorrect.  

## Creating an Issue

The CellProfiler team uses [GitHub issues](https://github.com/CellProfiler/CellProfiler/issues) to track upcoming and ongoing work. Examples of issues are:

* Documentation additions and corrections
* Errors starting or running CellProfiler
* Unexpected behavior or output
* Feature requests (e.g., new modules)

When there is an issue you would like to file, please check that the issue does not already exist. You can search open issues by including the default search filters "is: issue is: open". Once you are certain no issue exists, [create a new issue](https://github.com/CellProfiler/CellProfiler/issues/new) while keeping the following guidelines in mind:

1. Give your issue a descriptive title. A good title summarizes the issue in a single sentence. Be specific. Avoid general titles such as "Add documentation" or "Error running CellProfiler".

1. Provide specific information about your issue.
  * Documentation: What is the mistake? Where is the mistake? What is the correct documentation?
  * Errors: What is the exact error? How did it occur? Please provide error details from the GUI or output from the terminal.
  * Unexpected behavior: What behavior did you observe? How did it occur? What behavior do you expect?
  * Feature requests: What is the feature? Why would you like to have this functionality in CellProfiler?

1. Click "Submit new issue". After an issue is created, CellProfiler maintainers will add appropriate labels to categorize your issue. The maintainers may also ask clarifying questions and engage you and the greater community in discussion of your issue. Please be ready to engage in the discussion.

## Choosing an Issue

CellProfiler GitHub issues are categorized using labels. We encourage new contributors to check out issues with the  ["Easy"](https://github.com/CellProfiler/CellProfiler/issues?utf8=%E2%9C%93&q=is%3Aopen%20is%3Aissue%20label%3A%22Easy%22%20) label or ["Documentation"](https://github.com/CellProfiler/CellProfiler/issues?utf8=%E2%9C%93&q=is%3Aopen%20is%3Aissue%20label%3A%22Documentation%22%20) label. Issues with these labels will help you gain familiarity with the CellProfiler code base and can usually be resolved in an afternoon. Issues labeled ["Bug"](https://github.com/CellProfiler/CellProfiler/issues?utf8=%E2%9C%93&q=is%3Aopen%20is%3Aissue%20label%3A%22Bug%22%20) are great for contributors looking for a chance to dive into CellProfiler internals and strengthen their Python and debugging abilities.

## Development Environment

We recommend using the [PyCharm](https://www.jetbrains.com/pycharm/) editor. The Community Edition is available for free and contains all the features needed for easy CellProfiler development.

## Getting Started

To submit changes to CellProfiler, you'll need to have a [GitHub](https://github.com/) account with a copy of the CellProfiler repository. These instructions will help you copy the repository to your GitHub account, create a local copy of your repository, and keep your repository up to date.

1. Fork CellProfiler. Click the "Fork" button near the upper right of the [CellProfiler project page](https://github.com/CellProfiler/CellProfiler) to create a copy of the CellProfiler repository in your GitHub user account.

1. Create a local clone of your fork. Open your terminal and run the following commands in the directory you wish to clone your copy of the CellProfiler repository:

  ```
  $ git clone https://github.com/YOUR-USERNAME/CellProfiler
  $ cd CellProfiler
  ```

  Substitute `YOUR-USERNAME` with your GitHub user name.

1. Configure Git to sync your fork with the original CellProfiler repository. This will help you keep your copy up to date with ours. Use the command `git remote add` to add the original CellProfiler repository as a remote repository called upstream:

  ```
  $ git remote add upstream https://github.com/CellProfiler/CellProfiler
  ```

1. Ensure your CellProfiler repository is up to date. This is the only time we suggest pushing changes to the master branch of your repository. First, ensure you have no local changes by running `git status` on the master branch:

  ```
  $ git checkout master
  $ git status
  On branch master
  Your branch is up-to-date with 'origin/master'.
  nothing to commit, working tree clean
  ```

  Run the following commands to ensure your repository is up to date with the upstream repository:

  ```
  $ git fetch upstream
  $ git merge upstream/master
  $ git push origin master
  ```

  Your master branch is now updated to match ours. We recommend performing this step often, especially before creating a new branch.

## Contribution Workflow

In this section we provide a general outline of the process for submitting changes to CellProfiler. You'll learn how to create a branch where you'll make changes, add modified files and commit your changes to your branch, and create a pull request to submit your changes for review by CellProfiler maintainers.

1. Find the open [issue](https://github.com/CellProfiler/CellProfiler/issues) you want to resolve. Create a [new issue](https://github.com/CellProfiler/CellProfiler/issues/new) if necessary. Check out our guidelines for creating issues before submitting a new issue.

1. Create a branch using the issue number. We prefer branches which are named in the format "issues/ISSUE_NUMBER". This helps us track with issues are resolved and should be closed. Use the `git checkout` command to create a new branch and switch to it:

  ```
  $ git checkout -b issues/ISSUE_NUMBER
  ```

  Substitute `ISSUE_NUMBER` with the number of the issue you are going to resolve with your change. Each issue has a unique issue number which is displayed next to the issue title. An example might be, "Document which modules respect masks #2245". The issue number is 2245. You would create a branch named issues/2245.

1. Make changes on your branch. Use the `git add` command to add modified files and the `git commit` command to commit your changes:

  ```
  $ git add -u .
  $ git commit -m MESSAGE
  ```

  Substitute `MESSAGE` with an accurate description of your changes. Using our example issue from the previous step, a suitable commit message would be "Document modules which respect masks". Remember to include quotes (`"`) around your message.

  When adding new files, use the `git add` command to explicitly include them:

  ```
  $ git add FILENAME
  ```

  Substitute `FILENAME` with the name of the new file.

  You can add as many commits to your branch as necessary. We suggest each commit records a change with a single purpose. You should be able to describe a commit without using the word "and".

1. Push your local changes to your GitHub repository. Use the `git push` command to update your repository with your local changes:

  ```
  $ git push origin issues/ISSUE_NUMBER
  ```

  This step is required before you can make a pull request. Additionally, we recommend doing this often so you won't lose your work if you lose access to your workstation.

1. Create a pull request when you're ready to have your changes reviewed. Find your branch on your GitHub project page by navigating to:

  ```
  https://github.com/YOUR-USERNAME/CellProfiler/branches
  ```

  Substitute `YOUR-USERNAME` with your GitHub user name.

  Click "New Pull Request" to generate a pull request. Enter a suitable title in the "Title" field. Provide a description of your change in the "Leave a comment" field.

  Ensure you are creating a pull request on our project. Click the "compare across forks" link above the "Title" field. Ensure the "base fork" drop-down is set to "CellProfiler/CellProfiler" and the "base" branch drop-down is set to "master". See [Creating a pull request from a fork](https://help.github.com/articles/creating-a-pull-request-from-a-fork/) for more information.

  When you are ready, click "Create pull request".

1. Respond to feedback from maintainers. After a pull request is submitted it will be reviewed by CellProfiler maintainers. Be prepared to make revisions to your pull request before it is accepted.

1. Update your repository with your accepted change. Once accepted, your change is in the CellProfiler project's master branch. You can update your master branch by running the following commands:

  ```
  $ git checkout master
  $ git fetch upstream
  $ git merge upstream/master
  $ git push origin master
  ```

  Your master branch now includes your change. Congratulations on a successful contribution!

## Style Guide

We follow [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/), with the following recommendations:

### Imports

PEP 8 describes how to [group imports](https://www.python.org/dev/peps/pep-0008/#imports). Imports in each group should appear in alphabetical order. The PyCharm editor can do this automatically. Navigate to Code > Optimize Imports to automatically group and alphabetize imports, and remove unused imports.

We always prefer explicit imports. Please do not use import aliases such as `import numpy as np`.

## Instructions for Advanced Contributions

See the "Installation" subsection of our [Wiki](https://github.com/CellProfiler/CellProfiler/wiki) for instructions on installing CellProfiler from source on your operating system.

### Testing
We strongly encourage contributions with tests. We use [pytest](doc.pytest.org/en/latest/) as our test framework and have designed a set of test fixtures for unit testing CellProfiler modules. To execute the full test suite, run:

```
$ python setup.py test
```

You can run individual tests by providing the option `-a PATH_TO_TEST_FILE`.
