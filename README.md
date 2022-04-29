## Image Processing
by Lukas Trišauskas, Thomas Nadin-Hepburn, Nicolas Suckling, John Merritt

## Workflow

[❗] The image-processing repository consists of three branches

- Main: also known as the production branch, it contains live, working version of the app, you can only merge to it after a code review has been completed and no more revisions need to be made.
- Development: the default branch with the latest features and bug fixes that haven't been merged to production yet.
- Test: contains unit tests for all features, before any changes are merged to production, it must undergo testing.

## Temporary Branches

[❗] When adding a new feature, trying to fix a bug, or adding a hotfix, always create a new branch and use the following naming conventions for branch names<br>

adding new feature: <br>
    
    feature_name_of_the_feature
    
bug fix: <br>

    bug_name_of_the_bug

hotfix: <br>

    hotfix_name_of_the_hotfix
    
Hotfix
> Refered to as a software patch that is applied to "hot" systems: those which are live, currently running, and in production status, rather than development status.
> It implies that the change may bave been made quickly and outside normal development and testing processes.
    
## Creating a new branch

For example, if you wanted to add a new feature, you would create a new branch and name it e.g.

    git checkout -b feature_name_of_the_feature

Another example, if you wanted to add a bug fix, I would create a new branch and name it e.g.

    git checkout -b -bug_name_of_the_bug


## Standards

### Variables<br>
The naming convention that will be used for variables is `snake_case`.<br>

    my_variable = 0
    my_variable_second = 0

### Functions<br>
The naming convention that will be used for naming functions is `camelCase`.<br>

    def myFunction():
      ...

### Classes<br>
The naming convention that will be used for naming classes is `CamelCase`, unlike the function naming convention, for classes all words must start with a capital letter.<br>

    class MyClass():
      ...

