## Image Processing
by Lukas Trišauskas, Thomas Nadin-Hepburn, Nicolas Suckling, John Merritt

## Workflow

The image-processing repository consists of three branches

1. Main: also known as the production branch, it contains live, working version of the app, you can only merge to it after a code review has been completed and no more revisions need to be made.
2. Development: the default branch with the latest features and bug fixes that haven't been merged to production yet.
3. Test: contains unit tests for all features, before any changes are merged to production, it must undergo testing.



## Standards

### Variables<br>
The naming convention that will be used for variables is `snake_case`.<br>

    ```
    my_variable = 0
    my_variable_second = 0
    ```
### Functions<br>
The naming convention that will be used for naming functions is `camelCase`.<br>

    ```
    def myFunction():
      ...
    ```

### Classes<br>
The naming convention that will be used for naming classes is `CamelCase`, unlike the function naming convention, for classes all words must start with a capital letter.<br>

    ```
    class MyClass():
      ...
    ```

