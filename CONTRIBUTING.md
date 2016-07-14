# Contribution Guidelines
In general try to match the existing coding style as close as possible.
The following paragraphs describe tools which are used to enforces certain guidelines.

## Linting

Run cpplint using `make lint` to ensure that there are no linting errors.

## Code format

Run [clang-format](http://clang.llvm.org/docs/ClangFormat.html) on all `.h` and `.cpp` files. Also use it on `.cu` files,
but fix the kernel calls (`<<< >>>` are split by the formatter).
The formatting settings are specified in [.clang-format](.clang-format).

## Tests

Make sure that as much code as possible is covered by tests and new changes don't break existing tests by
running `make unit`. Coverage information can be generatione with the `COV` build type (run `cmake -DCMAKE_BUILD_TYPE=COV`).
