load("@rules_python//python:repositories.bzl", "python_register_toolchains")

def register_python_toolchain():
    python_register_toolchains(
        name = "python3_11",
        python_version = "3.11",
    )
