load("@pip_deps//:requirements.bzl", "install_deps")
load("@python3_11//:defs.bzl", "interpreter")

def load_pip_dependencies():
    install_deps(
        python_interpreter_target = interpreter,
    )
