"""This module contains Bazel rules for interfacing with non-Bazel build systems."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_FOREIGN_CC_VERSION = "0.11.1"

def rules_foreign_cc_workspace():
    http_archive(
        name = "rules_foreign_cc",
        sha256 = "4b33d62cf109bcccf286b30ed7121129cc34cf4f4ed9d8a11f38d9108f40ba74",
        strip_prefix = "rules_foreign_cc-{}".format(RULES_FOREIGN_CC_VERSION),
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/{}.tar.gz".format(RULES_FOREIGN_CC_VERSION),
    )
