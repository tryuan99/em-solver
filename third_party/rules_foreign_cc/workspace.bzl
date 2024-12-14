"""This module contains Bazel rules for interfacing with non-Bazel build systems."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_FOREIGN_CC_VERSION = "0.13.0"

def rules_foreign_cc_workspace():
    http_archive(
        name = "rules_foreign_cc",
        sha256 = "8e5605dc2d16a4229cb8fbe398514b10528553ed4f5f7737b663fdd92f48e1c2",
        strip_prefix = "rules_foreign_cc-{}".format(RULES_FOREIGN_CC_VERSION),
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/{}.tar.gz".format(RULES_FOREIGN_CC_VERSION),
    )
