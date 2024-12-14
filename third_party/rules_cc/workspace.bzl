"""This module contains Bazel rules for C++."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_CC_VERSION = "0.0.17"

def rules_cc_workspace():
    http_archive(
        name = "rules_cc",
        sha256 = "abc605dd850f813bb37004b77db20106a19311a96b2da1c92b789da529d28fe1",
        strip_prefix = "rules_cc-{}".format(RULES_CC_VERSION),
        url = "https://github.com/bazelbuild/rules_cc/archive/refs/tags/{}.tar.gz".format(RULES_CC_VERSION),
    )
