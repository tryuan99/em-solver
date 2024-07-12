"""This module contains Bazel rules for C++."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_CC_VERSION = "0.0.10-rc1"

def rules_cc_workspace():
    http_archive(
        name = "rules_cc",
        sha256 = "3baa0e51db60d24df85ebc82f2271df539533a21bdf0368705e79855d82ec4bd",
        strip_prefix = "rules_cc-{}".format(RULES_CC_VERSION),
        url = "https://github.com/bazelbuild/rules_cc/archive/refs/tags/{}.tar.gz".format(RULES_CC_VERSION),
    )
