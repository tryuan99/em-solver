"""This module contains Bazel rules for Protobuf."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_PROTO_VERSION = "6.0.2"

def rules_proto_workspace():
    http_archive(
        name = "rules_proto",
        sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
        strip_prefix = "rules_proto-{}".format(RULES_PROTO_VERSION),
        url = "https://github.com/bazelbuild/rules_proto/archive/refs/tags/{}.tar.gz".format(RULES_PROTO_VERSION),
    )
