"""This module loads all third-party workspaces."""

load("//third_party/abseil-cpp:workspace.bzl", "abseil_cpp_workspace")
load("//third_party/com_google_protobuf:workspace.bzl", "com_google_protobuf_workspace")
load("//third_party/eigen:workspace.bzl", "eigen_workspace")
load("//third_party/rules_cc:workspace.bzl", "rules_cc_workspace")
load("//third_party/rules_pkg:workspace.bzl", "rules_pkg_workspace")
load("//third_party/rules_proto:workspace.bzl", "rules_proto_workspace")
load("//third_party/rules_python:workspace.bzl", "rules_python_workspace")

def load_third_party_workspaces():
    """Loads all third-party workspaces."""
    abseil_cpp_workspace()
    com_google_protobuf_workspace()
    eigen_workspace()
    rules_cc_workspace()
    rules_pkg_workspace()
    rules_proto_workspace()
    rules_python_workspace()
