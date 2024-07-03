"""This module loads all third-party workspaces."""

load("//third_party/com_google_protobuf:workspace.bzl", "com_google_protobuf_workspace")
load("//third_party/rules_pkg:workspace.bzl", "rules_pkg_workspace")
load("//third_party/rules_proto:workspace.bzl", "rules_proto_workspace")
load("//third_party/rules_python:workspace.bzl", "rules_python_workspace")

def load_third_party_workspaces():
    com_google_protobuf_workspace()
    rules_pkg_workspace()
    rules_proto_workspace()
    rules_python_workspace()
