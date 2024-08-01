load("@//third_party/openblas:workspace.bzl", "OPENBLAS_VERSION")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "openblas",
    generate_args = ["-DBUILD_SHARED_LIBS=ON"],
    lib_source = "@openblas//:all",
    out_shared_libs = [
        "libopenblas.so",
        "libopenblas.so.{}".format(OPENBLAS_VERSION.split(".", 1)[0]),
        "libopenblas.so.{}".format(OPENBLAS_VERSION.rsplit(".", 1)[0]),
    ],
)
