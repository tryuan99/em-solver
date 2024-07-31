load("@//third_party/gmsh:workspace.bzl", "GMSH_VERSION")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "gmsh",
    generate_args = ["-DENABLE_BUILD_SHARED=1"],
    lib_source = "@gmsh//:all",
    out_shared_libs = [
        "libgmsh.so",
        "libgmsh.so.{}".format(GMSH_VERSION.rsplit(".", 1)[0]),
        "libgmsh.so.{}".format(GMSH_VERSION),
    ],
)
