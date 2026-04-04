load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

GMSH_VERSION = "4.15.2"

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "gmsh",
    cache_entries = {
        "ENABLE_BUILD_SHARED": "1",
        "ENABLE_FLTK": "0",
    },
    lib_source = "@gmsh//:all",
    out_shared_libs = select({
        "@platforms//os:osx": [
            "libgmsh.dylib",
            "libgmsh.{}.dylib".format(GMSH_VERSION.rsplit(".", 1)[0]),
            "libgmsh.{}.dylib".format(GMSH_VERSION),
        ],
        "@platforms//os:linux": [
            "libgmsh.so",
            "libgmsh.so.{}".format(GMSH_VERSION.rsplit(".", 1)[0]),
            "libgmsh.so.{}".format(GMSH_VERSION),
        ],
        "//conditions:default": [],
    }),
    targets = ["shared"],
)
