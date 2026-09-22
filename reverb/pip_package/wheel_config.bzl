"""Repository configuration for release and nightly wheel metadata."""

def _wheel_config_impl(ctx):
    name = ctx.getenv("WHEEL_NAME", "dm_reverb")
    wheel_type = ctx.getenv("ML_WHEEL_TYPE", "release")
    date = ctx.getenv("ML_WHEEL_BUILD_DATE", "")
    if wheel_type not in ["release", "nightly"]:
        fail("`ML_WHEEL_TYPE` must be `release` or `nightly`")
    if wheel_type == "nightly" and (len(date) != 8 or not date.isdigit()):
        fail("Nightly wheels require `ML_WHEEL_BUILD_DATE` in `YYYYMMDD` format")
    local_version = ctx.getenv("WHEEL_LOCAL_VERSION", "")
    if local_version:
        for segment in local_version.split("."):
            if not segment or any([
                c not in "abcdefghijklmnopqrstuvwxyz0123456789"
                for c in segment.elems()
            ]):
                fail("`WHEEL_LOCAL_VERSION` requires lowercase alphanumeric segments separated by dots")
    suffix = ".dev" + date if wheel_type == "nightly" else ""
    if local_version:
        suffix += "+" + local_version
    ctx.file("BUILD.bazel", "exports_files([\"config.bzl\"])\n")
    ctx.file("config.bzl", "WHEEL_NAME = %r\nWHEEL_VERSION_SUFFIX = %r\n" % (
        name,
        suffix,
    ))

wheel_config_repository = repository_rule(implementation = _wheel_config_impl)
