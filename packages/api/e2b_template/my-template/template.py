from e2b import AsyncTemplate

template = (
    AsyncTemplate()
    .from_image("e2bdev/base")
    .run_cmd("echo Hello World E2B!")
)