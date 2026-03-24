def build_animagine_prompt(
    scene: str,
    objects: list[str] | None = None,
    mood: str | None = None,
    style: str | None = None,
) -> str:
    objects = objects or []

    base_tags = [
        "masterpiece",
        "best quality",
        "very aesthetic",
        "anime coloring",
        "cinematic lighting",
        "detailed background",
    ]

    if style:
        base_tags.append(style)

    if scene:
        base_tags.append(scene)

    if mood:
        base_tags.append(mood)

    base_tags.extend(objects)

    return ", ".join(base_tags)


def default_negative_prompt() -> str:
    return (
        "low quality, worst quality, blurry, bad anatomy, bad hands, "
        "extra fingers, text, watermark, logo, jpeg artifacts"
    )