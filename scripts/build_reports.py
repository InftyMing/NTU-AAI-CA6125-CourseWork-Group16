from __future__ import annotations

from feedback_ell.reporting import (
    generate_chinese_report,
    generate_english_report,
    generate_video_materials,
)


def main() -> None:
    outputs = [
        generate_english_report(),
        generate_chinese_report(),
        *generate_video_materials(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
