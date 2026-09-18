# Example ainow plugin tool.
#
# Drop this file (or a symlink) into ~/.config/ainow/tools.d/ to make the
# utc_now tool available in every ainow session without editing ainow.py.
#
# Plugins keep private/specialised tools in the gitignored config overlay so
# the public ainow repo stays generic.
import datetime


def _utc_now(format: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(format)


TOOL_SPEC = {
    "name": "utc_now",
    "description": "Return the current UTC time.",
    "schema": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "strftime format string (default ISO8601)",
            },
        },
    },
    "call": _utc_now,
    "requires_approval": False,
}
