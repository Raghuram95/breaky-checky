"""The sign-in and access guard every protected endpoint sits behind.

The service decides on its own account what each role may reach, so hiding a page in the
dashboard is a convenience, never the protection. The area of a request is read from the route
it matched (for example /clearances/{assessment_id}/decision), and the guard runs before the
endpoint, so a role without the area is refused whether or not the thing it asks for exists.
"""

from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from config import config
from persistence.database import session_factory
from persistence.models import UserAccount
from persistence.repository import fetch_user_account


def area_of_route(method: str, route_path: str) -> str:
    """The area a request belongs to, decided from the path of the route it matched."""
    if method == "POST" and route_path.endswith(config.DECISION_PATH_ENDING):
        return config.AREA_DECISION
    for path_start, area in config.AREA_BY_PATH_START.items():
        if route_path == path_start or route_path.startswith(path_start + "/"):
            return area
    return None


def authorize_request(
    request: Request,
    session_token: Annotated[str | None, Header(alias=config.SESSION_TOKEN_HEADER)] = None,
) -> dict:
    """Decide who is asking and whether their role reaches the area; return their account."""
    account = None
    if session_token:
        with session_factory() as session:
            account = fetch_user_account(session, UserAccount.session_token == session_token)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=config.SIGN_IN_REQUIRED_MESSAGE
        )

    route = request.scope.get("route")
    area = area_of_route(request.method, getattr(route, "path", request.url.path))
    if area not in config.ROLE_AREA_ACCESS[account["role"]]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=config.AREA_REFUSED_MESSAGE.format(role=account["role"], area=area),
        )
    return account
