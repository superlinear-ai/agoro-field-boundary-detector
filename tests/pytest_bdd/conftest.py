"""Step implementations."""

from typing import Any, Dict

import pytest
from pytest_bdd import given, parsers, then, when

User = Dict[str, Any]
UserDatabase = Dict[str, Any]


@pytest.fixture()
def user_database() -> UserDatabase:
    """Create an empty user database that is reset for every test."""
    # This is a pytest fixture that can be supplied as an argument to other steps.
    return {}


@given(parsers.re('I am logged in as user "(?P<user_id>[^"]+)"'), target_fixture="me")
def me(user_id: str, user_database: UserDatabase) -> User:
    """Log in."""
    # Create user and register it in the user database.
    user = {"user_id": user_id, "logged_in": True, "inbox": []}
    user_database[user_id] = user
    # Note that @given returns a fixture value for use in other steps!
    return user


@when(parsers.re('I send a message "(?P<message>[^"]+)" to user "(?P<recipient>[^"]+)"'))
def step_send_message(me: User, message: str, recipient: str, user_database: UserDatabase) -> None:
    """Send a message to a user."""
    # Verify that I am logged in.
    assert me["logged_in"], "You must be logged in before you can send a message"
    # Send the message to the recipient.
    user_database[recipient] = {"logged_in": False, "inbox": [message]}


@then(parsers.re('The user "(?P<user_id>[^"]+)" should receive a message'))
def step_check_message(user_id: str, user_database: UserDatabase) -> None:
    """Check if the given user received a message."""
    # Verify that the user received a message.
    assert user_database[user_id]["inbox"], f"The user {user_id} did not receive a message"
