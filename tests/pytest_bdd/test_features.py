"""Test features."""

from pytest_bdd import scenarios

# Load all the .feature files in the features directory and turn their scenarios into tests.
scenarios("features")
