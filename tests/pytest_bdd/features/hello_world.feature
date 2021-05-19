Feature: Message sending

    Scenario: Send a message
        Given I am logged in as user "neo"
        When I send a message "there is no spoon" to user "morpheus"
        Then The user "morpheus" should receive a message
